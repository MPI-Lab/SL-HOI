import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from models.detection.layers import build_backbone, TransformerDecoderLayer, TransformerDecoder
from .loss import SetCriterionHOI, PostProcessHOITriplet
from .matcher import build_matcher
from .transformer import build_transformer
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class HOIDetector(nn.Module):
    def __init__(self, cfg, backbone, head_model, transformer):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        self.num_queries = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.backbone = backbone
        self.head_model = head_model
        self.num_register_tokens = cfg.MODEL.DINO.N_STORAGE_TOKENS
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)
        self.query_embed_o = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)
        self.pos_guided_embed = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)

        self.input_proj = nn.Sequential(nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                                        nn.GroupNorm(32, hidden_dim),)

        dino_embed_dim = cfg.MODEL.DINO.EMBED_DIM
        self.semantic_proj = nn.Sequential(
            nn.Linear(dino_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, dino_embed_dim),
            nn.LayerNorm(dino_embed_dim)
        )
        fusion_decoder_layer = TransformerDecoderLayer(
            dino_embed_dim, 16, dino_embed_dim * 4,
            dropout=0.1, activation="relu"
        )
        num_fus_dec_layers = 1
        fusion_decoder_norm = nn.LayerNorm(dino_embed_dim)
        self.fusion_decoder = TransformerDecoder(
            fusion_decoder_layer,
            num_fus_dec_layers,
            fusion_decoder_norm,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.hoi_class_fc = nn.Sequential(
            nn.Linear(dino_embed_dim, dino_embed_dim * 2),
            nn.LayerNorm(dino_embed_dim * 2)
        )
        self.num_dec_layers = cfg.MODEL.TRANSFORMER.INS_DEC_LAYERS

        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.dino_embed_dim = cfg.MODEL.DINO.EMBED_DIM
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        num_obj_classes = cfg.INPUT.NUM_OBJ_CLASSES

        classifier_train_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.TRAIN, map_location="cpu", weights_only=True)
        classifier_eval_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.EVAL, map_location="cpu", weights_only=True)

        if cfg.MODEL.DINO.WITH_DINO_LABEL:
            self.hoi_class_fc = nn.Sequential(
                nn.Linear(cfg.MODEL.DINO.EMBED_DIM, 2*cfg.MODEL.DINO.EMBED_DIM),
                nn.LayerNorm(2*cfg.MODEL.DINO.EMBED_DIM),
            )
            hoi_embedding_train = classifier_train_weights["hoi_embedding_train"]
            self.visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, hoi_embedding_train.shape[0], bias=False)
            with torch.no_grad():
                self.visual_projection.weight.copy_(hoi_embedding_train)
            if cfg.INPUT.DATASET_FILE == "hico" and cfg.ZERO_SHOT.TYPE != "default":
                hoi_embedding_eval = classifier_eval_weights["hoi_embedding_eval"]
                self.eval_visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, hoi_embedding_eval.shape[0], bias=False)
                with torch.no_grad():
                    self.eval_visual_projection.weight.copy_(classifier_eval_weights["hoi_embedding_eval"])
        else:
            raise ValueError("Please use DINO.txt label for HOI classification!")

        if cfg.MODEL.DINO.WITH_OBJ_DINO_LABEL:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, 2*cfg.MODEL.DINO.EMBED_DIM),
                nn.LayerNorm(2*cfg.MODEL.DINO.EMBED_DIM),
            )
            obj_embedding = classifier_eval_weights["obj_text_embedding_eval"]
            self.obj_visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, num_obj_classes + 1)
            with torch.no_grad():
                self.obj_visual_projection.weight.copy_(obj_embedding)
        else:
            raise ValueError("Please use DINO.txt label for object classification!")

        self.aux_loss = cfg.MODEL.HOI.DEEP_SUPERVISION
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.reset_parameters()
        self.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.cfg.MODEL.DINO.FREEZE:
                self.backbone.eval()
        return self

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head_model.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False
        for p in self.obj_visual_projection.parameters():
            p.requires_grad = False
        if self.cfg.ZERO_SHOT.TYPE != "default":
            for p in self.eval_visual_projection.parameters():
                p.requires_grad = False

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embed.weight)

    @torch.inference_mode()
    def get_semantics(self, tokens):
        image_tokens = self.head_model(tokens)
        class_token = image_tokens[:, 0]
        patch_tokens = image_tokens[:, self.num_register_tokens + 1 :]
        semantics = torch.cat([
            class_token.unsqueeze(1),
            torch.mean(patch_tokens, dim=1, keepdim=True),
            patch_tokens
        ], dim=1)
        return semantics

    def get_query_outputs(self, tokens, inter_query, return_token_outputs=False):
        bs, num_queries, embed_dim = inter_query.shape
        assert num_queries == self.num_queries, f"num_queries should be {self.num_queries}, but got {num_queries}"

        class_and_register_tokens = tokens[:, :self.num_register_tokens + 1, :]
        patch_tokens = tokens[:, self.num_register_tokens + 1:, :]
        query_tokens = torch.cat([
            class_and_register_tokens,
            inter_query,
            patch_tokens
        ], dim=1)

        query_start = self.num_register_tokens + 1
        query_end = query_start + num_queries

        token_outputs = self.head_model(
            query_tokens,
        )
        query_tokens = token_outputs[:, query_start:query_end, :]
        if return_token_outputs:
            class_token, patch_tokens = token_outputs[:, 0], token_outputs[:, query_end:]
            token_outputs = torch.cat((
                class_token.unsqueeze(1), 
                torch.mean(patch_tokens, dim=1, keepdim=True), 
                patch_tokens), dim=1
            )
            return query_tokens, token_outputs
        else:
            return query_tokens

    @torch.inference_mode()
    def get_backbone_features(self, samples):
        features, pos, tokens = self.backbone(samples, return_tokens=True)
        src, mask = features[-1].decompose()
        assert mask is not None
        return src, mask, pos[-1], tokens
    
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        src, mask, pos, tokens = self.get_backbone_features(samples)
        # for object detection, we do not need gradients for semantics
        semantics = self.get_semantics(tokens)

        if self.training:
            src = src.clone().detach().requires_grad_(True)
            pos = pos.clone().detach().requires_grad_(True)
            tokens = tokens.clone().detach().requires_grad_(True)
            semantics = semantics.clone().detach().requires_grad_(True)

        h_hs, o_hs = self.transformer(
            self.input_proj(src), mask,
            self.query_embed_h.weight,
            self.query_embed_o.weight,
            self.pos_guided_embed.weight,
            pos, self.semantic_proj(semantics)
        )

        inter_query = (h_hs + o_hs) / 2.0

        inter_hs_list = []
        for i in range(self.num_dec_layers):
            if i == self.num_dec_layers - 1: # return token_outputs only for the last layer
                query_tokens, token_outputs = self.get_query_outputs(
                    tokens, 
                    self.query_proj(inter_query[i]),
                    return_token_outputs=True
                )
            else:
                query_tokens = self.get_query_outputs(
                    tokens, 
                    self.query_proj(inter_query[i])
                )
            inter_hs_list.append(query_tokens)
        inter_hs = torch.stack(inter_hs_list, dim=0)

        # fusion inter_hs with token_outputs via fusion_decoder
        fused_inter_hs_list = []
        for i in range(self.num_dec_layers):
            fused_inter_hs_list.append(
                self.fusion_decoder(inter_hs[i].transpose(0, 1), token_outputs.transpose(0, 1))
            )
        inter_hs = torch.stack(fused_inter_hs_list, dim=0).transpose(1, 2)

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.cfg.MODEL.DINO.WITH_OBJ_DINO_LABEL:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            raise ValueError("Please use DINO.txt label for object classification!")

        if self.cfg.MODEL.DINO.WITH_DINO_LABEL:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            if self.cfg.INPUT.DATASET_FILE == "hico" and self.cfg.ZERO_SHOT.TYPE != "default" and (self.cfg.RUNTIME.EVAL or not self.training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
        else:
            raise ValueError("Please use DINO.txt label for HOI classification!")

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord):

        num_dec_layers = self.num_dec_layers
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-num_dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-num_dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-num_dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-num_dec_layers: -1]}
        outputs_auxes = []
        for i in range(num_dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detector(cfg, is_fresh_train: bool = True):
    device = torch.device(cfg.RUNTIME.DEVICE)

    if is_fresh_train:
        # only load pretrained weights for fresh training, not for evaluation or resumed training
        dinotxt_weights = cfg.MODEL.DINO.DINOTXT_WEIGHTS
        backbone_weights = cfg.MODEL.DINO.BACKBONE_WEIGHTS
    else:
        dinotxt_weights = None
        backbone_weights = None
    
    backbone_model, head_model = dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=is_fresh_train,
        dinotxt_weights=dinotxt_weights,
        backbone_weights=backbone_weights,
        bpe_path_or_url=cfg.MODEL.DINO.BPE_PATH_OR_URL,
        return_seperate_vision_tower=True
    )   # keep only the vision model
    backbone = build_backbone(cfg, backbone_model=backbone_model)

    transformer = build_transformer(cfg)
    model = HOIDetector(cfg, backbone, head_model, transformer)

    weight_dict = {}
    losses = ["hoi_labels", "obj_labels", "sub_obj_boxes"]
    weight_dict["loss_hoi_labels"] = cfg.LOSS.COEFFS.HOI_LOSS_COEF
    weight_dict["loss_obj_ce"] = cfg.LOSS.COEFFS.OBJ_LOSS_COEF
    weight_dict["loss_sub_bbox"] = cfg.LOSS.COEFFS.BBOX_LOSS_COEF
    weight_dict["loss_obj_bbox"] = cfg.LOSS.COEFFS.BBOX_LOSS_COEF
    weight_dict["loss_sub_giou"] = cfg.LOSS.COEFFS.GIOU_LOSS_COEF
    weight_dict["loss_obj_giou"] = cfg.LOSS.COEFFS.GIOU_LOSS_COEF

    if cfg.MODEL.HOI.DEEP_SUPERVISION:
        num_dec_layers = model.num_dec_layers
        aux_weight_dict = {}
        for i in range(num_dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    matcher = build_matcher(cfg)
    criterion = SetCriterionHOI(cfg, matcher=matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {"hoi": PostProcessHOITriplet(cfg)}

    return model, criterion, postprocessors
