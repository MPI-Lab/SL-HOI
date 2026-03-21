import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from models.detection.layers import build_backbone, TransformerDecoder, TransformerDecoderLayer
from .loss import SetCriterionHOI, PostProcessHOITriplet
from .matcher import build_matcher
from .transformer import build_transformer, MLP


class HOIDetector(nn.Module):
    def __init__(self, cfg, backbone, head_model, transformer):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.dtype = torch.float32    # using accelerate to handle mixed precision: bf16

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
        self.box_score_embed = MLP(hidden_dim * 2, hidden_dim, 1, 3)

        self.aux_loss = cfg.MODEL.HOI.DEEP_SUPERVISION
        self.hidden_dim = hidden_dim
        self.dino_embed_dim = dino_embed_dim
        self.cfg = cfg
        self.reset_parameters()
        self.freeze_dino()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.cfg.MODEL.DINO.FREEZE:
                self.backbone.eval()
        return self

    def freeze_dino(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head_model.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embed.weight)

    @staticmethod
    def _create_attention_mask(batch_size, num_heads, total_tokens, query_range, device, dtype):
        mask = torch.zeros(
            batch_size, num_heads, total_tokens, total_tokens, 
            device=device, dtype=dtype
        )
        query_start, query_end = query_range
        # cls, registers, patches => queries: MASKED
        mask[:, :, :query_start, query_start:query_end] = float('-inf')
        mask[:, :, query_end:, query_start:query_end] = float('-inf')
        return mask

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
        query_range = (query_start, query_end)

        # attention_mask = self._create_attention_mask(
        #     batch_size=bs,
        #     num_heads=self.head_model.blocks[0].attn.num_heads,
        #     total_tokens=query_tokens.shape[1],
        #     query_range=query_range,
        #     device=query_tokens.device,
        #     dtype=query_tokens.dtype
        # )

        token_outputs = self.head_model(
            query_tokens,
            # attn_mask=attention_mask,
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

    def forward(self, samples: NestedTensor, text_embeddings: torch.Tensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        src, mask, pos, tokens = self.get_backbone_features(samples)

        if self.training:
            src = src.clone().detach().requires_grad_(True)
            pos = pos.clone().detach().requires_grad_(True)
            tokens = tokens.clone().detach().requires_grad_(True)

        h_hs, o_hs = self.transformer(
            self.input_proj(src), mask, 
            self.query_embed_h.weight,
            self.query_embed_o.weight,
            self.pos_guided_embed.weight,
            pos
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
        outputs_coord = torch.cat([outputs_sub_coord, outputs_obj_coord], dim=-1)

        output_box_scores = self.box_score_embed(torch.cat([h_hs, o_hs], dim=-1))

        logit_scale = self.logit_scale.exp()
        inter_hs = self.hoi_class_fc(inter_hs)
        inter_hs = F.normalize(inter_hs, dim=-1)
        logits_per_hoi = logit_scale * inter_hs @ text_embeddings.t()

        out = {'logits_per_hoi': logits_per_hoi[-1], 'box_scores': output_box_scores[-1],
               'pred_boxes': outputs_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(logits_per_hoi, 
                                                            output_box_scores,
                                                            outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, logits_per_hoi, output_box_scores, outputs_coord):

        num_dec_layers = self.num_dec_layers
        aux_outputs = {'logits_per_hoi': logits_per_hoi[-num_dec_layers: -1],
                       'box_scores': output_box_scores[-num_dec_layers: -1],
                       'pred_boxes': outputs_coord[-num_dec_layers: -1]}
        outputs_auxes = []
        for i in range(num_dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


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

    weight_dict = {
        'loss_ce': cfg.LOSS.COEFFS.CLASS_LOSS_COEF, 
        'loss_bbox': cfg.LOSS.COEFFS.BBOX_LOSS_COEF,
        'loss_giou': cfg.LOSS.COEFFS.GIOU_LOSS_COEF,
        'loss_conf': cfg.LOSS.COEFFS.CONF_LOSS_COEF,
    }
    losses = ['labels', 'boxes', "confidences"]

    if cfg.MODEL.HOI.DEEP_SUPERVISION:
        num_dec_layers = model.num_dec_layers
        aux_weight_dict = {}
        for i in range(num_dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    matcher = build_matcher(cfg)
    criterion = SetCriterionHOI(matcher=matcher, weight_dict=weight_dict, 
                                eos_coef=cfg.LOSS.COEF_CE_LOSS_OBJ, losses=losses)
    criterion.to(device)
    postprocessors = {"hoi": PostProcessHOITriplet(cfg.EVAL.TEST_SCORE_THRESH)}

    return model, criterion, postprocessors
