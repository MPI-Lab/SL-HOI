from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.detection.layers import (TransformerEncoder, TransformerEncoderLayer, 
                                     TransformerDecoder, TransformerDecoderLayer,
                                     _get_clones, _get_activation_fn)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_ins_dec_layers=3, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation)
        instance_decoder_norm = nn.LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_ins_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed

        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)

        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)
        ins_tgt = torch.zeros_like(ins_query_embed)
        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        h_hs = ins_hs[:, :num_queries]
        o_hs = ins_hs[:, num_queries:]

        h_hs = h_hs.permute(0, 2, 1, 3)
        o_hs = o_hs.permute(0, 2, 1, 3)

        return h_hs, o_hs


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


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.HOI.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_ins_dec_layers=cfg.MODEL.TRANSFORMER.INS_DEC_LAYERS,
        return_intermediate_dec=cfg.MODEL.HOI.DEEP_SUPERVISION
    )
