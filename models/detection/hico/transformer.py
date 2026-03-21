from typing import Optional

import torch
from torch import nn, Tensor

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

        instance_decoder_layer = DualSourceDecoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, use_gate=True)
        instance_decoder_norm = nn.LayerNorm(d_model)
        self.instance_decoder = DualSourceDecoder(instance_decoder_layer, 
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

    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, semantic_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]
        mask = mask.flatten(1)
        semantic_embed = semantic_embed.permute(1, 0, 2)  # [S, B, C]

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        query_embed_o = (query_embed_o + pos_guided_embed).unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = (query_embed_h + pos_guided_embed).unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)
        ins_tgt = torch.zeros_like(ins_query_embed)
        ins_hs = self.instance_decoder(
            ins_tgt, memory, memory_key_padding_mask=mask,
            pos=pos_embed, query_pos=ins_query_embed, semantic_embed=semantic_embed
        )

        h_hs = ins_hs[:, :num_queries]
        o_hs = ins_hs[:, num_queries:]

        h_hs = h_hs.permute(0, 2, 1, 3)
        o_hs = o_hs.permute(0, 2, 1, 3)

        return h_hs, o_hs


class DualSourceDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                semantic_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        for i, layer in enumerate(self.layers):
            if query_pos is None:
                this_query_pos = None
            elif len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, memory_key_padding_mask, 
                           pos, this_query_pos, semantic_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class DualSourceDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", use_gate=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.use_gate = use_gate
        if use_gate:
            self.gate_linear = nn.Linear(d_model, 2)

        self.activation = _get_activation_fn(activation)

    def ffn(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm3(src + self.dropout3(src2))
        return src

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                semantic_embed: Optional[Tensor] = None):
        tgt2 = self.self_attn(query=self.with_pos_embed(tgt, query_pos), 
                              key=self.with_pos_embed(tgt, query_pos), value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), 
                                   key=semantic_embed, value=semantic_embed)[0]
        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, key_padding_mask=memory_key_padding_mask)[0]
        
        if self.use_gate:
            gate_weights = self.gate_linear(tgt).softmax(dim=-1)
            gate_w2= gate_weights[..., 0].unsqueeze(-1)     # -> [N_q, B, 1]
            gate_w3 = gate_weights[..., 1].unsqueeze(-1)    # -> [N_q, B, 1]
            tgt2 = gate_w2 * tgt2 + gate_w3 * tgt3
        else:
            tgt2 = tgt2 + tgt3
        
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.ffn(tgt)
        return tgt


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
