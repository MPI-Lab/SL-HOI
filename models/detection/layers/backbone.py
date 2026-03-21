"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

import math
import numpy as np
from typing import List, Optional, Union

from util.misc import NestedTensor
from .position_encoding import build_position_encoding


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DINOBackbone(nn.Module):
    def __init__(
        self,
        backbone_model: nn.Module,
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        layers_to_use: Union[int, List] = 1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.blocks_to_train = blocks_to_train
        self.patch_size = self.backbone.patch_size
        self.use_layernorm = use_layernorm

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        self.strides = [self.backbone.patch_size]

        # get embed_dim for each intermediate output
        n_all_layers = self.backbone.n_blocks
        blocks_to_take = (
            range(n_all_layers - layers_to_use, n_all_layers) if isinstance(layers_to_use, int) else layers_to_use
        )

        # if models do not define embed_dims, repeat embed_dim n_blocks times
        embed_dims = getattr(self.backbone, "embed_dims", [self.backbone.embed_dim] * self.backbone.n_blocks)
        embed_dims = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]

        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm2D(embed_dim) for embed_dim in embed_dims])

        self.num_channels = [sum(embed_dims)]
        self.layers_to_use = layers_to_use

    def forward(self, tensor_list: NestedTensor, return_tokens: bool = False):
        B, _, h, w = tensor_list.tensors.shape
        
        intermediate_outputs = self.backbone.get_intermediate_layers(
            tensor_list.tensors, n=self.layers_to_use, 
            return_class_token=True, return_extra_tokens=True
        )
        xs = [intermediate[0].reshape(B, h // self.patch_size, w // self.patch_size, -1) \
              .permute(0, 3, 1, 2).contiguous() for intermediate in intermediate_outputs]

        if self.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        xs = [torch.cat(xs, axis=1)]

        out: list[NestedTensor] = []
        for x in xs:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        
        # return also semantic tokens from the vision head
        if return_tokens:
            class_token = intermediate_outputs[-1][1]
            patch_tokens = intermediate_outputs[-1][0]
            register_tokens = intermediate_outputs[-1][2]
            return out, torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1)

        return out


class WindowsWrapper(torch.nn.Module):
    """
    This wrapper will take an input (NestedTensor) at size (h, w) and split it
    in `N = n_windows_h * n_windows_w` equally sized windows (the bottom and right windows might
    be a little bit smaller), with sizes that are multiples of the patch size (as the input should be).

    Then, the input will be resized at the size of the top left window (h / n_windows_h, w / n_windows_w).
    This resized input, plus the N windows, will be passed through the backbone.
    Then, the features of the resized input will be resized to the original input size, while the
    features of the windows will be concatenated side by side to reconstruct a feature map also
    corresponding to the original image's size.

    Finally, both the features from the windows and from the resized images are stacked.
    Compared to the output of the backbone of size [B, C, H, W], the output here is [B, 2 * C, H, W]
    """

    def __init__(self, backbone, n_windows_w, n_windows_h, patch_size, global_size=None):
        # Assuming image size is divisible by patch_size
        super().__init__()
        self._backbone = backbone
        self._n_windows_w = n_windows_w
        self._n_windows_h = n_windows_h
        self._patch_size = patch_size
        self._global_size = global_size
        self.strides = backbone.strides
        self.num_channels = [el * 2 for el in backbone.num_channels]  # resized + windows

    def forward(self, tensor_list: NestedTensor, return_tokens: bool = False):
        tensors = tensor_list.tensors
        original_h, original_w = tensors.shape[2], tensors.shape[3]
        # Get height and width of the windows, such that it is a multiple of the patch size
        window_h = math.ceil((original_h // self._n_windows_h) / self._patch_size) * self._patch_size
        window_w = math.ceil((original_w // self._n_windows_w) / self._patch_size) * self._patch_size
        all_h = [window_h] * (self._n_windows_h - 1) + [original_h - window_h * (self._n_windows_h - 1)]
        all_w = [window_w] * (self._n_windows_w - 1) + [original_w - window_w * (self._n_windows_w - 1)]
        all_h_cumsum = [0] + list(np.cumsum(all_h))
        all_w_cumsum = [0] + list(np.cumsum(all_w))
        window_patch_features = [[0 for _ in range(self._n_windows_w)] for _ in range(self._n_windows_h)]

        for ih in range(self._n_windows_h):
            for iw in range(self._n_windows_w):
                window_tensor = v2.functional.crop(
                    tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_mask = v2.functional.crop(
                    tensor_list.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_patch_features[ih][iw] = self._backbone(NestedTensor(tensors=window_tensor, mask=window_mask), 
                                                               return_tokens=False)[0]

        window_tensors = torch.cat(
            [
                torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1)  # type: ignore
                for ih in range(len(window_patch_features))
            ],
            dim=-2,
        )
        # Also compute the global features in a "preferential" setting, of lower resolution
        if self._global_size is None:
            resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
        else:
            resized_global_tensor = v2.functional.resize(tensors, size=self._global_size)
        global_features, tokens = self._backbone(
            NestedTensor(tensors=resized_global_tensor, mask=tensor_list.mask),
            return_tokens=return_tokens # return semantic tokens only for the global branch
        )  # mask is not used

        concat_tensors = torch.cat(
            [v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:]), window_tensors], dim=1
        )
        global_mask = F.interpolate(tensor_list.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        out = [NestedTensor(tensors=concat_tensors, mask=global_mask)]
        return out, tokens


class BackboneWithPositionEncoding(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor, return_tokens: bool = False):
        out: List[NestedTensor]
        tokens: torch.Tensor
        out, tokens = self[0](tensor_list, return_tokens=return_tokens)
        out = list(out)  # convert to list for consistency
        pos = [self[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(out)]

        # return also semantic tokens from the vision head
        if return_tokens:
            return out, pos, tokens
        return out, pos


def build_backbone(cfg, backbone_model: nn.Module = None):
    position_embedding = build_position_encoding(cfg)
    
    layers_to_use = cfg.MODEL.DINO.LAYERS_TO_USE
    if not layers_to_use:
        # e.g. [5, 11, 17, 23] for a backbone with 24 blocks
        layers_to_use = [m * backbone_model.n_blocks // 4 - 1 for m in range(1, 5)]

    backbone = DINOBackbone(backbone_model, train_backbone=False, blocks_to_train=None, 
        layers_to_use=layers_to_use, use_layernorm=False)
    
    if cfg.MODEL.DINO.N_WINDOWS_SQRT > 0:
        global_size = (224, 224) if cfg.INPUT.DATASET_FILE == "swig" else None
        backbone = WindowsWrapper(
            backbone, n_windows_w=cfg.MODEL.DINO.N_WINDOWS_SQRT, 
            n_windows_h=cfg.MODEL.DINO.N_WINDOWS_SQRT, 
            patch_size=backbone.patch_size,
            global_size=global_size
        )
    return BackboneWithPositionEncoding(backbone, position_embedding)
