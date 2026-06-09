# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from ..registry import REGRESSORS, get_function
from ..layers import PatchEmbed
from ..layers.block import Block
from ..layers.mlp import Mlp, CamMlp
from ..layers.rope import (
    RotaryPositionEmbedding2D,
    RotaryPositionEmbedding3D,
    PositionGetter,
    PositionGetter3D,
)
from ..layers.swiglu_ffn import SwiGLUFFNFused
from ..layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from ..pose_utils import create_attn_mask

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        rope_type="2d",  # "2d": (y,x) | "3d": (t,y,x)
        init_values=0.01,
        use_checkpoint=False,
        use_reentrant=True,
        use_chunkwise_checkpoint=False,
        use_cam_emb=False,
        use_cam_token=False,
        cam_emb_layer=[-5, None, 1],
        cam_embed_cfg=False,
        intermediate_layer_idx=[4, 11, 17, 23],  # used by the DPT head
        low_vram=False,
        gate_attn=None,  # "headwise", "elementwise", "per_token_headwise", or None/False
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            use_checkpoint=use_checkpoint,
            use_reentrant=use_reentrant,
            use_chunkwise_checkpoint=use_chunkwise_checkpoint,
            gate_attn=gate_attn,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope_type = rope_type if rope_freq > 0 else None
        if rope_freq <= 0:
            self.rope = None
            self.position_getter = None
        elif rope_type == "3d":
            self.rope = RotaryPositionEmbedding3D(frequency=rope_freq)
            self.position_getter = PositionGetter3D()
        else:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self.position_getter = PositionGetter()

        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.use_chunkwise_checkpoint = use_chunkwise_checkpoint

        # Select the layers to use camera embedding
        cam_emb_inds = range(depth)[cam_emb_layer[0]:cam_emb_layer[1]:cam_emb_layer[2]]
        cam_emb_list = [
            True if use_cam_emb and i in cam_emb_inds and not use_cam_token else False
                for i in range(depth)
        ]

        # Maybe create a camera MLP
        if use_cam_emb:
            self.cam_mlp = REGRESSORS.build(cam_embed_cfg)
        else:
            self.cam_mlp = None

        # Some bookkeepings
        self.use_cam_emb = use_cam_emb
        self.use_cam_token = use_cam_token
        self.cam_embed_cfg = cam_embed_cfg
        self.low_vram = low_vram

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    use_cam_emb=cam_emb_list[i],
                    cam_mlp=self.cam_mlp,
                    gate_attn=gate_attn,
                )
                for i in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    use_cam_emb=cam_emb_list[i],
                    cam_mlp=self.cam_mlp,
                    gate_attn=gate_attn,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.intermediate_layer_idx = intermediate_layer_idx

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        use_checkpoint=False,
        use_reentrant=True,
        use_chunkwise_checkpoint=False,
        gate_attn=None,  # "headwise", "elementwise", "per_token_headwise", or None/False
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
                use_checkpoint=use_checkpoint,
                use_reentrant=use_reentrant,
                use_chunkwise_checkpoint=use_chunkwise_checkpoint,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None, # dim 9 [quaternion, translation, fx, fy]
        camera_dropout: float = False,
        attn_mode: str = "full",
        kv_cache_list: List[List[torch.Tensor]] = None,
        block_attn_mask: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            cameras(torch.Tensor):
            camera_dropout(float):
            attn_mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        if self.low_vram:
            assert not self.training, "low_vram is only supported in inference"
        
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)

        if not self.low_vram:
            patch_tokens = self.patch_embed(images)
        else:
            # split into parts of 200 images
            patch_tokens = []
            max_batch_size = 200
            for i in range(0, B * S, max_batch_size):
                _img = images[i:i+max_batch_size]
                _tokens = self.patch_embed(_img)["x_norm_patchtokens"]
                patch_tokens.append(_tokens)
            patch_tokens = torch.cat(patch_tokens, dim=0)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        # xy: need to improve in the future
        is_ref_token_exist = kv_cache_list is None or kv_cache_list[0][0] is None
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, is_ref_token_exist=is_ref_token_exist)
        register_token = slice_expand_and_flatten(self.register_token, B, S, is_ref_token_exist=is_ref_token_exist)

        # Deal with possible camera conditioning
        if self.use_cam_emb and self.use_cam_token and cameras is not None:
            # Perform dropout by multiplying the network output with zeroes
            camera_prior = self.cam_mlp(cameras).reshape(B * S, 1, -1) * (1 - camera_dropout)
            camera_token = camera_token + camera_prior

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        del camera_token, register_token, patch_tokens

        pos = None
        if self.rope is not None:
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            if self.rope_type == "3d":
                pos_patch = self.position_getter(B, S, patch_h, patch_w, device=images.device)
                pos_patch = pos_patch + 1
                pos_patch = pos_patch.view(B, S, -1, 3)
                pos_special = torch.zeros(B, S, self.patch_start_idx, 3, device=images.device, dtype=pos_patch.dtype)
                pos = torch.cat([pos_special, pos_patch], dim=2)
            else:
                pos = self.position_getter(B * S, patch_h, patch_w, device=images.device)
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        attn_mask = None
        if kv_cache_list is None and block_attn_mask is None:
            attn_mask = create_attn_mask(S, P=P, mode=attn_mode, device=tokens.device)

        output_dict = {}
        if not self.use_chunkwise_checkpoint:
            frame_idx = 0
            global_idx = 0
            for _ in range(self.aa_block_num):
                layer_idx = frame_idx
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos, cam=cameras, cam_drop=camera_dropout
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, cam=cameras, cam_drop=camera_dropout, attn_mask=attn_mask, 
                            kv_cache_list=kv_cache_list, block_attn_mask=block_attn_mask
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")
            
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    if layer_idx + i in self.intermediate_layer_idx:
                        concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                        if self.low_vram:
                            concat_inter = concat_inter.cpu()
                        output_dict[layer_idx + i] = concat_inter
        else:
            def forward_layers_chunkwise(tokens, start_block_idx, end_block_idx):
                output_dict = {}
                for block_idx in range(start_block_idx, end_block_idx):
                    layer_idx = block_idx * self.aa_block_size
                    for attn_type in self.aa_order:
                        if attn_type == "frame":
                            tokens, _, frame_intermediates = self._process_frame_attention(
                                tokens, B, S, P, C, layer_idx, pos=pos, cam=cameras, cam_drop=camera_dropout
                            )
                        elif attn_type == "global":
                            tokens, _, global_intermediates = self._process_global_attention(
                                tokens, B, S, P, C, layer_idx, pos=pos, cam=cameras, cam_drop=camera_dropout, block_attn_mask=block_attn_mask
                            )
                    for j in range(len(frame_intermediates)):
                        if layer_idx + j in self.intermediate_layer_idx:
                            output_dict[layer_idx + j] = torch.cat([frame_intermediates[j], global_intermediates[j]], dim=-1)
                return tokens, output_dict

            chunk_block_num = 2
            assert self.aa_block_num % chunk_block_num == 0, \
                f"Please make sure the aa_block_num ({self.aa_block_num}) is divisible by chunk_block_num ({chunk_block_num})"
            for block_idx in range(0, self.aa_block_num, chunk_block_num):
                tokens, output_dict_idx = torch.utils.checkpoint.checkpoint(
                    forward_layers_chunkwise,
                    tokens,
                    block_idx, block_idx + chunk_block_num,
                    use_reentrant=self.use_reentrant,
                )
                output_dict.update(output_dict_idx)

        assert self.depth - 1 in output_dict, \
            f"Please make sure the last layer ({self.depth - 1}) is in the output_dict: {output_dict.keys()}"
        output_dict[-1] = output_dict[self.depth - 1]

        return output_dict, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, cam=None, cam_drop=False):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None:
            pos_dim = pos.shape[-1]
            if pos.shape != (B * S, P, pos_dim):
                pos = pos.view(B, S, P, pos_dim).view(B * S, P, pos_dim)

        if cam is not None:
            cam_dim = cam.shape[-1]
            if cam.shape != (B * S, cam_dim):
                cam = cam.view(B, S, cam_dim).view(B * S, cam_dim)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint and not self.use_chunkwise_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens, pos, cam, cam_drop,
                    use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.frame_blocks[frame_idx](
                    tokens, pos, cam, cam_drop
                )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, cam=None, cam_drop=False, attn_mask=None, kv_cache_list=None, block_attn_mask=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None:
            pos_dim = pos.shape[-1]
            if pos.shape != (B, S * P, pos_dim):
                pos = pos.view(B, S, P, pos_dim).view(B, S * P, pos_dim)

        if cam is not None:
            cam_dim = cam.shape[-1]
            if cam.shape != (B * S, cam_dim):
                cam = cam.view(B, S, cam_dim).view(B * S, cam_dim)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            # Each block should have its own kv_cache if kv_cache_list is provided
            block_kv_cache = kv_cache_list[global_idx] if kv_cache_list is not None else None
            if self.use_checkpoint and not self.use_chunkwise_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.global_blocks[global_idx],
                    tokens, pos, cam, cam_drop, attn_mask, block_kv_cache, block_attn_mask,
                    use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens, pos, cam, cam_drop, attn_mask, block_kv_cache, block_attn_mask,
                )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))


        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S, is_ref_token_exist=True):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    if is_ref_token_exist:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


@REGRESSORS.register_module()
class CameraCondition(nn.Module):
    def __init__(
        self,
        in_features: int = 9,
        out_features: int = 1024,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        actvation: str = "GELU",
        drop: float = 0.0,
        bias: bool = True,
        use_modulation: bool = True,
        use_gate: bool = True,
        use_transformer: bool = True,
        use_checkpoint=False,
        use_reentrant=False,
    ):
        super().__init__()

        # Determine the hidden features
        if use_transformer: hidden_features = out_features
        else: hidden_features = out_features * mlp_ratio

        # Embed the input camera parameters to tokens
        self.embed = Mlp(
            in_features=in_features,
            hidden_features=out_features//2,
            out_features=out_features,
            drop=0.0,
        )
        # self.embed = nn.Linear(in_features, hidden_features)

        if use_transformer:
            # A tiny transformer block to process the camera tokens
            self.trunk = nn.Sequential(
                *[
                    Block(
                        dim=hidden_features,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        init_values=init_values,
                    )
                    for _ in range(trunk_depth)
                ]
            )
        else:
            self.trunk = nn.Sequential(
                *[
                    *[
                        get_function(actvation),
                        nn.Linear(hidden_features, hidden_features, bias=bias),
                    ]
                ] * trunk_depth,
                # For modulation, we need to output the same number of features
                nn.SiLU(),
                nn.Linear(hidden_features, out_features, bias=bias),
            )

        if use_modulation:
            # Module for producing modulation parameters
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(out_features, 3 * out_features if use_gate else 2 * out_features, bias=True),
            )

        # Zero initialize both the embedding layer and the norm layer
        self.apply(zero_init)

        # Normalization layers
        self.token_norm = nn.LayerNorm(hidden_features)
        self.trunk_norm = nn.LayerNorm(out_features, elementwise_affine=False, eps=1e-6)

        # Bookkeepings
        self.use_modulation = use_modulation
        self.use_gate = use_gate
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the reverse camera head

        Args:
            x (Tensor), (B, S, C): B is batch size, S is source view number, and C is channels

        Returns:
            Tensor: Output tensor of shape (B, S, C)
        """
        # Embed the input
        x = self.token_norm(self.embed(x))

        # Process through the trunk
        if self.training and self.use_checkpoint:
            # Use gradient checkpointing for memory efficiency
            x = torch.utils.checkpoint.checkpoint(
                self.trunk, x, use_reentrant=self.use_reentrant
            )
        else:
            x = self.trunk(x)

        # Generate modulation parameters and
        # split them into shift, scale, and gate components
        if self.use_modulation:
            if self.use_gate:
                # If using gate, split into shift, scale, and gate
                shift, scale, gate = self.modulation(x).chunk(3, dim=-1)
                x = gate * (self.trunk_norm(x) * (1 + scale) + shift)
            else:
                # If not using gate, split into shift and scale
                shift, scale = self.modulation(x).chunk(2, dim=-1)
                x = self.trunk_norm(x) * (1 + scale) + shift
        # No modulation, just normalize the output
        else:
            x = self.trunk_norm(x)

        return x


def zero_init(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
