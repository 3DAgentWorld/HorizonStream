# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any

from functools import partial
from dataclasses import dataclass

from ..layers.block import Block
from ..layers.attention import LinearAttention, GLAAttention
from ..models.aggregator import Aggregator
from ..layers.group_encoder import GroupEncoder

@dataclass
class KVCache:
    """
    Explicit KV cache delta for one attention layer (one sample).
    """
    kv_ori: Optional[torch.Tensor] = None  # [2, B, H, T_old, D]
    kv_new: Optional[torch.Tensor] = None  # [2, B, H, T_new, D]


@dataclass
class GLACacheLayer:
    state_ori: Optional[Dict[str, Any]] = None
    state_new: Optional[Dict[str, Any]] = None


class GLACache:
    """Minimal recurrent-state cache for the GLA attention path."""

    def __init__(self, num_layers: int):
        self.layers: List[GLACacheLayer] = [GLACacheLayer() for _ in range(int(num_layers))]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        return self.layers[int(layer_idx)].state_ori

    def reset(self) -> None:
        for layer in self.layers:
            layer.state_ori = None
            layer.state_new = None

    def advance(self, *, is_last_chunk: bool) -> None:
        for layer in self.layers:
            layer.state_ori = None if is_last_chunk else layer.state_new
            layer.state_new = None

    def update(
        self,
        *,
        recurrent_state: torch.Tensor | None = None,
        conv_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        layer = self.layers[int(layer_idx)]
        if layer.state_new is not None:
            return layer.state_new

        state = {
            "recurrent_state": None,
            "conv_state": None,
        }
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state.detach()
        if conv_state is not None:
            if isinstance(conv_state, (tuple, list)):
                state["conv_state"] = tuple(t.detach() if torch.is_tensor(t) else t for t in conv_state)
            else:
                state["conv_state"] = conv_state.detach() if torch.is_tensor(conv_state) else conv_state

        layer.state_new = state
        return state


logger = logging.getLogger(__name__)

def slice_expand_and_flatten(
    token_tensor: torch.Tensor, # (2, X, C)
    B: int,
    F: int,
    Win: int,
    is_first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    assert token_tensor.ndim == 3 and token_tensor.shape[0] == 2
    tok_ref = token_tensor[0]  # [X, C]
    tok_src = token_tensor[1]  # [X, C]
    
    if is_first_chunk:
        grid_idx = torch.arange(F, device=device)
        diagonal_mask = (grid_idx[:, None] == grid_idx[None, :]) 
        assert F == Win, f"First chunk requires F({F}) == Win({Win})"
        final_mask = diagonal_mask.view(F, F, 1, 1)
    else:
        final_mask = torch.zeros(F, Win, 1, 1, device=device, dtype=torch.bool)
        final_mask[:, -1] = True

    out = torch.where(final_mask, tok_ref, tok_src) # (F, Win, X, C)

    return out.unsqueeze(0).expand(B, -1, -1, -1, -1).flatten(0, 2) # (B * F * Win, X, C)

class HorizonStreamAggregator(Aggregator):
    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        dense_intermediate_dims=[588, 768, 1024],
        fuse_grm_priori: bool = True,
        num_pose_tokens=32,
        global_attn_impl: str = "softmax",  # "softmax" | "linear"
        global_attn_arch: str = "swa",      # "swa" | "gla"
        gla_expand_v: float = 1.0,
        gla_use_short_conv: bool = True,
        gla_conv_size: int = 4,
        gla_conv_bias: bool = False,
        gla_allow_neg_eigval: bool = False,
        gla_norm_eps: float = 1e-5,
        gla_serial_layers: Optional[List[int]] = None,
        memory_mode: str = "token",
        memory_size: int = 0,
        chunk_block_num=2,
        max_detach_length: int = 1000,
        use_register_token: bool = True,
        enable_metric_readout_token: bool = False,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            **kwargs,
        )

        self.pose_token = nn.Parameter(torch.randn(2, num_pose_tokens, embed_dim))
        nn.init.normal_(self.pose_token, std=1e-6)

        self.global_attn_arch = str(global_attn_arch).lower() if global_attn_arch is not None else "swa"
        if self.global_attn_arch in ("swa", "full", "attn", "none"):
            self.global_attn_arch = "swa"
        elif self.global_attn_arch in ("gla", "gla_hybrid", "gla+attn", "gla_swa"):
            self.global_attn_arch = "gla"
        else:
            raise ValueError(f"Unknown global_attn_arch: {global_attn_arch}")

        self.global_attn_impl = str(global_attn_impl).lower()
        if self.global_attn_impl in ("linear", "lin"):
            for blk in self.global_blocks:
                old_attn = blk.attn
                qk_norm = not isinstance(old_attn.q_norm, nn.Identity)
                lin_attn = LinearAttention(
                    dim=old_attn.qkv.in_features,
                    num_heads=old_attn.num_heads,
                    qkv_bias=old_attn.qkv.bias is not None,
                    proj_bias=old_attn.proj.bias is not None,
                    attn_drop=old_attn.attn_drop.p,
                    proj_drop=old_attn.proj_drop.p,
                    norm_layer=nn.LayerNorm,
                    qk_norm=qk_norm,
                    fused_attn=old_attn.fused_attn,
                    rope=old_attn.rope,
                    gate_attn=old_attn.gate_attn,
                )
                lin_attn.load_state_dict(old_attn.state_dict(), strict=False)
                blk.attn = lin_attn
        elif self.global_attn_impl not in ("softmax", "sdpa", "swa"):
            raise ValueError(f"Unknown global_attn_impl: {global_attn_impl}")

        num_global_layers = len(self.global_blocks)
        if self.global_attn_arch == "gla":
            if memory_mode not in ("token", "tokens") or int(memory_size) != 0:
                raise ValueError("GLA port on this branch only supports memory_mode=token and memory_size=0.")
            if gla_serial_layers is None:
                gla_serial_layers = list(self.intermediate_layer_idx)
            self.gla_serial_layers = self._normalize_layer_indices(gla_serial_layers, num_global_layers)
            serial_layer_set = set(self.gla_serial_layers)
            self.global_is_serial = [i in serial_layer_set for i in range(num_global_layers)]
        else:
            self.gla_serial_layers = []
            self.global_is_serial = [False for _ in range(num_global_layers)]

        self.global_gla_blocks = nn.ModuleDict()
        if self.global_attn_arch == "gla":
            for layer_idx, blk in enumerate(self.global_blocks):
                if not self.global_is_serial[layer_idx]:
                    continue
                old_attn = blk.attn
                gla_block = Block(
                    dim=embed_dim,
                    num_heads=int(getattr(old_attn, "num_heads", kwargs.get("num_heads", 16))),
                    mlp_ratio=float(kwargs.get("mlp_ratio", 4.0)),
                    qkv_bias=bool(kwargs.get("qkv_bias", True)),
                    proj_bias=bool(kwargs.get("proj_bias", True)),
                    ffn_bias=bool(kwargs.get("ffn_bias", True)),
                    drop=float(kwargs.get("drop", 0.0)),
                    attn_drop=float(kwargs.get("attn_drop", 0.0)),
                    init_values=kwargs.get("init_values", 0.01),
                    drop_path=float(kwargs.get("drop_path", 0.0)),
                    act_layer=kwargs.get("act_layer", nn.GELU),
                    norm_layer=kwargs.get("norm_layer", nn.LayerNorm),
                    attn_class=partial(
                        GLAAttention,
                        head_dim=getattr(old_attn, "head_dim", None),
                        layer_idx=int(layer_idx),
                        expand_v=float(gla_expand_v),
                        mode="chunk",
                        use_short_conv=bool(gla_use_short_conv),
                        allow_neg_eigval=bool(gla_allow_neg_eigval),
                        conv_size=int(gla_conv_size),
                        conv_bias=bool(gla_conv_bias),
                        norm_eps=float(gla_norm_eps),
                    ),
                    qk_norm=bool(kwargs.get("qk_norm", False)),
                    fused_attn=bool(kwargs.get("fused_attn", True)),
                    rope=None,
                    gate_attn=kwargs.get("gate_attn", None),
                )
                gla_block.load_state_dict(blk.state_dict(), strict=False)
                self.global_gla_blocks[str(layer_idx)] = gla_block

        if fuse_grm_priori:
            self.group_embed = GroupEncoder(
                embed_dim=embed_dim,
                patch_size=patch_size,
                dense_intermediate_dims=dense_intermediate_dims,
            )
            self.fusion_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fuse_grm_priori = fuse_grm_priori
        self.num_pose_tokens = num_pose_tokens
        self.chunk_block_num = chunk_block_num
        self.max_detach_length = max_detach_length
        self.use_register_token = use_register_token
        self.enable_metric_readout_token = enable_metric_readout_token
        if not use_register_token:
            self.register_token.requires_grad_(False)
        if self.enable_metric_readout_token:
            self.metric_readout_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.normal_(self.metric_readout_token, std=1e-6)
        self.camera_token.requires_grad_(False)


    @torch.no_grad()
    def build_global_attn_mask(
        self,
        F: int,
        P: int,
        Win: int,
        X: int,
        device: torch.device,
    ):
        """
        Returns:
            attn_mask_bool: torch.bool, shape (L, S)
                True  => allowed / keep
                False => masked out (SDPA will turn ~mask to -inf per your reference)
        """
        assert Win == F, f"First chunk requires Win({Win}) == F({F})"

        A = P + Win * X

        L = F * A
        r = torch.arange(L, device=device, dtype=torch.long)

        br = r // A         # (L,) row block id
        orr = r % A         # (L,) row offset within block
        bc = br             # (S,) col block id
        occ = orr           # (S,) col offset within block

        BR = br[:, None]     # block id of each row, (L,1) [0..., 1..., ..., F-1...]
        OR = orr[:, None]    # offset of each row, (L,1) [0...A-1] * F    
        BC = bc[None, :]     # block id of each col, (1,S) [0..., 1..., ..., CS-1...]
        OC = occ[None, :]    # offset of each col, (1,S) [0...A-1] * CS

        # active rows in row-block i: offsets < P + (i+1)*X
        row_ok = OR < (P + (BR + 1) * X) # corresponding to valid query

        # allow patch tokens (first P cols) from previous blocks j < i
        prev_patch_ok = (BC < BR) & (OC < P) # cross-attention block

        # allow within diagonal block i up to P + (i+1)*X
        diag_ok = (BC == BR) & (OC < (P + (BR + 1) * X)) # self-attention block

        allowed = row_ok & (prev_patch_ok | diag_ok)

        # match mask.fill_diagonal_(0): always allow self
        allowed |= (r[:, None] == r[None, :])

        return allowed

    @staticmethod
    def _normalize_layer_indices(layer_indices: Optional[List[int]], num_layers: int) -> List[int]:
        if layer_indices is None:
            return []
        normalized = []
        for raw_idx in layer_indices:
            idx = int(raw_idx)
            if idx < 0:
                idx += int(num_layers)
            if idx < 0 or idx >= int(num_layers):
                raise ValueError(f"Layer index {raw_idx} is out of range for num_layers={num_layers}")
            normalized.append(idx)
        return sorted(set(normalized))

    def sync_gla_from_global_blocks(self) -> None:
        for key, gla_block in self.global_gla_blocks.items():
            gla_block.load_state_dict(self.global_blocks[int(key)].state_dict(), strict=False)

    def _run_gla_global_block(
        self,
        global_idx: int,
        patch_tokens: torch.Tensor,
        pose_tokens: torch.Tensor,
        B: int,
        F: int,
        P: int,
        C: int,
        X: int,
        Win: int,
        gla_cache: GLACache,
    ) -> torch.Tensor:
        gla_block = self.global_gla_blocks[str(global_idx)]
        pose_flat = pose_tokens.view(B * F, Win * X, C)
        all_tokens = torch.cat([patch_tokens, pose_flat], dim=1)
        return gla_block(
            all_tokens.reshape(B, F * (P + Win * X), C),
            pos=None,
            attn_mask=None,
            kv_cache=gla_cache,
        ).view(B * F, P + Win * X, C)

    def forward(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None,  # dim 9 [quaternion, translation, fx, fy]
        camera_dropout: float = 0.0,
        frame_kv_caches: List[KVCache] = None,
        global_kv_caches: List[KVCache] = None,
        group_maps: torch.Tensor = None,  # [3*B, V, 3, H, W]
        n_views: int = 1,
        window_size: int = 10,
        chunk_idx: int = 0,
        rope_frame_start: int = 0,
        gla_cache: GLACache = None,
    ) -> Tuple[Dict, torch.Tensor, int]:
        V = int(n_views.item()) if hasattr(n_views, "item") else int(n_views)
        Win = int(window_size.item()) if hasattr(window_size, "item") else int(window_size)

        if self.low_vram:
            assert not self.training, "low_vram is only supported in inference"

        B, S, C_in, H, W = images.shape
        assert S % V == 0, "S is not divisible by n_views."
        assert B == 1, "Batch size must be one."

        # ---- required inputs guards (clear errors) ----
        assert frame_kv_caches is not None and global_kv_caches is not None, "kv caches must be provided"
        assert len(frame_kv_caches) == len(self.frame_blocks), "frame_kv_caches length mismatch"
        assert len(global_kv_caches) == len(self.global_blocks), "global_kv_caches length mismatch"
        if self.global_attn_arch == "gla":
            assert gla_cache is not None, "gla_cache must be provided when global_attn_arch=gla"
            assert len(gla_cache) == len(self.global_blocks), "gla_cache length mismatch"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.reshape(B * S, C_in, H, W)

        if not self.low_vram:
            patch_tokens = self.patch_embed(images)
        else:
            patch_tokens = []
            max_batch_size = 200
            for i in range(0, B * S, max_batch_size):
                _img = images[i:i + max_batch_size]
                _tokens = self.patch_embed(_img)["x_norm_patchtokens"]
                patch_tokens.append(_tokens)
            patch_tokens = torch.cat(patch_tokens, dim=0)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, _, C = patch_tokens.shape

        F = S // V  # number of frames
        patch_tokens = patch_tokens.reshape(B * F, -1, C)
        base_patch_tokens = patch_tokens

        if self.fuse_grm_priori:
            assert group_maps is not None, "fuse_grm_priori=True but group_maps is None"
            base_patch_tokens = self.fusion_norm_layer(
                base_patch_tokens + self.group_embed(group_maps).expand(B * F, -1, C)
            )
        register_tokens = None
        special_tokens = []
        if self.use_register_token:
            register_tokens = self.register_token[:, 1:, ...].expand(B, F, *self.register_token.shape[2:]).flatten(0, 1) # (B*F, 4, C)
            special_tokens.append(register_tokens)
        if self.enable_metric_readout_token:
            metric_readout_tokens = self.metric_readout_token.expand(B * F, -1, -1)
            special_tokens.append(metric_readout_tokens)
        if len(special_tokens):
            patch_tokens = torch.cat([*special_tokens, base_patch_tokens], dim=1)
        else:
            patch_tokens = base_patch_tokens

        _, P, C = patch_tokens.shape
        special_patch_tokens = (register_tokens.shape[1] if register_tokens is not None else 0) + int(self.enable_metric_readout_token)

        X = self.num_pose_tokens
        A = P + Win * X
        is_first_chunk = chunk_idx == 0
        global_attn_mask = None
        if not is_first_chunk:
            self._Check_KV_Cache(frame_kv_caches[0], global_kv_caches[0], B, P, Win, A)
        else:
            assert F == Win, f"First chunk requires F({F}) == Win({Win})"
            assert frame_kv_caches[0].kv_ori is None
            global_attn_mask = self.build_global_attn_mask(F, P, Win, X, device=images.device)
            assert global_attn_mask.shape == (F * A, F * A), "patch_attn_mask shape mismatch"

        pose_tokens = slice_expand_and_flatten(
            self.pose_token, B, F, Win, is_first_chunk=is_first_chunk, device=images.device
        ) # (B * F * Win, X, C)
        assert pose_tokens.shape == (B * F * Win, X, C)

        patch_pos = None
        all_pos = None
        pose_pos = None
        if self.rope is not None:
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            if self.rope_type == "3d":
                patch_pos = self.position_getter(B, F, patch_h, patch_w, device=images.device)
                if rope_frame_start:
                    patch_pos[..., 0] = patch_pos[..., 0] + int(rope_frame_start)
                patch_pos = patch_pos + 1
                patch_pos = patch_pos.view(B, F, -1, 3)
                patch_pos_dtype = patch_pos.dtype
                if special_patch_tokens > 0:
                    register_pos = torch.zeros(
                        B,
                        F,
                        special_patch_tokens,
                        3,
                        device=images.device,
                        dtype=patch_pos_dtype,
                    )
                    patch_pos = torch.cat([register_pos, patch_pos], dim=2).contiguous()
                pose_pos = torch.zeros(
                    (B, F, Win * X, 3),
                    device=images.device,
                    dtype=patch_pos_dtype,
                )
                all_pos = torch.cat([patch_pos, pose_pos], dim=2).contiguous().view(B * F, A, 3)
                patch_pos = patch_pos.view(B * F, P, 3)
                pose_pos = pose_pos.view(B * F * Win, X, 3)
            else:
                patch_pos = self.position_getter(
                    B * S, patch_h, patch_w, device=images.device
                )
                patch_pos = patch_pos + 1
                patch_pos = patch_pos.reshape(B * F, -1, 2)
                if special_patch_tokens > 0:
                    register_pos = torch.zeros(
                        (B * F, special_patch_tokens, 2),
                        device=images.device,
                        dtype=patch_pos.dtype,
                    )
                    patch_pos = torch.cat([register_pos, patch_pos], dim=1).contiguous()

                pose_pos = torch.zeros(
                    (B * F, Win * X, 2),
                    device=images.device,
                    dtype=patch_pos.dtype,
                )
                all_pos = torch.cat([patch_pos, pose_pos], dim=1).contiguous()
                pose_pos = pose_pos.view(B * F * Win, X, 2)

        output_dict: Dict[int, torch.Tensor] = {}
        output_pose_tokens = None  # (B, F, Win, 2C)

        frame_patch_dict: Dict[int, torch.Tensor] = {}
        global_patch_dict: Dict[int, torch.Tensor] = {}
        frame_pose_dict: Dict[int, torch.Tensor] = {}
        global_pose_dict: Dict[int, torch.Tensor] = {}
        
        unfold_idx = None
        if not is_first_chunk:
            unfold_idx = (torch.arange(F, device=images.device)[:, None] + torch.arange(Win, device=images.device)[None, :])  # (F,Win)
            unfold_idx = unfold_idx.reshape(-1)

        if not self.use_chunkwise_checkpoint:
            frame_idx = 0
            global_idx = 0
            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        patch_tokens, pose_tokens, frame_idx, frame_patch_inter, frame_pose_inter = self._process_causal_frame_attention(
                            patch_tokens, pose_tokens, B, F, P, C, X, Win, frame_idx, unfold_idx,
                            patch_pos=patch_pos, pose_pos=pose_pos, kv_cache_list=frame_kv_caches, cam=cameras, cam_drop=camera_dropout,
                        )
                    elif attn_type == "global":
                        patch_tokens, pose_tokens, global_idx, global_patch_inter, global_pose_inter = self._process_causal_global_attention(
                            patch_tokens, pose_tokens, B, F, P, C, X, Win, global_idx, unfold_idx,
                            all_pos=all_pos, kv_cache_list=global_kv_caches, attn_mask=global_attn_mask, gla_cache=gla_cache,
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                frame_patch_dict.update(frame_patch_inter)
                global_patch_dict.update(global_patch_inter)
                frame_pose_dict.update(frame_pose_inter)
                global_pose_dict.update(global_pose_inter)

        else:
            def forward_layers_chunkwise(patch_tokens, pose_tokens, start_block_idx, end_block_idx):
                frame_idx_local = start_block_idx * self.aa_block_size
                global_idx_local = start_block_idx * self.aa_block_size

                for _ in range(start_block_idx, end_block_idx):
                    for attn_type in self.aa_order:
                        if attn_type == "frame":                      # 只 reset rank0 的峰值统计
                            patch_tokens, pose_tokens, frame_idx_local, frame_patch_inter, frame_pose_inter = self._process_causal_frame_attention(
                                patch_tokens, pose_tokens, B, F, P, C, X, Win, frame_idx_local, unfold_idx,
                                patch_pos=patch_pos, pose_pos=pose_pos, kv_cache_list=frame_kv_caches, cam=cameras, cam_drop=camera_dropout,
                            )
                        elif attn_type == "global":                      # 只 reset rank0 的峰值统计
                            patch_tokens, pose_tokens, global_idx_local, global_patch_inter, global_pose_inter = self._process_causal_global_attention(
                                patch_tokens, pose_tokens, B, F, P, C, X, Win, global_idx_local, unfold_idx,
                                all_pos=all_pos, kv_cache_list=global_kv_caches, attn_mask=global_attn_mask, gla_cache=gla_cache,
                            )
                        else:
                            raise ValueError(f"Unknown attention type: {attn_type}")

                    frame_patch_dict.update(frame_patch_inter)
                    global_patch_dict.update(global_patch_inter)
                    frame_pose_dict.update(frame_pose_inter)
                    global_pose_dict.update(global_pose_inter)

                return patch_tokens, pose_tokens

            assert self.aa_block_num % self.chunk_block_num == 0, \
                f"Please make sure the aa_block_num ({self.aa_block_num}) is divisible by chunk_block_num ({self.chunk_block_num})"
            for block_idx in range(0, self.aa_block_num, self.chunk_block_num):
                patch_tokens, pose_tokens = torch.utils.checkpoint.checkpoint(
                    forward_layers_chunkwise,
                    patch_tokens, pose_tokens,
                    block_idx, block_idx + self.chunk_block_num,
                    use_reentrant=self.use_reentrant,
                )

        for idx in self.intermediate_layer_idx:
            output_dict[idx] = torch.cat([frame_patch_dict[idx], global_patch_dict[idx]], dim=-1)
        last_idx = self.intermediate_layer_idx[-1]
        output_pose_tokens = torch.cat([frame_pose_dict[last_idx], global_pose_dict[last_idx]], dim=-1)

        assert self.depth - 1 in output_dict, \
            f"Please make sure the last layer ({self.depth - 1}) is in the output_dict: {output_dict.keys()}"
        output_dict[-1] = output_dict[self.depth - 1]

        assert output_pose_tokens is not None
        # output_pose_tokens: (B, F, Win, 2C)  -> batch-first
        return output_dict, output_pose_tokens, special_patch_tokens

    def _Check_KV_Cache(self, frame_kv_cache, global_kv_cache, B, P, Win, A):
        _, B_f, Win_1_f, H_f, P_f, D_f = frame_kv_cache.kv_ori.shape    # (2, B, Win-1, H, P_f, C//H)
        _, B_g, H_g, Win_1_g, P_g, D_g = global_kv_cache.kv_ori.shape   # (2, B, H, Win-1, P_g, C//H)

        assert B_f == B_g == B and Win_1_f == Win_1_g == Win - 1 and H_f == H_g and P_f == P_g == P and D_f == D_g, \
            f"frame_shape: {frame_kv_cache.kv_ori.shape}, global_shape: {global_kv_cache.kv_ori.shape}"
        assert frame_kv_cache.kv_new is None and global_kv_cache.kv_new is None

    def _process_causal_frame_attention(
        self, patch_tokens, pose_tokens, B, F, P, C, X, Win,
        frame_idx, unfold_idx, patch_pos=None, pose_pos=None, kv_cache_list=None, cam=None, cam_drop=None,
    ):
        """
        patch_tokens: (B*F,P,C)
        pose_tokens : (B,F,Win,X,C)
        """
        if cam is not None:
            cam_dim = cam.shape[-1]
            if cam.shape != (B * F * Win, cam_dim):
                cam = cam.view(B, F, Win, cam_dim).view(B * F * Win, cam_dim)
        assert patch_tokens.shape == (B * F, P, C)
        assert pose_tokens.shape == (B * F * Win, X, C)
        assert patch_pos is not None and pose_pos is not None
        assert patch_pos.shape[:2] == (B * F, P)
        assert pose_pos.shape[:2] == (B * F * Win, X)

        patch_intermediates: Dict[int, torch.Tensor] = {}
        pose_intermediates: Dict[int, torch.Tensor] = {}

        for _ in range(self.aa_block_size):
            # ---- patch step ----
            local_kv_cache = KVCache()
            patch_tokens = self.frame_blocks[frame_idx](patch_tokens, patch_pos, kv_cache=local_kv_cache)

            kv_remain_shape = local_kv_cache.kv_new.shape[-3:] # (H, P, C//H)
            kv_cache = kv_cache_list[frame_idx]
            local_kv_cache.kv_new = local_kv_cache.kv_new.reshape(2, B, F, *kv_remain_shape) # (2, B, F, H, P, C//H)
            
            # Update local kv cache
            if kv_cache.kv_ori is None: # First chunk
                local_kv_cache.kv_ori = local_kv_cache.kv_new
            else:
                local_kv_cache.kv_ori = torch.cat([kv_cache.kv_ori, local_kv_cache.kv_new], dim=2) # (2, B, Win - 1 + F, H, P, C//H)
            
            # if local_kv_cache.kv_ori.shape[2] > self.max_detach_length:
            #     local_kv_cache.kv_ori = detach_kv_cache(local_kv_cache.kv_ori, dim=2, keep_tail_length=self.max_detach_length)
            # Update global kv cache
            if kv_cache.kv_new is None: # avoid reloading in checkpoint
                kv_cache.kv_new = local_kv_cache.kv_ori[:, :, 1-Win:].detach() # (2, B, Win-1, H, P, C//H)
                
            # Expand local kv cache
            if kv_cache.kv_ori is None: # First chunk
                local_kv_cache.kv_ori = local_kv_cache.kv_ori.unsqueeze(2).expand(2, B, F, Win, *kv_remain_shape).flatten(1, 3) # (2, B*F*Win, H, P, C//H)
            else:
                s0, s1, s2, s3, s4, s5 = local_kv_cache.kv_ori.stride()
                kv_cache_win = local_kv_cache.kv_ori.as_strided(size=(2, B, F, Win, *kv_remain_shape), stride=(s0, s1, s2, s2, s3, s4, s5)) # (2,B,F,Win,H,P,C//H)
                local_kv_cache.kv_ori = kv_cache_win.flatten(1, 3) # (2, B*F*Win, H, P, C//H)
                # kv_cache_select = local_kv_cache.kv_ori.index_select(dim=2, index=unfold_idx) # (2, B, F*Win, H, P, C//H)
                # local_kv_cache.kv_ori = kv_cache_select.flatten(1, 2) # (2, B*F*Win, H, P, C//H)
                # local_kv_cache.kv_ori = local_kv_cache.kv_ori.unfold(dimension=2, size=Win, step=1) # (2, B, F, H, P, C//H, Win)
                # local_kv_cache.kv_ori = local_kv_cache.kv_ori.permute(0, 1, 2, 6, 3, 4, 5).flatten(1, 3) # (2, B*F*Win, H, P, C//H)
                
            # Avoid updating local.kv_new in attention
            local_kv_cache.kv_new = torch.tensor([], device=local_kv_cache.kv_ori.device)
            
            pose_tokens = self.frame_blocks[frame_idx](pose_tokens, pose_pos, kv_cache=local_kv_cache, cams=cam, cam_drop=cam_drop)

            if frame_idx in self.intermediate_layer_idx and frame_idx not in patch_intermediates:
                patch_intermediates[frame_idx] = patch_tokens.reshape(B, F, P, C)  # (B,F,P,C)
                if frame_idx == self.intermediate_layer_idx[-1]:
                    pose_intermediates[frame_idx] = pose_tokens.view(B, F, Win, X, C)[..., 0, :]     # (B,F,Win,C)

            frame_idx += 1

        return patch_tokens, pose_tokens, frame_idx, patch_intermediates, pose_intermediates

    def _process_causal_global_attention(
        self, patch_tokens, pose_tokens, B, F, P, C, X, Win,
        global_idx, unfold_idx, all_pos=None, kv_cache_list=None, attn_mask=None, gla_cache: GLACache = None,
    ):
        """
        patch_tokens: (B*F,P,C)
        pose_tokens : (B*F*Win,X,C)
        NOTE: pose_pos is expected to be all-zeros; we simply crop [:Win] (same for every row).
        """
        assert patch_tokens.shape == (B * F, P, C)
        assert pose_tokens.shape == (B * F * Win, X, C)
        assert all_pos is not None

        # reshape patch tokens for global blocks
        A = P + Win * X
        all_tokens = torch.cat([patch_tokens, pose_tokens.view(B * F, Win * X, C)], dim = 1) # (B*F,A,C)
        assert A == all_tokens.shape[1]
        pos_dim = all_pos.shape[-1]
        assert all_pos.shape == (B * F, A, pos_dim)

        patch_intermediates: Dict[int, torch.Tensor] = {}
        pose_intermediates: Dict[int, torch.Tensor] = {}

        for _ in range(self.aa_block_size):
            if self.global_attn_arch == "gla" and self.global_is_serial[global_idx]:
                all_tokens = self._run_gla_global_block(
                    global_idx,
                    patch_tokens,
                    pose_tokens,
                    B,
                    F,
                    P,
                    C,
                    X,
                    Win,
                    gla_cache,
                )
                patch_tokens = all_tokens[:, :P]
                pose_tokens = all_tokens[:, P:].reshape(B * F * Win, X, C)

            kv_cache = kv_cache_list[global_idx]

            if kv_cache.kv_ori is None: # First chunk
                all_tokens = all_tokens.reshape(B, F * A, C)
                all_pos = all_pos.view(B, F * A, all_pos.shape[-1])
                assert attn_mask is not None
                all_tokens = self.global_blocks[global_idx](all_tokens, all_pos, attn_mask=attn_mask, kv_cache=kv_cache) # (B, F * A, C)
                all_tokens = all_tokens.reshape(B * F, A, C) # (B*F,P+Win*X,C)
                all_pos = all_pos.view(B * F, A, all_pos.shape[-1])
                # trim kv cache
                if kv_cache.kv_new.ndim == 5: # (2, B, H, F*A ,C//H) avoid reloading in checkpoint
                    kv_cache.kv_new = kv_cache.kv_new[..., (1 - Win) * A:, :].detach() # (2, B, H, (Win - 1)*A, C//H)
                    _, _, H, _, D = kv_cache.kv_new.shape
                    kv_cache.kv_new = kv_cache.kv_new.reshape(2, B, H, Win - 1, A, D)[..., :P, :] # (2, B, H, Win - 1, P, C//H)
            else:
                global_block = self.global_blocks[global_idx]
                q_cache, k, v = global_block.attn.qkv_forward(global_block.norm1(all_tokens), all_pos) # (B * F, H, A, C//H)
                _, H, _, D = k.shape
                new_kv = torch.stack([k, v], dim=0)   # (2, B * F, H, A, C//H)
                kv_pose = new_kv[..., - Win * X:, :] # (2, B * F, H, Win * X, C//H)
                kv_patch = new_kv.reshape(2, B, F, H, A, D).transpose(2, 3)[..., :P, :] # (2, B, H, F, P, C//H)
                
                new_kv_cache = KVCache()
                new_kv_cache.kv_ori = torch.cat([kv_cache.kv_ori, kv_patch], dim=3) # (2, B, H, Win - 1 + F, P, C//H)
                # if new_kv_cache.kv_ori.shape[3] > self.max_detach_length:
                #     new_kv_cache.kv_ori = detach_kv_cache(new_kv_cache.kv_ori, dim=3, keep_tail_length=self.max_detach_length)
                if kv_cache.kv_new is None: # avoid overwriting in checkpoint
                    kv_cache.kv_new = new_kv_cache.kv_ori[..., (1 - Win):, :, :].detach() # (2, B, H, Win - 1, P, C//H)
                kv_cache_select = new_kv_cache.kv_ori.index_select(dim=3, index=unfold_idx) # (2, B, H, F*Win, P, C//H)
                kv_cache_select = kv_cache_select.view(2, B, H, F, Win, P, D).transpose(2, 3) # (2, B, F, H, Win, P, C//H)
                new_kv_cache.kv_ori = kv_cache_select.reshape(2, B * F, H, Win * P, D)
                # new_kv_cache.kv_ori = new_kv_cache.kv_ori.unfold(dimension=3, size=Win, step=1) # (2, B, H, F, P, C//H, Win)
                # new_kv_cache.kv_ori = new_kv_cache.kv_ori.permute(0, 1, 3, 2, 6, 4, 5).reshape(2, B * F, H, Win * P, D)
                new_kv_cache.kv_ori = torch.cat([new_kv_cache.kv_ori, kv_pose], dim=3) # (2, B * F, H, Win * P + Win * X, C//H)
                
                all_tokens = global_block(all_tokens, all_pos, kv_cache=new_kv_cache, q_cache=q_cache) # (B * F, A, C)
                
            patch_tokens = all_tokens[:, :P] # (B*F,P,C)
            pose_tokens = all_tokens[:, P:].reshape(B * F * Win, X, C)
            
            if global_idx in self.intermediate_layer_idx and global_idx not in patch_intermediates:
                patch_intermediates[global_idx] = patch_tokens.reshape(B, F, P, C)
                if global_idx == self.intermediate_layer_idx[-1]:
                    pose_intermediates[global_idx] = pose_tokens.view(B, F, Win, X, C)[..., 0, :]  # (B,F,Win,C)

            global_idx += 1

        return patch_tokens, pose_tokens, global_idx, patch_intermediates, pose_intermediates
