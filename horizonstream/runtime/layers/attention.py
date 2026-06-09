# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import sys
import torch
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from einops import rearrange
except Exception:
    rearrange = None

try:
    from .block_sparse_attention import block_sparse_attention
except Exception:
    block_sparse_attention = None

_FLA_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "third_party",
    "flash-linear-attention",
)
if os.path.isdir(_FLA_VENDOR_ROOT) and _FLA_VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _FLA_VENDOR_ROOT)

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        gate_attn: str = None,  # "headwise", "elementwise", "per_token_headwise", or None/False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        if gate_attn is True:
            gate_attn = "per_token_headwise"
        elif gate_attn is False or gate_attn is None:
            gate_attn = None
        self.gate_attn = gate_attn

        try:
            import flash_attn_interface
            if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
                self.flash_attn_interface = flash_attn_interface
            else:
                self.flash_attn_interface = None
        except ImportError:
            self.flash_attn_interface = None
        self.flash_attn_interface = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
  
        if self.gate_attn == "headwise":
            self.gate_proj = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)
        elif self.gate_attn == "elementwise":
            self.gate_proj = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)
        elif self.gate_attn == "per_token_headwise":
            self.gate_proj = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)  # sigmoid(2) ≈ 0.88

    def qkv_forward(self, x: Tensor, pos=None):
        B, N, C = x.shape
        # [B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        return q, k, v  # (B, H, A*F, C//H)

    def _compute_gate(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        if self.gate_attn == "headwise":
            x_pooled = x.mean(dim=1)
            gate = torch.sigmoid(self.gate_proj(x_pooled))
            gate = gate.view(B, self.num_heads, 1, 1)
        elif self.gate_attn == "elementwise":
            gate = torch.sigmoid(self.gate_proj(x))
            gate = gate.view(B, N, self.num_heads, self.head_dim)
            gate = gate.permute(0, 2, 1, 3)
        elif self.gate_attn == "per_token_headwise":
            gate = torch.sigmoid(self.gate_proj(x))
            gate = gate.permute(0, 2, 1).unsqueeze(-1)
        else:
            gate = None
        return gate

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, q_cache=None, block_attn_mask=None) -> Tensor:
        B, N, C = x.shape
        gate = self._compute_gate(x) if self.gate_attn else None
        if q_cache is None:
            # [B, num_heads, N, head_dim]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            if kv_cache is not None:
                if kv_cache.kv_ori is not None:
                    k_cache, v_cache = kv_cache.kv_ori.unbind(0)
                    k = torch.cat([k_cache, k], dim=2)
                    v = torch.cat([v_cache, v], dim=2)
                if kv_cache.kv_new is None: # Inplace update kv_cache but not overwrite
                    kv_cache.kv_new = torch.stack([k, v], dim=0)   # [2, B, num_heads, N, head_dim]
        else:
            q = q_cache
            k, v = kv_cache.kv_ori.unbind(0)

        if self.flash_attn_interface is not None:
            if self.attn_drop.p > 0.0 and self.training:
                raise AssertionError("Dropout is not supported with flash attention v3")
            # FA3 will return an additional logsumexp of each row of the matrix QK^T * scaling
            # https://github.com/Dao-AILab/flash-attention/blob/4d3d2ff2163ac011bce1b16a2eb2ca90a75f9628/hopper/flash_attn_interface.py#L495-L571
            x = self.flash_attn_interface.flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
            )[0]
            x = x.to(q.dtype)
        elif self.fused_attn and block_attn_mask is not None:
            n_images = block_attn_mask.shape[-1]
            q = rearrange(q.to(torch.bfloat16), "B H (N L) C -> B N L H C", N=n_images)
            k = rearrange(k.to(torch.bfloat16), "B H (N L) C -> B N L H C", N=n_images)
            v = rearrange(v.to(torch.bfloat16), "B H (N L) C -> B N L H C", N=n_images)

            x = block_sparse_attention(
                q,
                k,
                v,
                attn_mask=block_attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
            x = rearrange(x, "B N L H C -> B H (N L) C").to(q.dtype)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if gate is not None:
            x = x * gate
            
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LinearAttention(Attention):
    """
    Feature-map linear attention; masked calls use softmax attention.
    """

    def __init__(self, *args, feature_map: str = "elu", eps: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_map = feature_map
        self.eps = float(eps)

    def _phi(self, x: Tensor) -> Tensor:
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        if self.feature_map == "relu":
            return F.relu(x) + 1e-6
        if self.feature_map == "softplus":
            return F.softplus(x)
        raise ValueError(f"Unknown feature_map: {self.feature_map}")

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, q_cache=None, block_attn_mask=None) -> Tensor:
        if attn_mask is not None or block_attn_mask is not None:
            return super().forward(
                x,
                pos=pos,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                q_cache=q_cache,
                block_attn_mask=block_attn_mask,
            )

        B, N, C = x.shape
        gate = self._compute_gate(x) if self.gate_attn else None

        if q_cache is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            if kv_cache is not None:
                if kv_cache.kv_ori is not None:
                    k_cache, v_cache = kv_cache.kv_ori.unbind(0)
                    k = torch.cat([k_cache, k], dim=2)
                    v = torch.cat([v_cache, v], dim=2)
                if kv_cache.kv_new is None:
                    kv_cache.kv_new = torch.stack([k, v], dim=0)
        else:
            q = q_cache
            k, v = kv_cache.kv_ori.unbind(0)

        q = q * self.scale
        qf = self._phi(q.float())
        kf = self._phi(k.float())
        vf = v.float()
        kv = torch.einsum("bhld,bhlm->bhdm", kf, vf)
        z = kf.sum(dim=2)
        out = torch.einsum("bhld,bhdm->bhlm", qf, kv)
        denom = torch.einsum("bhld,bhd->bhl", qf, z).unsqueeze(-1)
        out = (out / (denom + self.eps)).to(dtype=q.dtype)

        if self.attn_drop.p > 0.0 and self.training:
            out = self.attn_drop(out)
        if gate is not None:
            out = out * gate.to(dtype=out.dtype)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


from fla.layers.kda import KimiDeltaAttention as KDA


class GLAAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        layer_idx: int = 0,
        expand_v: float = 1.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.layer_idx = int(layer_idx)
        self._warned_pos_ignored = False
        self.gla = KDA(
            hidden_size=dim,
            expand_v=float(expand_v),
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_v_heads=None,
            mode=str(mode),
            use_short_conv=bool(use_short_conv),
            allow_neg_eigval=bool(allow_neg_eigval),
            conv_size=int(conv_size),
            conv_bias=bool(conv_bias),
            layer_idx=self.layer_idx,
            norm_eps=float(norm_eps),
        )

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, q_cache=None, block_attn_mask=None) -> Tensor:
        if q_cache is not None:
            raise AssertionError("GLAAttention does not support q_cache.")
        if attn_mask is not None or block_attn_mask is not None:
            raise AssertionError("GLAAttention does not support arbitrary attention masks.")
        if pos is not None and not self._warned_pos_ignored:
            warnings.warn("GLAAttention ignores `pos`.", stacklevel=2)
            self._warned_pos_ignored = True

        y, _, _ = self.gla(
            hidden_states=x,
            attention_mask=None,
            past_key_values=kv_cache,
            use_cache=kv_cache is not None,
            output_attentions=False,
        )
        return y


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, attn_mask=None, kv_cache=None, q_cache=None, block_attn_mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, attn_mask=attn_mask, kv_cache=kv_cache, block_attn_mask=block_attn_mask)

        B, N, C = x.shape
        gate = None
        if self.gate_attn == "headwise":
            x_pooled = x.mean(dim=1)
            gate = torch.sigmoid(self.gate_proj(x_pooled)).view(B, 1, self.num_heads, 1)
        elif self.gate_attn == "elementwise":
            gate = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_heads, C // self.num_heads)
        elif self.gate_attn == "per_token_headwise":
            gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        if gate is not None:
            x = x * gate
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
