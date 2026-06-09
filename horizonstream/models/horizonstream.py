from __future__ import annotations

import logging
import os
import inspect
from typing import Any, Dict, List

import torch
import torch.nn as nn

from horizonstream.utils.vendor.models.components.heads.dpt_head import DPTHead
from horizonstream.runtime.heads.camera_head import BatchCameraHead
from horizonstream.runtime.models.horizonstream import (
    GLACache,
    KVCache,
    HorizonStreamAggregator,
)

logger = logging.getLogger(__name__)


class HorizonStream(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        cfg = dict(cfg or {})

        agg_cfg = dict(cfg.pop("agg_regator_cfg", {}) or {})
        cam_cfg = dict(cfg.pop("cam_decoder_cfg", {}) or {})
        dpt_cfg = dict(cfg.pop("dpt_decoder_cfg", {}) or {})
        metric_readout_head_cfg = dict(cfg.pop("metric_readout_head_cfg", {}) or {})

        self.load_pretrained = bool(cfg.pop("load_pretrained", False))
        self.enable_metric_readout_token = bool(cfg.pop("enable_metric_readout_token", False))
        self.frames_chunk_size = int(cfg.pop("frames_chunk_size", 1))
        self.use_xyz_head = bool(cfg.pop("use_xyz_head", False))
        self.rope_temporal_period = int(agg_cfg.pop("rope_temporal_period", 0) or 0)

        self.agg_regator_ckpt = cfg.pop("agg_regator_ckpt", None)
        self.cam_decoder_ckpt = cfg.pop("cam_decoder_ckpt", None)
        self.dpt_decoder_ckpt = cfg.pop("dpt_decoder_ckpt", None)

        for key in (
            "use_checkpoint",
            "use_reentrant",
            "use_chunkwise_checkpoint",
            "use_cam_emb",
            "cam_embed_cfg",
            "low_vram",
            "gate_attn",
        ):
            if key in cfg and key not in agg_cfg:
                agg_cfg[key] = cfg.pop(key)

        embed_dim = int(agg_cfg.get("embed_dim", 1024))
        if "intermediate_layer_idx" in agg_cfg and "intermediate_layer_idx" not in dpt_cfg:
            dpt_cfg["intermediate_layer_idx"] = agg_cfg["intermediate_layer_idx"]
        cam_cfg.setdefault("dim_in", embed_dim * 2)
        dpt_cfg.setdefault("dim_in", embed_dim * 2)
        dpt_cfg.setdefault("output_dim", 2)
        dpt_cfg.setdefault("activation", "exp")
        dpt_cfg.setdefault("conf_activation", "expp1")

        self.agg_regator = HorizonStreamAggregator(
            enable_metric_readout_token=self.enable_metric_readout_token,
            **agg_cfg,
        )
        self.cam_decoder = BatchCameraHead(**cam_cfg)
        dpt_params = inspect.signature(DPTHead).parameters
        dpt_cfg = {k: v for k, v in dpt_cfg.items() if k in dpt_params}
        self.dpt_decoder = DPTHead(**dpt_cfg)

        self.metric_readout_head = None
        if self.enable_metric_readout_token:
            self.metric_readout_head = self._build_metric_readout_head(
                int(cam_cfg["dim_in"]),
                metric_readout_head_cfg,
            )

        if self.load_pretrained:
            self.load_component_pretrained()

    def _build_metric_readout_head(self, dim_in: int, metric_readout_head_cfg: Dict[str, Any]) -> nn.Sequential:
        hidden_dims = list(metric_readout_head_cfg.get("hidden_dims", [256, 128]))
        init_bias = float(metric_readout_head_cfg.get("init_bias", 0.0))
        layers: List[nn.Module] = []
        last_dim = dim_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        metric_readout_head = nn.Sequential(*layers)
        for module in metric_readout_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(metric_readout_head[-1].bias, init_bias)
        return metric_readout_head

    def _build_empty_caches(self):
        frame_kv_caches = [KVCache() for _ in range(len(self.agg_regator.frame_blocks))]
        global_kv_caches = [KVCache() for _ in range(len(self.agg_regator.global_blocks))]
        gla_cache = None
        if getattr(self.agg_regator, "global_attn_arch", "swa") == "gla":
            gla_cache = GLACache(len(self.agg_regator.global_blocks))
        return frame_kv_caches, global_kv_caches, gla_cache

    def build_sequence_state(self) -> Dict[str, Any]:
        frame_kv_caches, global_kv_caches, gla_cache = self._build_empty_caches()
        return {
            "frame_kv_caches": frame_kv_caches,
            "global_kv_caches": global_kv_caches,
            "gla_cache": gla_cache,
        }

    def advance_sequence_state(self, state: Dict[str, Any], *, is_last_chunk: bool) -> None:
        frame_kv_caches = state["frame_kv_caches"]
        global_kv_caches = state["global_kv_caches"]
        gla_cache = state.get("gla_cache")

        for global_kv, frame_kv in zip(global_kv_caches, frame_kv_caches):
            if is_last_chunk:
                global_kv.kv_ori = None
                frame_kv.kv_ori = None
            else:
                global_kv.kv_ori = global_kv.kv_new
                frame_kv.kv_ori = frame_kv.kv_new
            global_kv.kv_new = None
            frame_kv.kv_new = None

        if gla_cache is not None:
            gla_cache.advance(is_last_chunk=is_last_chunk)

    def _output_dict_to_list(self, output_dict: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        valid_keys = [k for k in output_dict.keys() if k >= 0]
        if not valid_keys:
            raise ValueError("Aggregator did not return any non-negative intermediate layer outputs.")
        required_keys = set(valid_keys)
        required_keys.update(int(idx) for idx in getattr(self.dpt_decoder, "intermediate_layer_idx", []))
        max_idx = max(required_keys)
        outputs: List[torch.Tensor | None] = [None] * (max_idx + 1)
        for idx, tensor in output_dict.items():
            if idx >= 0:
                outputs[idx] = tensor
        missing = [idx for idx in sorted(required_keys) if outputs[idx] is None]
        if missing:
            raise ValueError(f"Missing intermediate layers for DPT head: {missing}")
        return outputs

    def _predict_metric_scale(self, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int):
        if not self.enable_metric_readout_token or self.metric_readout_head is None:
            return None, None
        metric_readout_token_idx = patch_start_idx - 1
        metric_readout_features = aggregated_tokens_list[-1][:, :, metric_readout_token_idx, :].mean(dim=1).float()
        metric_logits = self.metric_readout_head(metric_readout_features).squeeze(-1)
        predicted_metric_scale = torch.exp(metric_logits)
        return predicted_metric_scale, metric_readout_features

    def _scale_chunk_camera_maps(self, chunk_cam_maps, scale_factor: torch.Tensor, B: int, F: int):
        if scale_factor is None:
            return chunk_cam_maps
        row_scale = scale_factor.view(B, 1, 1).expand(B, F, 1).reshape(B * F, 1, 1)
        scaled_chunk_cam_maps = []
        for chunk_cam_map in chunk_cam_maps:
            row_scale_typed = row_scale.to(dtype=chunk_cam_map.dtype, device=chunk_cam_map.device)
            scaled_translation = chunk_cam_map[..., :3] * row_scale_typed
            scaled_chunk_cam_maps.append(torch.cat([scaled_translation, chunk_cam_map[..., 3:]], dim=-1))
        return scaled_chunk_cam_maps

    def load_component_pretrained(self):
        if self.agg_regator_ckpt and os.path.exists(self.agg_regator_ckpt):
            self.agg_regator.load_state_dict(
                torch.load(self.agg_regator_ckpt, map_location="cpu", weights_only=False),
                strict=False,
            )
            if hasattr(self.agg_regator, "sync_gla_from_global_blocks"):
                self.agg_regator.sync_gla_from_global_blocks()
        if self.cam_decoder_ckpt and os.path.exists(self.cam_decoder_ckpt):
            self.cam_decoder.load_state_dict(
                torch.load(self.cam_decoder_ckpt, map_location="cpu", weights_only=False),
                strict=False,
            )
        if self.dpt_decoder_ckpt and os.path.exists(self.dpt_decoder_ckpt):
            self.dpt_decoder.load_state_dict(
                torch.load(self.dpt_decoder_ckpt, map_location="cpu", weights_only=False),
                strict=False,
            )

    def forward_chunk(
        self,
        images: torch.Tensor,
        *,
        window_size: int,
        chunk_idx: int,
        frame_kv_caches,
        global_kv_caches,
        gla_cache,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, S, 3, H, W], got {tuple(images.shape)}")
        B, S, _, _, _ = images.shape
        if B != 1:
            raise ValueError("HorizonStream window inference currently supports batch size 1 only.")
        Win = int(window_size.item()) if hasattr(window_size, "item") else int(window_size)
        output_dict, win_pose_tokens, patch_start_idx = self.agg_regator(
            images,
            frame_kv_caches=frame_kv_caches,
            global_kv_caches=global_kv_caches,
            n_views=1,
            window_size=Win,
            chunk_idx=chunk_idx,
            rope_frame_start=((chunk_idx * S) % self.rope_temporal_period) if self.rope_temporal_period > 0 else 0,
            gla_cache=gla_cache,
        )
        aggregated_tokens_list = self._output_dict_to_list(output_dict)
        chunk_cam_maps_raw = self.cam_decoder(win_pose_tokens, chunk_idx=chunk_idx)

        predicted_metric_scale, metric_readout_features = self._predict_metric_scale(
            aggregated_tokens_list,
            patch_start_idx,
        )
        chunk_cam_maps = self._scale_chunk_camera_maps(chunk_cam_maps_raw, predicted_metric_scale, B, S)

        depth, depth_conf = self.dpt_decoder(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            frames_chunk_size=self.frames_chunk_size,
        )

        if predicted_metric_scale is not None:
            scale = predicted_metric_scale.to(dtype=depth.dtype, device=depth.device).view(B, 1, 1, 1, 1)
            depth = depth * scale

        outputs = {
            "chunk_cam_map": chunk_cam_maps[-1],
            "depth": depth,
            "depth_conf": depth_conf,
        }
        if chunk_idx == 0 and S == Win:
            cam_dim = chunk_cam_maps[-1].shape[-1]
            outputs["pose_enc"] = chunk_cam_maps[-1].reshape(B, -1, Win, cam_dim)[:, -1, :, :9]
        if predicted_metric_scale is not None:
            outputs["predicted_metric_scale"] = predicted_metric_scale
            outputs["metric_readout_features"] = metric_readout_features
        return outputs

    def forward_window(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, S, _, _, _ = images.shape
        state = self.build_sequence_state()
        outputs = self.forward_chunk(
            images,
            window_size=S,
            chunk_idx=0,
            frame_kv_caches=state["frame_kv_caches"],
            global_kv_caches=state["global_kv_caches"],
            gla_cache=state.get("gla_cache"),
        )
        self.advance_sequence_state(state, is_last_chunk=True)
        return outputs

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_window(images)
