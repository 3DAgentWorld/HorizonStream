from __future__ import annotations

import os
from typing import Any, Dict

import torch

from horizonstream.models.horizonstream import HorizonStream
from horizonstream.utils.hub import resolve_checkpoint_path


class HorizonStreamModel(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any] | None):
        super().__init__()
        cfg = cfg or {}

        ckpt_path = resolve_checkpoint_path(
            cfg.get("checkpoint", None),
            cfg.get("hf", None),
        )

        model_cfg = dict(cfg.get("horizonstream_cfg", {}) or {})
        self.horizonstream = HorizonStream(**model_cfg)

        if ckpt_path:
            self.load_checkpoint(ckpt_path, strict=bool(cfg.get("strict_load", True)))

    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            raise TypeError("Unsupported checkpoint format")

        def strip_wrappers(key: str) -> str:
            # Training checkpoints may be wrapped by several container prefixes.
            # Strip them repeatedly so we can match both wrapper and bare module states.
            prefixes = ("module.", "model.", "state_dict.", "sampler.")
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        changed = True
            return key

        normalized_state = {strip_wrappers(k): v for k, v in state.items()}

        full_keys = set(self.state_dict().keys())
        core_keys = set(self.horizonstream.state_dict().keys())
        full_matches = sum(1 for k in normalized_state.keys() if k in full_keys)
        core_matches = sum(1 for k in normalized_state.keys() if k in core_keys)

        if full_matches == 0 and core_matches > 0:
            # Checkpoint is likely dumped from sampler/core module directly.
            # Remove an optional horizonstream. prefix if it still exists.
            core_state = {
                k.removeprefix("horizonstream."): v for k, v in normalized_state.items()
            }
            missing, unexpected = self.horizonstream.load_state_dict(core_state, strict=False)
        else:
            missing, unexpected = self.load_state_dict(normalized_state, strict=False)

        if missing or unexpected:
            msg = (
                "checkpoint mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"(full_matches={full_matches}, core_matches={core_matches})"
            )
            if missing:
                sample_missing = ", ".join(missing[:20])
                msg += f"\n  missing(sample up to 20): {sample_missing}"
            if unexpected:
                sample_unexpected = ", ".join(unexpected[:20])
                msg += f"\n  unexpected(sample up to 20): {sample_unexpected}"
            if strict:
                raise RuntimeError(msg)
            print(msg)

    def forward_window(self, images: torch.Tensor):
        return self.horizonstream.forward_window(images)

    def build_sequence_state(self):
        return self.horizonstream.build_sequence_state()

    def advance_sequence_state(self, state, *, is_last_chunk: bool):
        self.horizonstream.advance_sequence_state(state, is_last_chunk=is_last_chunk)

    def forward_chunk(
        self,
        images: torch.Tensor,
        *,
        window_size: int,
        chunk_idx: int,
        state,
    ):
        return self.horizonstream.forward_chunk(
            images,
            window_size=window_size,
            chunk_idx=chunk_idx,
            frame_kv_caches=state["frame_kv_caches"],
            global_kv_caches=state["global_kv_caches"],
            gla_cache=state.get("gla_cache"),
        )

    def forward(self, images: torch.Tensor):
        return self.forward_window(images)
