from __future__ import annotations

import os
import urllib.request
from typing import Iterable, List, Tuple


SALAD_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
DINO_VITB14_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def path_exists(path: str) -> bool:
    return bool(path) and os.path.exists(expand_path(path))


def download_file(url: str, path: str, *, label: str = "asset") -> str:
    local_path = expand_path(path)
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[horizonstream] downloading {label}: {url} -> {local_path}", flush=True)
    tmp_path = local_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, local_path)
    return local_path


def ensure_loop_assets(loop_cfg: dict, *, auto_download: bool = True) -> Tuple[bool, List[str]]:
    methods = {str(m).strip().lower() for m in (loop_cfg.get("methods") or []) if str(m).strip()}
    messages: List[str] = []

    if "salad" in methods:
        salad_ckpt = str(loop_cfg.get("salad_ckpt_path", "") or "")
        dino_weights = str(loop_cfg.get("salad_dino_weights_path", "") or "")
        if auto_download:
            try:
                if salad_ckpt and not path_exists(salad_ckpt):
                    download_file(SALAD_CKPT_URL, salad_ckpt, label="loop checkpoint")
                if dino_weights and not path_exists(dino_weights):
                    download_file(DINO_VITB14_URL, dino_weights, label="loop backbone weights")
            except Exception as exc:
                messages.append(f"failed to download loop assets: {exc}")
        missing = [p for p in (salad_ckpt, dino_weights) if p and not path_exists(p)]
        if missing:
            messages.append(
                "loop assets are missing: "
                + ", ".join(missing)
                + ". Loop closure will be skipped."
            )

    return not messages, messages


def print_asset_warnings(messages: Iterable[str]) -> None:
    for message in messages:
        print(f"[horizonstream] warning: {message}", flush=True)
