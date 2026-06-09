#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from horizonstream.utils.loop_assets import (  # noqa: E402
    DINO_VITB14_URL,
    SALAD_CKPT_URL,
    download_file,
)


HF_BASE = "https://huggingface.co/NicolasCC/HorizonStream/resolve/main"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download HorizonStream release weights and loop assets."
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--weights-dir", default="weights")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))
    weights_dir = os.path.abspath(os.path.expanduser(args.weights_dir))

    checkpoint_files = ["HorizonStream.pt", "skyseg.onnx"]

    for name in checkpoint_files:
        download_file(f"{HF_BASE}/{name}", os.path.join(ckpt_dir, name), label=name)

    download_file(
        SALAD_CKPT_URL,
        os.path.join(weights_dir, "dino_salad.ckpt"),
        label="loop checkpoint",
    )
    download_file(
        DINO_VITB14_URL,
        os.path.join(weights_dir, "dinov2_vitb14_pretrain.pth"),
        label="loop backbone weights",
    )

    print("[horizonstream] weights ready", flush=True)
    print(f"  checkpoints: {ckpt_dir}", flush=True)
    print(f"  loop weights: {weights_dir}", flush=True)


if __name__ == "__main__":
    main()
