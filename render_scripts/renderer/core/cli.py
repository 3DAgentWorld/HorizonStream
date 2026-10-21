"""Command-line entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm.auto import tqdm

from .configuration import load_settings
from .pipeline import SceneRenderPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a reconstructed scene")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", default="render.mp4")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--skip_first_frames", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=None)
    parser.add_argument("--disable_sky_mask", action="store_true")
    parser.add_argument("--disable_glb_export", action="store_true")
    parser.add_argument("--disable_minimap", action="store_true")
    parser.add_argument("--keep_frame_images", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_config = Path(__file__).resolve().parents[2] / "configs" / "render_config.yaml"
    cfg = load_settings(args.config or default_config)
    cfg.io.input_path = args.input_path
    cfg.io.output_path = args.output_path
    if args.width is not None:
        cfg.render.width = args.width
    if args.height is not None:
        cfg.render.height = args.height
    if args.fps is not None:
        cfg.io.fps = args.fps
    if args.skip_first_frames is not None:
        cfg.io.skip_first_frames = args.skip_first_frames
    if args.max_frames is not None:
        cfg.io.max_frames = args.max_frames
    if args.frame_step is not None:
        cfg.io.frame_step = args.frame_step
    if args.disable_sky_mask:
        cfg.filters.sky_mask_enabled = False
    if args.disable_glb_export:
        cfg.export.glb_enabled = False
    if args.disable_minimap:
        cfg.minimap.enabled = False
    if args.keep_frame_images:
        cfg.io.keep_frame_images = True

    def _log(msg: str) -> None:
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")

    SceneRenderPipeline(cfg, log=_log).run()
    return 0
