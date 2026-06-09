import argparse
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from horizonstream.eval.trajectory import evaluate_output_root


def default_config_path() -> str:
    return os.path.join(REPO_ROOT, "configs", "horizonstream_infer.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HorizonStream trajectory outputs against GT poses.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--img-path", default=None, help="GT/input generalizable meta root")
    parser.add_argument("--seq-list", default=None)
    parser.add_argument("--seq-match-mode", default=None)
    parser.add_argument("--data-roots-file", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument(
        "--pose-variants",
        default="main,offline,loop",
        help="Comma-separated pose variants to evaluate.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def load_data_cfg(args: argparse.Namespace) -> dict:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    data_cfg = dict(cfg.get("data", {}) or {})
    if args.img_path is not None:
        data_cfg["img_path"] = args.img_path
    if args.seq_list is not None:
        data_cfg["seq_list"] = [item.strip() for item in args.seq_list.split(",") if item.strip()]
    if args.seq_match_mode is not None:
        data_cfg["seq_match_mode"] = args.seq_match_mode
    if args.data_roots_file is not None:
        data_cfg["data_roots_file"] = args.data_roots_file
    if args.camera is not None:
        data_cfg["camera"] = args.camera
    data_cfg["format"] = "generalizable"
    return data_cfg


def main() -> None:
    args = build_parser().parse_args()
    variants = [item.strip() for item in args.pose_variants.split(",") if item.strip()]
    summary = evaluate_output_root(
        args.output_root,
        data_cfg=load_data_cfg(args),
        variants=variants,
        verbose=not args.quiet,
        save=True,
    )
    print(
        f"[horizonstream-eval] wrote {os.path.join(args.output_root, 'eval_summary.json')} "
        f"for {summary['num_sequences']} sequence(s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
