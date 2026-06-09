import argparse
import copy
import json
import os
from typing import Any, Dict, Optional

import infer as infer_cli
from horizonstream.core.infer import run_inference_cfg
from horizonstream.eval.trajectory import evaluate_output_root
from horizonstream.loop.online_loop_reinfer import run_online_loop_reinfer
from horizonstream.utils.loop_assets import ensure_loop_assets, print_asset_warnings


def build_parser() -> argparse.ArgumentParser:
    parser = infer_cli.build_parser()
    parser.description = "Run HorizonStream inference, optional loop closure, and trajectory evaluation."

    def add_optional(*flags, **kwargs):
        if any(flag in parser._option_string_actions for flag in flags):
            return
        parser.add_argument(*flags, **kwargs)

    parser.add_argument("--with-loop", action="store_true", help="Run online loop closure after inference.")
    parser.add_argument("--skip-infer", action="store_true", help="Use an existing --output-root instead of running inference.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not run trajectory evaluation.")
    parser.add_argument("--loop-output-root", default=None)
    add_optional("--loop-preset", default=None, help="Preset name from online_loop_groups.")
    add_optional("--loop-methods", default=None, help="Comma-separated methods: dbow,salad")
    add_optional("--dbow-vocab-path", default=None)
    add_optional("--dbow-backend", choices=["auto", "python", "dpretrieval"], default=None)
    add_optional("--salad-ckpt-path", default=None)
    add_optional("--salad-dino-weights-path", default=None)
    add_optional("--salad-score-thresh", type=float, default=None)
    add_optional("--retrieval-top-k", type=int, default=None)
    add_optional("--temporal-exclusion", type=int, default=None)
    add_optional("--min-frame-separation", type=int, default=None)
    add_optional("--nms-radius", type=int, default=None)
    add_optional("--loop-chunk-size", type=int, default=None)
    add_optional("--max-candidates", type=int, default=None)
    add_optional("--max-reinfer-candidates", type=int, default=None)
    add_optional("--loop-edge-score-threshold", type=float, default=None)
    add_optional("--pose-graph-backend", choices=["pypose", "scipy"], default=None)
    add_optional("--pose-graph-model", choices=["sim3", "se3"], default=None)
    add_optional("--pose-graph-loop-weight", type=float, default=None)
    add_optional("--pose-graph-trans-weight", type=float, default=None)
    add_optional("--pose-graph-rot-weight", type=float, default=None)
    add_optional("--pose-graph-inlier-cap", type=int, default=None)
    parser.add_argument("--skip-loop-pointcloud", action="store_true")
    parser.add_argument("--no-auto-loop-output-suffix", action="store_true")
    parser.add_argument("--gt-img-path", default=None)
    parser.add_argument("--gt-seq-list", default=None)
    parser.add_argument("--gt-seq-match-mode", default=None)
    parser.add_argument("--gt-camera", default=None)
    parser.add_argument(
        "--eval-pose-variants",
        default="main,offline,loop",
        help="Comma-separated pose variants for trajectory evaluation.",
    )
    return parser


def _comma_list(text: Optional[str]):
    if text is None:
        return None
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _apply_gt_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if not any(
        value is not None
        for value in (args.gt_img_path, args.gt_seq_list, args.gt_seq_match_mode, args.gt_camera)
    ):
        return
    gt_data = dict(cfg.get("data", {}) or {})
    if args.gt_img_path is not None:
        gt_data["img_path"] = args.gt_img_path
    if args.gt_seq_list is not None:
        gt_data["seq_list"] = _comma_list(args.gt_seq_list)
    if args.gt_seq_match_mode is not None:
        gt_data["seq_match_mode"] = args.gt_seq_match_mode
    if args.gt_camera is not None:
        gt_data["camera"] = args.gt_camera
    gt_data["format"] = "generalizable"
    cfg["gt_data"] = gt_data


def _apply_loop_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("online_loop", {})
    loop = cfg["online_loop"]
    if args.loop_preset is not None:
        groups = cfg.get("online_loop_groups", {}) or {}
        if args.loop_preset not in groups:
            raise ValueError(
                f"Unknown --loop-preset {args.loop_preset!r}; available presets: {sorted(groups)}"
            )
        loop.update(groups[args.loop_preset] or {})
    if args.loop_methods is not None:
        loop["methods"] = _comma_list(args.loop_methods)
    if args.dbow_vocab_path is not None:
        loop["dbow_vocab_path"] = args.dbow_vocab_path
    if args.dbow_backend is not None:
        loop["dbow_backend"] = args.dbow_backend
    if args.salad_ckpt_path is not None:
        loop["salad_ckpt_path"] = args.salad_ckpt_path
    if args.salad_dino_weights_path is not None:
        loop["salad_dino_weights_path"] = args.salad_dino_weights_path
    if args.salad_score_thresh is not None:
        loop["salad_score_thresh"] = float(args.salad_score_thresh)
    if args.retrieval_top_k is not None:
        loop["retrieval_top_k"] = int(args.retrieval_top_k)
    if args.temporal_exclusion is not None:
        loop["temporal_exclusion"] = int(args.temporal_exclusion)
    if args.min_frame_separation is not None:
        loop["min_frame_separation"] = int(args.min_frame_separation)
    if args.nms_radius is not None:
        loop["nms_radius"] = int(args.nms_radius)
    if args.loop_chunk_size is not None:
        loop["loop_chunk_size"] = int(args.loop_chunk_size)
    if args.max_candidates is not None:
        loop["max_candidates"] = int(args.max_candidates)
    if args.max_reinfer_candidates is not None:
        loop["max_reinfer_candidates"] = int(args.max_reinfer_candidates)
    if args.loop_edge_score_threshold is not None:
        loop["loop_edge_score_threshold"] = float(args.loop_edge_score_threshold)
    if args.pose_graph_backend is not None:
        loop["pose_graph_backend"] = args.pose_graph_backend
    if args.pose_graph_model is not None:
        loop["pose_graph_model"] = args.pose_graph_model
    if args.pose_graph_loop_weight is not None:
        loop["pose_graph_loop_weight"] = float(args.pose_graph_loop_weight)
    if args.pose_graph_trans_weight is not None:
        loop["pose_graph_trans_weight"] = float(args.pose_graph_trans_weight)
    if args.pose_graph_rot_weight is not None:
        loop["pose_graph_rot_weight"] = float(args.pose_graph_rot_weight)
    if args.pose_graph_inlier_cap is not None:
        loop["pose_graph_inlier_cap"] = int(args.pose_graph_inlier_cap)
    if args.no_auto_loop_output_suffix:
        loop["auto_output_suffix"] = False


def _run_eval(output_root: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    variants = _comma_list(args.eval_pose_variants) or ["main", "offline", "loop"]
    data_cfg = cfg.get("gt_data", cfg.get("data", {}))
    return evaluate_output_root(
        output_root,
        data_cfg=data_cfg,
        variants=variants,
        verbose=True,
        save=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = infer_cli.load_config_with_overrides(args)
    _apply_gt_overrides(cfg, args)
    _apply_loop_overrides(cfg, args)
    cfg = infer_cli.prepare_video_input_if_needed(cfg)

    output_root = cfg.get("output", {}).get("root", "outputs_horizonstream")
    output_root = os.path.abspath(os.path.expanduser(output_root))
    pipeline_summary: Dict[str, Any] = {
        "output_root": output_root,
        "ran_infer": False,
        "ran_loop": False,
        "ran_eval": False,
    }

    if not args.skip_infer:
        run_inference_cfg(cfg)
        pipeline_summary["ran_infer"] = True

    if not args.skip_eval:
        pipeline_summary["infer_eval"] = _run_eval(output_root, cfg, args)
        pipeline_summary["ran_eval"] = True

    if args.with_loop:
        loop_cfg = copy.deepcopy(cfg)
        loop_cfg.setdefault("online_loop", {})
        loop_cfg["online_loop"]["reuse_online_poses"] = True
        loop_cfg["online_loop"]["online_output_root"] = output_root
        loop_cfg["online_loop"]["auto_output_suffix"] = False
        loop_cfg.setdefault("output", {})["root"] = output_root
        output_cfg = cfg.get("output", {})
        if (
            args.skip_loop_pointcloud
            or not bool(output_cfg.get("save_points", True))
            or not bool(output_cfg.get("save_depth", True))
        ):
            loop_cfg["online_loop"]["rebuild_pointcloud"] = False
        if args.loop_output_root is not None:
            print(
                "[horizonstream-pipeline] --loop-output-root is ignored; "
                "loop results are written into --output-root",
                flush=True,
            )
        ok, messages = ensure_loop_assets(loop_cfg["online_loop"], auto_download=True)
        if not ok:
            print_asset_warnings(messages)
            pipeline_summary["loop_skipped"] = True
            pipeline_summary["loop_skip_reason"] = messages
        else:
            loop_summary = run_online_loop_reinfer(loop_cfg)
            loop_output_root = loop_summary.get(
                "output_root",
                loop_cfg.get("output", {}).get("root", output_root),
            )
            pipeline_summary["ran_loop"] = True
            pipeline_summary["loop_output_root"] = loop_output_root
            pipeline_summary["loop_summary"] = loop_summary
            if not args.skip_eval:
                pipeline_summary["loop_eval"] = _run_eval(loop_output_root, cfg, args)

    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)
    print(f"[horizonstream-pipeline] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
