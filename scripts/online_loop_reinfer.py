import argparse
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from horizonstream.loop.online_loop_reinfer import run_online_loop_reinfer
from horizonstream.utils.loop_assets import ensure_loop_assets, print_asset_warnings


def default_config_path() -> str:
    return os.path.join(REPO_ROOT, "configs", "horizonstream_infer.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone online HorizonStream loop closure with loop-window re-inference."
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--device", default=None)
    parser.add_argument("--img-path", default=None)
    parser.add_argument("--seq-list", default=None)
    parser.add_argument("--seq-match-mode", default=None)
    parser.add_argument("--format", default=None)
    parser.add_argument("--data-roots-file", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--gt-img-path", default=None)
    parser.add_argument("--gt-seq-list", default=None)
    parser.add_argument("--gt-seq-match-mode", default=None)
    parser.add_argument("--gt-camera", default=None)
    parser.add_argument("--camera-preprocess", action="store_true")
    parser.add_argument("--no-camera-preprocess", action="store_true")
    parser.add_argument("--camera-preprocess-strict", action="store_true")
    parser.add_argument("--no-camera-preprocess-strict", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-file", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--sliding-size", type=int, default=None)

    parser.add_argument("--loop-preset", default=None, help="Preset name from online_loop_groups.")
    parser.add_argument("--loop-methods", default=None, help="Comma-separated methods: dbow,salad")
    parser.add_argument("--dbow-vocab-path", default=None)
    parser.add_argument("--dbow-backend", choices=["auto", "python", "dpretrieval"], default=None)
    parser.add_argument("--salad-ckpt-path", default=None)
    parser.add_argument("--salad-dino-weights-path", default=None)
    parser.add_argument("--salad-backbone", default=None)
    parser.add_argument("--salad-image-size", default=None, help="HxW, e.g. 336x336")
    parser.add_argument("--salad-batch-size", type=int, default=None)
    parser.add_argument("--salad-score-thresh", type=float, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--dbow-thresh", type=float, default=None)
    parser.add_argument("--dbow-num-repeat", type=int, default=None)
    parser.add_argument("--temporal-exclusion", type=int, default=None)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--min-frame-separation", type=int, default=None)
    parser.add_argument("--loop-chunk-size", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-reinfer-candidates", type=int, default=None)
    parser.add_argument("--dbow-direct-loop-edges", action="store_true")
    parser.add_argument("--dbow-direct-edge-mode", default=None)
    parser.add_argument("--dbow-direct-max-edges", type=int, default=None)
    parser.add_argument("--dbow-direct-inliers", type=int, default=None)
    parser.add_argument("--loop-edge-score-threshold", type=float, default=None)
    parser.add_argument("--reuse-online-poses", action="store_true")
    parser.add_argument(
        "--pose-graph-backend",
        choices=["pypose", "scipy"],
        default=None,
    )
    parser.add_argument("--pose-graph-model", choices=["sim3", "se3"], default=None)
    parser.add_argument(
        "--pose-graph-update-mode",
        choices=["all", "translation_only", "translation_scale"],
        default=None,
    )
    parser.add_argument("--pose-graph-loop-weight", type=float, default=None)
    parser.add_argument("--pose-graph-trans-weight", type=float, default=None)
    parser.add_argument("--pose-graph-rot-weight", type=float, default=None)
    parser.add_argument("--pose-graph-scale-weight", type=float, default=None)
    parser.add_argument("--pose-graph-inlier-cap", type=int, default=None)
    parser.add_argument("--pose-graph-max-iterations", type=int, default=None)
    parser.add_argument("--pose-graph-lambda-init", type=float, default=None)
    parser.add_argument("--pose-graph-solver-verbose", action="store_true")
    parser.add_argument("--retrieval-log-every", type=int, default=None)
    parser.add_argument("--no-auto-output-suffix", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--online-output-root", default=None,
                        help="Online infer output root to read depth/rgb for pointcloud rebuild")
    parser.add_argument("--rebuild-pointcloud", action="store_true", default=None)
    parser.add_argument("--skip-pointcloud-rebuild", action="store_true")
    parser.add_argument("--max-full-pointcloud-points", type=int, default=None)
    parser.add_argument("--max-frame-pointcloud-points", type=int, default=None)
    parser.add_argument("--point-mask-sky", action="store_true")
    parser.add_argument("--no-point-mask-sky", action="store_true")
    parser.add_argument("--point-depth-min", type=float, default=None)
    parser.add_argument("--point-depth-max", type=float, default=None)
    parser.add_argument("--point-depth-percentile-min", type=float, default=None)
    parser.add_argument("--point-depth-percentile-max", type=float, default=None)
    parser.add_argument("--point-sky-color-filter", action="store_true")
    parser.add_argument("--no-point-sky-color-filter", action="store_true")
    parser.add_argument("--point-sky-white-threshold", type=int, default=None)
    parser.add_argument("--point-sky-bright-any-threshold", type=int, default=None)
    parser.add_argument("--point-sky-blue-b-min", type=int, default=None)
    parser.add_argument("--point-sky-blue-r-max", type=int, default=None)
    parser.add_argument("--point-sky-blue-g-min", type=int, default=None)
    parser.add_argument("--point-sky-blue-rg-diff-max", type=int, default=None)
    parser.add_argument("--point-sky-blue-bg-diff-min", type=int, default=None)
    parser.add_argument("--point-sky-blue-br-diff-min", type=int, default=None)
    parser.add_argument("--point-outlier-filter", action="store_true")
    parser.add_argument("--no-point-outlier-filter", action="store_true")
    parser.add_argument("--point-radius-outlier-radius", type=float, default=None)
    parser.add_argument("--point-radius-outlier-min-points", type=int, default=None)
    parser.add_argument("--point-voxel-size", type=float, default=None)
    parser.add_argument("--point-random-sample-ratio", type=float, default=None)
    parser.add_argument("--save-frame-points", action="store_true")
    parser.add_argument("--no-save-frame-points", action="store_true")
    return parser


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("output", {})
    cfg.setdefault("online_loop", {})

    if args.loop_preset is not None:
        groups = cfg.get("online_loop_groups", {}) or {}
        if args.loop_preset not in groups:
            raise ValueError(
                f"Unknown --loop-preset {args.loop_preset!r}; "
                f"available presets: {sorted(groups)}"
            )
        cfg["online_loop"].update(groups[args.loop_preset] or {})

    if args.device is not None:
        cfg["device"] = args.device
    if args.img_path is not None:
        cfg["data"]["img_path"] = args.img_path
    if args.seq_list is not None:
        cfg["data"]["seq_list"] = [item.strip() for item in args.seq_list.split(",") if item.strip()]
    if args.seq_match_mode is not None:
        cfg["data"]["seq_match_mode"] = args.seq_match_mode
    if args.format is not None:
        cfg["data"]["format"] = args.format
    if args.data_roots_file is not None:
        cfg["data"]["data_roots_file"] = args.data_roots_file
    if args.camera is not None:
        cfg["data"]["camera"] = args.camera
    if args.camera_preprocess:
        cfg["data"]["camera_preprocess"] = True
    if args.no_camera_preprocess:
        cfg["data"]["camera_preprocess"] = False
    if args.camera_preprocess_strict:
        cfg["data"]["camera_preprocess_strict"] = True
    if args.no_camera_preprocess_strict:
        cfg["data"]["camera_preprocess_strict"] = False
    if args.max_frames is not None and args.max_frames > 0:
        cfg["data"]["max_frames"] = args.max_frames
    if any(
        value is not None
        for value in (args.gt_img_path, args.gt_seq_list, args.gt_seq_match_mode, args.gt_camera)
    ):
        gt_data = dict(cfg["data"])
        if args.gt_img_path is not None:
            gt_data["img_path"] = args.gt_img_path
        if args.gt_seq_list is not None:
            gt_data["seq_list"] = [
                item.strip() for item in args.gt_seq_list.split(",") if item.strip()
            ]
        if args.gt_seq_match_mode is not None:
            gt_data["seq_match_mode"] = args.gt_seq_match_mode
        if args.gt_camera is not None:
            gt_data["camera"] = args.gt_camera
        cfg["gt_data"] = gt_data
    if args.output_root is not None:
        cfg["output"]["root"] = args.output_root
    if args.checkpoint is not None:
        cfg["model"]["checkpoint"] = args.checkpoint
    if args.hf_repo is not None or args.hf_file is not None:
        cfg["model"].setdefault("hf", {})
        if args.hf_repo is not None:
            cfg["model"]["hf"]["repo_id"] = args.hf_repo
        if args.hf_file is not None:
            cfg["model"]["hf"]["filename"] = args.hf_file
        if args.checkpoint is None:
            cfg["model"]["checkpoint"] = None
    if args.window_size is not None:
        cfg["inference"]["window_size"] = int(args.window_size)
    if args.sliding_size is not None:
        cfg["inference"]["sliding_size"] = int(args.sliding_size)

    loop = cfg["online_loop"]
    if args.loop_methods is not None:
        loop["methods"] = [item.strip() for item in args.loop_methods.split(",") if item.strip()]
    if args.dbow_vocab_path is not None:
        loop["dbow_vocab_path"] = args.dbow_vocab_path
    if args.dbow_backend is not None:
        loop["dbow_backend"] = args.dbow_backend
    if args.salad_ckpt_path is not None:
        loop["salad_ckpt_path"] = args.salad_ckpt_path
    if args.salad_dino_weights_path is not None:
        loop["salad_dino_weights_path"] = args.salad_dino_weights_path
    if args.salad_backbone is not None:
        loop["salad_backbone"] = args.salad_backbone
    if args.salad_image_size is not None:
        parts = args.salad_image_size.lower().replace(",", "x").split("x")
        if len(parts) != 2:
            raise ValueError("--salad-image-size must be HxW, e.g. 336x336")
        loop["salad_image_size"] = [int(parts[0]), int(parts[1])]
    if args.salad_batch_size is not None:
        loop["salad_batch_size"] = int(args.salad_batch_size)
    if args.salad_score_thresh is not None:
        loop["salad_score_thresh"] = float(args.salad_score_thresh)
    if args.retrieval_top_k is not None:
        loop["retrieval_top_k"] = int(args.retrieval_top_k)
    if args.dbow_thresh is not None:
        loop["dbow_thresh"] = float(args.dbow_thresh)
    if args.dbow_num_repeat is not None:
        loop["dbow_num_repeat"] = int(args.dbow_num_repeat)
    if args.temporal_exclusion is not None:
        loop["temporal_exclusion"] = int(args.temporal_exclusion)
    if args.nms_radius is not None:
        loop["nms_radius"] = int(args.nms_radius)
    if args.min_frame_separation is not None:
        loop["min_frame_separation"] = int(args.min_frame_separation)
    if args.loop_chunk_size is not None:
        loop["loop_chunk_size"] = int(args.loop_chunk_size)
    if args.max_candidates is not None:
        loop["max_candidates"] = int(args.max_candidates)
    if args.max_reinfer_candidates is not None:
        loop["max_reinfer_candidates"] = int(args.max_reinfer_candidates)
    if args.dbow_direct_loop_edges:
        loop["dbow_direct_loop_edges"] = True
    if args.dbow_direct_edge_mode is not None:
        loop["dbow_direct_edge_mode"] = str(args.dbow_direct_edge_mode)
    if args.dbow_direct_max_edges is not None:
        loop["dbow_direct_max_edges"] = int(args.dbow_direct_max_edges)
    if args.dbow_direct_inliers is not None:
        loop["dbow_direct_inliers"] = int(args.dbow_direct_inliers)
    if args.loop_edge_score_threshold is not None:
        loop["loop_edge_score_threshold"] = float(args.loop_edge_score_threshold)
    if args.reuse_online_poses:
        loop["reuse_online_poses"] = True
    if args.pose_graph_backend is not None:
        loop["pose_graph_backend"] = args.pose_graph_backend
    if args.pose_graph_model is not None:
        loop["pose_graph_model"] = args.pose_graph_model
    if args.pose_graph_update_mode is not None:
        loop["pose_graph_update_mode"] = args.pose_graph_update_mode
    if args.pose_graph_loop_weight is not None:
        loop["pose_graph_loop_weight"] = float(args.pose_graph_loop_weight)
    if args.pose_graph_trans_weight is not None:
        loop["pose_graph_trans_weight"] = float(args.pose_graph_trans_weight)
    if args.pose_graph_rot_weight is not None:
        loop["pose_graph_rot_weight"] = float(args.pose_graph_rot_weight)
    if args.pose_graph_scale_weight is not None:
        loop["pose_graph_scale_weight"] = float(args.pose_graph_scale_weight)
    if args.pose_graph_inlier_cap is not None:
        loop["pose_graph_inlier_cap"] = int(args.pose_graph_inlier_cap)
    if args.pose_graph_max_iterations is not None:
        loop["pose_graph_max_iterations"] = int(args.pose_graph_max_iterations)
    if args.pose_graph_lambda_init is not None:
        loop["pose_graph_lambda_init"] = float(args.pose_graph_lambda_init)
    if args.pose_graph_solver_verbose:
        loop["pose_graph_solver_verbose"] = True
    if args.retrieval_log_every is not None:
        loop["retrieval_log_every"] = int(args.retrieval_log_every)
    if args.no_auto_output_suffix:
        loop["auto_output_suffix"] = False
    if args.quiet:
        loop["verbose"] = False
    if args.online_output_root is not None:
        cfg["online_loop"]["online_output_root"] = args.online_output_root
    if args.rebuild_pointcloud is True:
        cfg["online_loop"]["rebuild_pointcloud"] = True
    if args.skip_pointcloud_rebuild:
        cfg["online_loop"]["rebuild_pointcloud"] = False
    if args.max_full_pointcloud_points is not None:
        cfg["output"]["max_full_pointcloud_points"] = int(args.max_full_pointcloud_points)
    if args.max_frame_pointcloud_points is not None:
        cfg["output"]["max_frame_pointcloud_points"] = int(args.max_frame_pointcloud_points)
    if args.point_mask_sky:
        cfg["output"]["point_mask_sky"] = True
    if args.no_point_mask_sky:
        cfg["output"]["point_mask_sky"] = False
    if args.point_depth_min is not None:
        cfg["output"]["point_depth_min"] = float(args.point_depth_min)
    if args.point_depth_max is not None:
        cfg["output"]["point_depth_max"] = float(args.point_depth_max)
    if args.point_depth_percentile_min is not None:
        cfg["output"]["point_depth_percentile_min"] = float(args.point_depth_percentile_min)
    if args.point_depth_percentile_max is not None:
        cfg["output"]["point_depth_percentile_max"] = float(args.point_depth_percentile_max)
    if args.point_sky_color_filter:
        cfg["output"]["point_sky_color_filter"] = True
    if args.no_point_sky_color_filter:
        cfg["output"]["point_sky_color_filter"] = False
    point_sky_color_args = {
        "point_sky_white_threshold": args.point_sky_white_threshold,
        "point_sky_bright_any_threshold": args.point_sky_bright_any_threshold,
        "point_sky_blue_b_min": args.point_sky_blue_b_min,
        "point_sky_blue_r_max": args.point_sky_blue_r_max,
        "point_sky_blue_g_min": args.point_sky_blue_g_min,
        "point_sky_blue_rg_diff_max": args.point_sky_blue_rg_diff_max,
        "point_sky_blue_bg_diff_min": args.point_sky_blue_bg_diff_min,
        "point_sky_blue_br_diff_min": args.point_sky_blue_br_diff_min,
    }
    for key, value in point_sky_color_args.items():
        if value is not None:
            cfg["output"][key] = value
    if args.point_outlier_filter:
        cfg["output"]["point_outlier_filter"] = True
    if args.no_point_outlier_filter:
        cfg["output"]["point_outlier_filter"] = False
    if args.point_radius_outlier_radius is not None:
        cfg["output"]["point_radius_outlier_radius"] = float(args.point_radius_outlier_radius)
    if args.point_radius_outlier_min_points is not None:
        cfg["output"]["point_radius_outlier_min_points"] = int(args.point_radius_outlier_min_points)
    if args.point_voxel_size is not None:
        cfg["output"]["point_voxel_size"] = float(args.point_voxel_size)
    if args.point_random_sample_ratio is not None:
        cfg["output"]["point_random_sample_ratio"] = float(args.point_random_sample_ratio)
    if args.save_frame_points:
        cfg["output"]["save_frame_points"] = True
    if args.no_save_frame_points:
        cfg["output"]["save_frame_points"] = False

    cfg["data"]["format"] = cfg["data"].get("format", "generalizable")
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config_with_overrides(args)
    ok, messages = ensure_loop_assets(cfg.get("online_loop", {}), auto_download=True)
    if not ok:
        print_asset_warnings(messages)
        return
    run_online_loop_reinfer(cfg)


if __name__ == "__main__":
    main()
