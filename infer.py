import argparse
import copy
import os

import cv2
import yaml

from horizonstream.core.infer import run_inference_cfg
from horizonstream.utils.loop_assets import ensure_loop_assets, print_asset_warnings


def default_config_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "configs",
        "horizonstream_infer.yaml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--img-path", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for video frame sampling.")
    parser.add_argument("--video-stride", type=int, default=None)
    parser.add_argument("--video-scene-name", default=None)
    parser.add_argument("--video-cache-root", default=None)
    parser.add_argument("--video-overwrite", action="store_true")
    parser.add_argument("--no-video-overwrite", action="store_true")
    parser.add_argument("--seq-list", default=None)
    parser.add_argument("--seq-match-mode", default=None, help="exact | contains")
    parser.add_argument("--format", default=None, help="generalizable")
    parser.add_argument("--data-roots-file", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--abs-pose-source", choices=["online", "offline"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-file", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--sliding-size", type=int, default=None)
    parser.add_argument("--offload-outputs-to-cpu", action="store_true")
    parser.add_argument("--no-offload-outputs-to-cpu", action="store_true")
    parser.add_argument("--offline-motion-averaging", action="store_true")
    parser.add_argument("--no-offline-motion-averaging", action="store_true")
    parser.add_argument("--rope-temporal-period", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
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
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--no-save-points", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--no-save-depth", action="store_true")
    parser.add_argument("--save-depth-conf", action="store_true")
    parser.add_argument("--no-save-depth-conf", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--no-mask-sky", action="store_true")
    parser.add_argument("--camera-preprocess", action="store_true")
    parser.add_argument("--no-camera-preprocess", action="store_true")
    parser.add_argument("--camera-preprocess-strict", action="store_true")
    parser.add_argument("--no-camera-preprocess-strict", action="store_true")
    parser.add_argument("--no-loop", action="store_true", help="Disable loop closure after inference.")
    parser.add_argument("--no-loop-auto-download", action="store_true", help="Do not download missing loop assets automatically.")
    parser.add_argument("--loop-preset", default=None, help="Preset name from online_loop_groups.")
    parser.add_argument("--loop-methods", default="salad", help="Comma-separated loop retrieval methods.")
    parser.add_argument("--salad-ckpt-path", default=None)
    parser.add_argument("--salad-dino-weights-path", default=None)
    parser.add_argument("--salad-score-thresh", type=float, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--temporal-exclusion", type=int, default=None)
    parser.add_argument("--min-frame-separation", type=int, default=None)
    parser.add_argument("--loop-edge-score-threshold", type=float, default=None)
    parser.add_argument("--pose-graph-loop-weight", type=float, default=None)
    parser.add_argument("--pose-graph-trans-weight", type=float, default=None)
    parser.add_argument("--pose-graph-rot-weight", type=float, default=None)
    return parser


def load_config_with_overrides(args) -> dict:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("output", {})

    if args.device is not None:
        cfg["device"] = args.device

    if args.img_path is not None:
        cfg["data"]["img_path"] = args.img_path

    if args.video_path is not None:
        cfg["data"]["video_path"] = args.video_path

    if args.video_stride is not None:
        cfg["data"]["video_stride"] = int(args.video_stride)

    if args.fps is not None:
        cfg["data"]["video_sample_fps"] = float(args.fps)

    if args.video_scene_name is not None:
        cfg["data"]["video_scene_name"] = args.video_scene_name

    if args.video_cache_root is not None:
        cfg["data"]["video_cache_root"] = args.video_cache_root

    if args.video_overwrite:
        cfg["data"]["video_overwrite"] = True
    if args.no_video_overwrite:
        cfg["data"]["video_overwrite"] = False

    if args.seq_list is not None:
        cfg["data"]["seq_list"] = [s.strip() for s in args.seq_list.split(",") if s.strip()]

    if args.seq_match_mode is not None:
        cfg["data"]["seq_match_mode"] = args.seq_match_mode

    if args.format is not None:
        cfg["data"]["format"] = args.format

    if args.data_roots_file is not None:
        cfg["data"]["data_roots_file"] = args.data_roots_file

    if args.camera is not None:
        cfg["data"]["camera"] = args.camera

    if args.max_frames is not None and args.max_frames > 0:
        cfg["data"]["max_frames"] = args.max_frames

    if args.camera_preprocess:
        cfg["data"]["camera_preprocess"] = True
    if args.no_camera_preprocess:
        cfg["data"]["camera_preprocess"] = False
    if args.camera_preprocess_strict:
        cfg["data"]["camera_preprocess_strict"] = True
    if args.no_camera_preprocess_strict:
        cfg["data"]["camera_preprocess_strict"] = False

    if args.output_root is not None:
        cfg["output"]["root"] = args.output_root

    if args.abs_pose_source is not None:
        cfg["output"]["abs_pose_source"] = args.abs_pose_source

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
        cfg["inference"]["window_size"] = args.window_size

    if args.sliding_size is not None:
        cfg["inference"]["sliding_size"] = args.sliding_size

    if args.offload_outputs_to_cpu:
        cfg["inference"]["offload_outputs_to_cpu"] = True
    if args.no_offload_outputs_to_cpu:
        cfg["inference"]["offload_outputs_to_cpu"] = False

    if args.offline_motion_averaging:
        cfg["inference"]["enable_offline_motion_averaging"] = True
    if args.no_offline_motion_averaging:
        cfg["inference"]["enable_offline_motion_averaging"] = False

    if args.rope_temporal_period is not None:
        cfg.setdefault("model", {})
        cfg["model"].setdefault("horizonstream_cfg", {})
        cfg["model"]["horizonstream_cfg"].setdefault("agg_regator_cfg", {})
        cfg["model"]["horizonstream_cfg"]["agg_regator_cfg"]["rope_temporal_period"] = int(args.rope_temporal_period)

    if args.max_full_pointcloud_points is not None:
        cfg["output"]["max_full_pointcloud_points"] = args.max_full_pointcloud_points

    if args.max_frame_pointcloud_points is not None:
        cfg["output"]["max_frame_pointcloud_points"] = args.max_frame_pointcloud_points

    if args.point_mask_sky:
        cfg["output"]["point_mask_sky"] = True
    if args.no_point_mask_sky:
        cfg["output"]["point_mask_sky"] = False

    if args.point_depth_min is not None:
        cfg["output"]["point_depth_min"] = args.point_depth_min

    if args.point_depth_max is not None:
        cfg["output"]["point_depth_max"] = args.point_depth_max

    if args.point_depth_percentile_min is not None:
        cfg["output"]["point_depth_percentile_min"] = args.point_depth_percentile_min

    if args.point_depth_percentile_max is not None:
        cfg["output"]["point_depth_percentile_max"] = args.point_depth_percentile_max

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
        cfg["output"]["point_radius_outlier_radius"] = args.point_radius_outlier_radius
    if args.point_radius_outlier_min_points is not None:
        cfg["output"]["point_radius_outlier_min_points"] = args.point_radius_outlier_min_points

    if args.point_voxel_size is not None:
        cfg["output"]["point_voxel_size"] = args.point_voxel_size

    if args.point_random_sample_ratio is not None:
        cfg["output"]["point_random_sample_ratio"] = args.point_random_sample_ratio

    if args.save_frame_points:
        cfg["output"]["save_frame_points"] = True
    if args.no_save_frame_points:
        cfg["output"]["save_frame_points"] = False

    if args.save_videos:
        cfg["output"]["save_videos"] = True
    if args.no_save_videos:
        cfg["output"]["save_videos"] = False

    if args.save_points:
        cfg["output"]["save_points"] = True
    if args.no_save_points:
        cfg["output"]["save_points"] = False

    if args.save_depth:
        cfg["output"]["save_depth"] = True
    if args.no_save_depth:
        cfg["output"]["save_depth"] = False
    if args.save_depth_conf:
        cfg["output"]["save_depth_conf"] = True
    if args.no_save_depth_conf:
        cfg["output"]["save_depth_conf"] = False

    if args.save_images:
        cfg["output"]["save_images"] = True
    if args.no_save_images:
        cfg["output"]["save_images"] = False

    if args.mask_sky:
        cfg["output"]["mask_sky"] = True
    if args.no_mask_sky:
        cfg["output"]["mask_sky"] = False

    cfg["data"]["format"] = "generalizable"
    return cfg


def _comma_list(text):
    if text is None:
        return None
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _apply_loop_overrides(cfg: dict, args) -> None:
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
    if args.loop_edge_score_threshold is not None:
        loop["loop_edge_score_threshold"] = float(args.loop_edge_score_threshold)
    if args.pose_graph_loop_weight is not None:
        loop["pose_graph_loop_weight"] = float(args.pose_graph_loop_weight)
    if args.pose_graph_trans_weight is not None:
        loop["pose_graph_trans_weight"] = float(args.pose_graph_trans_weight)
    if args.pose_graph_rot_weight is not None:
        loop["pose_graph_rot_weight"] = float(args.pose_graph_rot_weight)


def run_loop_closure_cfg(cfg: dict, preflight_only: bool = False, auto_download: bool = True) -> bool:
    loop_cfg = copy.deepcopy(cfg)
    output_root = os.path.abspath(os.path.expanduser(loop_cfg.get("output", {}).get("root", "outputs_horizonstream")))
    loop_cfg.setdefault("online_loop", {})
    loop_cfg["online_loop"]["reuse_online_poses"] = True
    loop_cfg["online_loop"]["online_output_root"] = output_root
    loop_cfg["online_loop"]["auto_output_suffix"] = False
    loop_cfg.setdefault("output", {})["root"] = output_root
    output_cfg = loop_cfg.get("output", {})
    if not bool(output_cfg.get("save_points", True)) or not bool(output_cfg.get("save_depth", True)):
        loop_cfg["online_loop"]["rebuild_pointcloud"] = False

    ok, messages = ensure_loop_assets(loop_cfg["online_loop"], auto_download=auto_download)
    if not ok:
        print_asset_warnings(messages)
        return False
    if preflight_only:
        return True

    from horizonstream.loop.online_loop_reinfer import run_online_loop_reinfer

    run_online_loop_reinfer(loop_cfg)
    return True


def prepare_video_input_if_needed(cfg: dict) -> dict:
    data_cfg = cfg.setdefault("data", {})
    video_path = data_cfg.get("video_path", None)
    if video_path in (None, "", "null"):
        return cfg

    output_cfg = cfg.setdefault("output", {})
    stride = max(1, int(data_cfg.get("video_stride", 1)))
    target_fps = data_cfg.get("video_sample_fps", None)
    if target_fps not in (None, "", "null"):
        target_fps = float(target_fps)
        if output_cfg.get("video_fps", None) in (None, "", "null"):
            output_cfg["video_fps"] = target_fps
    elif output_cfg.get("video_fps", None) in (None, "", "null"):
        cap = cv2.VideoCapture(os.path.abspath(os.path.expanduser(str(video_path))))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
        cap.release()
        if source_fps > 0.0:
            output_cfg["video_fps"] = source_fps / float(stride)

    data_cfg["format"] = "video"
    data_cfg["camera_preprocess"] = False
    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config_with_overrides(args)
    _apply_loop_overrides(cfg, args)
    cfg = prepare_video_input_if_needed(cfg)
    run_loop = False
    if not args.no_loop:
        run_loop = run_loop_closure_cfg(
            cfg,
            preflight_only=True,
            auto_download=not args.no_loop_auto_download,
        )
    run_inference_cfg(cfg)
    if run_loop:
        run_loop_closure_cfg(cfg, auto_download=False)


if __name__ == "__main__":
    main()
