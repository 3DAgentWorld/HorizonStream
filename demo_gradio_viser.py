from __future__ import annotations

import os
import socket
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import yaml

from demo_viser_only import _visualize_sequence_dir_simple
from horizonstream.utils.loop_assets import ensure_loop_assets, print_asset_warnings


DEFAULT_CHECKPOINT = os.getenv("HORIZONSTREAM_CHECKPOINT", "checkpoints/HorizonStream.pt")
DEFAULT_DEVICE = os.getenv("HORIZONSTREAM_DEVICE", "cuda")
DEFAULT_MAX_FULL_POINTS = os.getenv("HORIZONSTREAM_DEMO_MAX_FULL_POINTS", "null")
DEFAULT_LOOP_PRESET = "kitti"
SESSION_ROOT = os.getenv(
    "HORIZONSTREAM_GRADIO_SESSION_ROOT",
    os.path.join(tempfile.gettempdir(), "horizonstream_gradio_sessions"),
)


def default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "configs", "horizonstream_infer.yaml")


def _new_session_dir() -> str:
    os.makedirs(SESSION_ROOT, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return tempfile.mkdtemp(prefix=f"horizonstream_{stamp}_", dir=SESSION_ROOT)


def _resolve_file_path(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and "name" in item:
        return item["name"]
    if hasattr(item, "name"):
        return item.name
    return str(item)


def _resolve_image_paths(uploaded_images) -> list[str]:
    if uploaded_images is None:
        uploaded_images = []
    if not isinstance(uploaded_images, (list, tuple)):
        uploaded_images = [uploaded_images]
    image_paths = [_resolve_file_path(item) for item in uploaded_images]
    image_paths = [os.path.abspath(os.path.expanduser(p)) for p in image_paths if p]
    if not image_paths:
        raise gr.Error("Upload one or more images, or provide a video.")
    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Image file not found: {missing[0]}")
    return image_paths


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("output", {})
    return cfg


def _demo_loop_defaults() -> dict:
    cfg = _load_config(default_config_path())
    loop_cfg = deepcopy(cfg.get("online_loop", {}) or {})
    groups = cfg.get("online_loop_groups", {}) or {}
    if DEFAULT_LOOP_PRESET in groups:
        loop_cfg.update(groups[DEFAULT_LOOP_PRESET] or {})
    return loop_cfg


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def _find_sequence_dir(output_root: str) -> str:
    candidates = []
    for path in Path(output_root).iterdir():
        if path.is_dir() and (path / "poses" / "abs_pose.txt").is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No HorizonStream sequence output found under {output_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _run_loop_closure(
    cfg: dict,
    output_root: str,
    salad_ckpt_path: str,
    salad_dino_weights_path: str,
    salad_score_thresh: float,
    retrieval_top_k: int,
    temporal_exclusion: int,
    min_frame_separation: int,
    loop_edge_score_threshold: float,
    pose_graph_loop_weight: float,
    pose_graph_trans_weight: float,
    pose_graph_rot_weight: float,
    download_loop_assets: bool,
    preflight_only: bool = False,
) -> bool:
    loop_cfg = deepcopy(cfg)
    loop_cfg.setdefault("online_loop", {})
    groups = loop_cfg.get("online_loop_groups", {}) or {}
    if DEFAULT_LOOP_PRESET in groups:
        loop_cfg["online_loop"].update(groups[DEFAULT_LOOP_PRESET] or {})
    loop_cfg["online_loop"]["methods"] = ["salad"]
    loop_cfg["online_loop"]["reuse_online_poses"] = True
    loop_cfg["online_loop"]["online_output_root"] = output_root
    loop_cfg["online_loop"]["rebuild_pointcloud"] = True
    loop_cfg["online_loop"]["auto_output_suffix"] = False
    loop_cfg.setdefault("output", {})["root"] = output_root
    if salad_ckpt_path and salad_ckpt_path.strip():
        loop_cfg["online_loop"]["salad_ckpt_path"] = salad_ckpt_path.strip()
    if salad_dino_weights_path and salad_dino_weights_path.strip():
        loop_cfg["online_loop"]["salad_dino_weights_path"] = salad_dino_weights_path.strip()
    loop_overrides = {
        "salad_score_thresh": salad_score_thresh,
        "retrieval_top_k": retrieval_top_k,
        "temporal_exclusion": temporal_exclusion,
        "min_frame_separation": min_frame_separation,
        "loop_edge_score_threshold": loop_edge_score_threshold,
        "pose_graph_loop_weight": pose_graph_loop_weight,
        "pose_graph_trans_weight": pose_graph_trans_weight,
        "pose_graph_rot_weight": pose_graph_rot_weight,
    }
    for key, value in loop_overrides.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and float(value) < 0:
            continue
        loop_cfg["online_loop"][key] = int(value) if key in {"retrieval_top_k", "temporal_exclusion", "min_frame_separation"} else float(value)

    ok, messages = ensure_loop_assets(
        loop_cfg["online_loop"],
        auto_download=bool(download_loop_assets),
    )
    if not ok:
        print_asset_warnings(messages)
        return False
    if preflight_only:
        return True

    from horizonstream.loop.online_loop_reinfer import run_online_loop_reinfer

    run_online_loop_reinfer(loop_cfg)
    return True

def _iframe_html(host: str, port: int, height: int) -> str:
    host = (host or "127.0.0.1").strip()
    if host.startswith("http://") or host.startswith("https://"):
        url = f"{host.rstrip('/')}:{port}"
    else:
        url = f"http://{host}:{port}"
    return (
        f'<iframe src="{url}" '
        'style="width:100%; '
        f'height:{int(height)}px; '
        'border:1px solid #d0d7de; border-radius:8px;" '
        'allow="clipboard-read; clipboard-write"></iframe>'
    )


def _run_video(
    uploaded_video,
    uploaded_images,
    local_video_path: str,
    fps: Optional[float],
    video_stride: int,
    max_frames: Optional[int],
    run_loop_closure: bool,
    salad_score_thresh: float,
    retrieval_top_k: int,
    temporal_exclusion: int,
    min_frame_separation: int,
    loop_edge_score_threshold: float,
    pose_graph_loop_weight: float,
    pose_graph_trans_weight: float,
    pose_graph_rot_weight: float,
    public_host: str,
    iframe_height: int,
):
    has_images = bool(uploaded_images)
    video_path = _resolve_file_path(uploaded_video) or (local_video_path or "").strip()
    if not has_images and not video_path:
        raise gr.Error("Upload images, upload a video, or provide a local video path.")

    session_dir = _new_session_dir()
    output_root = os.path.join(session_dir, "outputs")
    os.makedirs(output_root, exist_ok=True)

    cfg = _load_config(default_config_path())
    cfg["device"] = DEFAULT_DEVICE
    cfg["model"]["checkpoint"] = DEFAULT_CHECKPOINT
    if has_images:
        image_paths = _resolve_image_paths(uploaded_images)
        cfg["data"]["format"] = "image_list"
        cfg["data"]["image_paths"] = image_paths
        cfg["data"]["image_scene_name"] = "images"
        cfg["data"]["img_path"] = None
        cfg["data"]["seq_list"] = None
        cfg["data"]["camera"] = None
        cfg["data"]["video_path"] = None
        cfg["data"]["video_sample_fps"] = None
        cfg["data"]["video_scene_name"] = None
        print(
            f"[demo] image upload: frames={len(image_paths)} direct_paths=1",
            flush=True,
        )
    else:
        video_path = os.path.abspath(os.path.expanduser(video_path))
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cfg["data"]["video_path"] = video_path
        cfg["data"]["video_stride"] = max(1, int(video_stride))
        cfg["data"]["video_scene_name"] = Path(video_path).stem
        if fps is not None and float(fps) > 0:
            cfg["data"]["video_sample_fps"] = float(fps)
        else:
            cfg["data"]["video_sample_fps"] = None
    if max_frames is not None and int(max_frames) > 0:
        cfg["data"]["max_frames"] = int(max_frames)
    else:
        cfg["data"]["max_frames"] = None
    cfg["inference"]["offload_outputs_to_cpu"] = False
    cfg["output"]["root"] = output_root
    cfg["output"]["save_points"] = True
    cfg["output"]["save_frame_points"] = False
    cfg["output"]["save_videos"] = False
    cfg["output"]["save_depth"] = True
    cfg["output"]["save_depth_conf"] = False
    cfg["output"]["save_depth_vis"] = False
    cfg["output"]["save_images"] = True
    cfg["output"]["save_plots"] = False
    cfg["output"]["run_eval"] = False
    cfg["output"]["max_full_pointcloud_points"] = DEFAULT_MAX_FULL_POINTS
    from infer import prepare_video_input_if_needed
    from horizonstream.core.infer import run_inference_cfg

    cfg = prepare_video_input_if_needed(cfg)

    loop_ready = False
    if bool(run_loop_closure):
        # Preflight before spending minutes on inference.
        loop_ready = _run_loop_closure(
            cfg,
            output_root=output_root,
            salad_ckpt_path="",
            salad_dino_weights_path="",
            salad_score_thresh=salad_score_thresh,
            retrieval_top_k=retrieval_top_k,
            temporal_exclusion=temporal_exclusion,
            min_frame_separation=min_frame_separation,
            loop_edge_score_threshold=loop_edge_score_threshold,
            pose_graph_loop_weight=pose_graph_loop_weight,
            pose_graph_trans_weight=pose_graph_trans_weight,
            pose_graph_rot_weight=pose_graph_rot_weight,
            download_loop_assets=True,
            preflight_only=True,
        )

    run_inference_cfg(cfg)
    sequence_dir = _find_sequence_dir(output_root)
    pose_file = "abs_pose.txt"

    if loop_ready:
        _run_loop_closure(
            cfg,
            output_root=output_root,
            salad_ckpt_path="",
            salad_dino_weights_path="",
            salad_score_thresh=salad_score_thresh,
            retrieval_top_k=retrieval_top_k,
            temporal_exclusion=temporal_exclusion,
            min_frame_separation=min_frame_separation,
            loop_edge_score_threshold=loop_edge_score_threshold,
            pose_graph_loop_weight=pose_graph_loop_weight,
            pose_graph_trans_weight=pose_graph_trans_weight,
            pose_graph_rot_weight=pose_graph_rot_weight,
            download_loop_assets=False,
        )
        if os.path.isfile(os.path.join(sequence_dir, "poses", "loop_abs_pose.txt")):
            pose_file = "loop_abs_pose.txt"

    port = _find_free_port()
    share_url = _visualize_sequence_dir_simple(
        sequence_dir=sequence_dir,
        pose_file=pose_file,
        gt_pose_file=None,
        pose_convention="auto",
        pose_layout="r9t3",
        points_dir=None,
        poses_dir=None,
        rgb_dir=None,
        intri_file=None,
        port=port,
        share=False,
        background_mode=True,
        point_size=0.01,
        ply_stride=1,
        video_width=320,
        cleanup_paths=[session_dir],
    )

    viewer_url = share_url or f"http://{(public_host or '127.0.0.1').strip()}:{port}"
    iframe_html = (
        f'<iframe src="{viewer_url}" '
        'style="width:100%; '
        f'height:{int(iframe_height)}px; '
        'border:1px solid #d0d7de; border-radius:8px;" '
        'allow="clipboard-read; clipboard-write"></iframe>'
    )

    summary_bits = [f"Pose file loaded: `{pose_file}`"]
    loop_summary_path = os.path.join(sequence_dir, "summary.json")
    if os.path.isfile(loop_summary_path):
        try:
            import json

            with open(loop_summary_path, "r") as f:
                summary = json.load(f)
            summary_bits.append(
                "Loop: "
                f"candidates={summary.get('num_candidates', 0)}, "
                f"refined={summary.get('num_refined_candidates', 0)}, "
                f"edges={summary.get('num_loop_edges', 0)}"
            )
        except Exception:
            pass
    return iframe_html, "\n\n".join(summary_bits)


def main() -> None:
    loop_defaults = _demo_loop_defaults()
    default_salad_score_thresh = float(loop_defaults.get("salad_score_thresh", 0.85))
    default_retrieval_top_k = int(loop_defaults.get("retrieval_top_k", 3))
    default_temporal_exclusion = int(loop_defaults.get("temporal_exclusion", 10))
    default_min_frame_separation = int(loop_defaults.get("min_frame_separation", 30))
    default_loop_edge_score_threshold = float(loop_defaults.get("loop_edge_score_threshold", 0.025))
    default_pose_graph_loop_weight = float(loop_defaults.get("pose_graph_loop_weight", 0.01))

    with gr.Blocks(title="HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction") as demo:
        gr.Markdown("# HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction")
        gr.Markdown("Upload a video or multiple images, run HorizonStream, then inspect the output in the embedded Viser viewer.")

        with gr.Row():
            uploaded_video = gr.File(label="Upload Video", file_count="single", file_types=["video"])
            uploaded_images = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])
            local_video_path = gr.Textbox(label="Local Video Path", placeholder="/path/to/video.mov")

        with gr.Accordion("Inference", open=True):
            with gr.Row():
                fps = gr.Number(label="Target FPS (0 = source/stride)", value=0)
                video_stride = gr.Slider(label="Video Stride", minimum=1, maximum=30, step=1, value=1)
                max_frames = gr.Number(label="Max Frames (0 = all)", value=0)

        with gr.Accordion("Loop Closure", open=False):
            run_loop_closure = gr.Checkbox(label="Run Loop Closure", value=True)
            with gr.Row():
                salad_score_thresh = gr.Slider(
                    label="Loop Score Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=default_salad_score_thresh,
                    info="If no loop is detected, lower this threshold. If extra false loops are detected, raise it.",
                )
                retrieval_top_k = gr.Slider(label="Retrieval Top-K", minimum=1, maximum=20, step=1, value=default_retrieval_top_k)
                temporal_exclusion = gr.Number(label="Temporal Exclusion", value=default_temporal_exclusion)
                min_frame_separation = gr.Number(label="Min Frame Separation", value=default_min_frame_separation)
            with gr.Row():
                loop_edge_score_threshold = gr.Number(
                    label="Loop Edge Score Threshold",
                    value=default_loop_edge_score_threshold,
                )
                pose_graph_loop_weight = gr.Number(
                    label="Loop Weight",
                    value=default_pose_graph_loop_weight,
                    info="If a loop is detected but the trajectory does not close enough, increase this weight.",
                )
                pose_graph_trans_weight = gr.Number(label="Translation Weight", value=1.0)
                pose_graph_rot_weight = gr.Number(label="Rotation Weight", value=1.2)

        with gr.Accordion("Viewer", open=False):
            with gr.Row():
                public_host = gr.Textbox(label="Public Host", value="127.0.0.1")
                iframe_height = gr.Slider(label="Iframe Height", minimum=500, maximum=1200, step=50, value=800)

        run_btn = gr.Button("Run Demo", variant="primary")
        status = gr.Markdown()
        viser_frame = gr.HTML()

        run_btn.click(
            _run_video,
            inputs=[
                uploaded_video,
                uploaded_images,
                local_video_path,
                fps,
                video_stride,
                max_frames,
                run_loop_closure,
                salad_score_thresh,
                retrieval_top_k,
                temporal_exclusion,
                min_frame_separation,
                loop_edge_score_threshold,
                pose_graph_loop_weight,
                pose_graph_trans_weight,
                pose_graph_rot_weight,
                public_host,
                iframe_height,
            ],
            outputs=[viser_frame, status],
        )

    launch_kwargs = {"server_name": os.getenv("HORIZONSTREAM_GRADIO_HOST", "0.0.0.0")}
    if os.getenv("HORIZONSTREAM_GRADIO_PORT"):
        launch_kwargs["server_port"] = int(os.environ["HORIZONSTREAM_GRADIO_PORT"])
    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
