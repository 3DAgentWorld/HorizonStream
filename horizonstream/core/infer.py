from __future__ import annotations

import os
import time
import warnings
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from horizonstream.core.model import HorizonStreamModel
from horizonstream.data.dataloader import HorizonStreamDataLoader
from horizonstream.io.save_images import save_image_sequence, save_video
from horizonstream.io.save_points import prepare_pointcloud, save_pointcloud
from horizonstream.io.save_poses_txt import save_intri_txt, save_w2c_txt
from horizonstream.io.save_trajectory_plot import _load_gt_w2c_from_sequence, save_trajectory_comparison_plot
from horizonstream.eval.io import frame_stems
from horizonstream.utils.depth import colorize_depth, unproject_depth_to_points
from horizonstream.utils.sky_mask import compute_sky_mask
from horizonstream.utils.vendor.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from horizonstream.runtime.motion_averaging import affine_padding, compute_motion_averaged_camera_maps


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
    module=r"torch\.utils\.checkpoint",
)


def _to_uint8_rgb(images):
    imgs = images.detach().cpu().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255.0).astype(np.uint8)
    return imgs


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sync_device_for_timing(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _apply_sky_mask(depth, mask):
    if mask is None:
        return depth
    m = (mask > 0).astype(np.float32)
    return depth * m


def _parse_depth_bounds(output_cfg):
    depth_min = output_cfg.get("point_depth_min", None)
    if depth_min in (None, "", "null"):
        depth_min = None
    else:
        depth_min = float(depth_min)
    depth_max = output_cfg.get("point_depth_max", None)
    if depth_max in (None, "", "null"):
        depth_max = None
    else:
        depth_max = float(depth_max)
    if depth_min is not None and depth_max is not None and depth_max <= depth_min:
        depth_max = None
    return depth_min, depth_max


def _pointcloud_filter_kwargs(output_cfg):
    return {
        "sky_color_filter": bool(output_cfg.get("point_sky_color_filter", True)),
        "sky_white_threshold": int(output_cfg.get("point_sky_white_threshold", 240)),
        "sky_bright_any_threshold": int(output_cfg.get("point_sky_bright_any_threshold", 200)),
        "sky_blue_b_min": int(output_cfg.get("point_sky_blue_b_min", 155)),
        "sky_blue_r_max": int(output_cfg.get("point_sky_blue_r_max", 195)),
        "sky_blue_g_min": int(output_cfg.get("point_sky_blue_g_min", 100)),
        "sky_blue_rg_diff_max": int(output_cfg.get("point_sky_blue_rg_diff_max", 95)),
        "sky_blue_bg_diff_min": int(output_cfg.get("point_sky_blue_bg_diff_min", 8)),
        "sky_blue_br_diff_min": int(output_cfg.get("point_sky_blue_br_diff_min", 15)),
        "outlier_filter": bool(output_cfg.get("point_outlier_filter", True)),
        "radius_outlier_radius": float(output_cfg.get("point_radius_outlier_radius", 0.10)),
        "radius_outlier_min_points": int(output_cfg.get("point_radius_outlier_min_points", 16)),
    }


def _build_point_valid_mask(depth, sky_mask, depth_min, depth_max):
    valid = np.isfinite(depth)
    if depth_min is not None:
        valid &= depth > float(depth_min)
    if depth_max is not None:
        valid &= depth < float(depth_max)
    if sky_mask is not None:
        valid &= np.asarray(sky_mask) > 0
    return valid


def _apply_depth_percentile_bounds(depth, valid_mask, pmin, pmax):
    if pmin is None and pmax is None:
        return valid_mask
    values = depth[valid_mask]
    if values.size == 0:
        return valid_mask
    lo = None if pmin is None else float(np.percentile(values, float(pmin)))
    hi = None if pmax is None else float(np.percentile(values, float(pmax)))
    valid = valid_mask.copy()
    if lo is not None:
        valid &= depth >= lo
    if hi is not None:
        valid &= depth <= hi
    return valid


def _camera_points_to_world(points, extri):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(extri[:3, :3], dtype=np.float64)
    t = np.asarray(extri[:3, 3], dtype=np.float64)
    world = (R.T @ (pts.T - t[:, None])).T
    return world.astype(np.float32, copy=False)


def _mask_points_and_colors(points, colors, mask):
    pts = points.reshape(-1, 3)
    cols = None if colors is None else colors.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if mask is not None:
        valid &= mask.reshape(-1) > 0
    pts = pts[valid]
    if cols is not None:
        cols = cols[valid]
    return pts, cols


def _mask_valid_ratio(mask):
    if mask is None:
        return None
    valid = np.asarray(mask).reshape(-1) > 0
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def _resize_long_edge(arr, long_edge_size, interpolation):
    h, w = arr.shape[:2]
    scale = float(long_edge_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _prepare_mask_for_model(mask, size, crop, patch_size, target_shape, square_ok=False):
    if mask is None:
        return None
    long_edge = (
        round(size * max(mask.shape[1] / mask.shape[0], mask.shape[0] / mask.shape[1]))
        if size == 224
        else size
    )
    mask = _resize_long_edge(mask, long_edge, cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        target_w = 2 * half
        target_h = 2 * half
        if crop:
            mask = mask[cy - half : cy + half, cx - half : cx + half]
        else:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        target_w = 2 * halfw
        target_h = 2 * halfh
        if crop:
            mask = mask[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _save_full_pointcloud(
    path,
    point_chunks,
    color_chunks,
    max_points=None,
    voxel_size=None,
    random_sample_ratio=None,
    filter_kwargs=None,
    seed=0,
):
    point_chunks = [p for p in point_chunks if p is not None and len(p) > 0]
    if not point_chunks:
        return
    points = np.concatenate(point_chunks, axis=0)
    raw_count = int(points.shape[0])
    colors = None
    if color_chunks and len(color_chunks) == len(point_chunks):
        colors = np.concatenate(color_chunks, axis=0)
    points, colors = prepare_pointcloud(
        points,
        colors=colors,
        voxel_size=voxel_size,
        random_sample_ratio=random_sample_ratio,
        max_points=max_points,
        **(filter_kwargs or {}),
        seed=seed,
    )
    np.save(os.path.splitext(path)[0] + ".npy", points.astype(np.float32, copy=False))
    save_pointcloud(path, points, colors=colors, max_points=None, seed=seed)


def _pad_extri(extri: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(4, dtype=extri.dtype, device=extri.device).unsqueeze(0).expand(extri.shape[0], -1, -1).clone()
    eye[:, :3, :4] = extri
    return eye


def _chunk_schedule(num_frames: int, window_size: int, sliding_size: int) -> List[Tuple[int, int]]:
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [(0, num_frames)]
    chunks = [(0, window_size)]
    start = window_size
    while start < num_frames:
        end = min(start + sliding_size, num_frames)
        chunks.append((start, end))
        start = end
    return chunks


def _append_chunk_depths(
    outputs: dict,
    global_depth: List[torch.Tensor],
    global_depth_conf: List[torch.Tensor],
):
    for idx in range(outputs["depth"].shape[1]):
        global_depth.append(outputs["depth"][0, idx].detach().cpu())
        global_depth_conf.append(outputs["depth_conf"][0, idx].detach().cpu())


def _save_pose_variant(
    pose_dir: str,
    prefix: str,
    cam_map: torch.Tensor,
    image_size_hw: Tuple[int, int],
    frame_ids: List[int],
    save_intri: bool = True,
):
    extri, intri = pose_encoding_to_extri_intri(cam_map, image_size_hw=image_size_hw)
    extri_np = extri[0].detach().cpu().numpy()
    intri_np = intri[0].detach().cpu().numpy()
    pose_name = "abs_pose.txt" if prefix == "abs" else f"{prefix}_abs_pose.txt"
    save_w2c_txt(os.path.join(pose_dir, pose_name), extri_np, frame_ids)
    if save_intri:
        save_intri_txt(os.path.join(pose_dir, f"{prefix}_intri.txt"), intri_np, frame_ids)
    return extri_np, intri_np


def run_inference_cfg(cfg: dict):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    output_cfg = cfg.get("output", {})

    window_size = int(infer_cfg.get("window_size", 10))
    sliding_size = int(infer_cfg.get("sliding_size", window_size))
    offload_outputs_to_cpu = bool(infer_cfg.get("offload_outputs_to_cpu", False))
    enable_offline_motion_averaging = bool(infer_cfg.get("enable_offline_motion_averaging", False))
    abs_pose_source = str(output_cfg.get("abs_pose_source", "online")).strip().lower()
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if sliding_size <= 0:
        raise ValueError("sliding_size must be positive.")
    if abs_pose_source == "offline" and not enable_offline_motion_averaging:
        raise ValueError(
            "output.abs_pose_source='offline' requires "
            "inference.enable_offline_motion_averaging=true or --offline-motion-averaging."
        )

    print(f"[horizonstream] device={device}", flush=True)
    model = HorizonStreamModel(model_cfg).to(device)
    model.eval()
    print("[horizonstream] model ready", flush=True)

    loader = HorizonStreamDataLoader(data_cfg)
    print(
        f"[horizonstream] runtime artifacts: offload_outputs_to_cpu={offload_outputs_to_cpu}",
        flush=True,
    )
    out_root = output_cfg.get("root", "outputs_horizonstream")
    _ensure_dir(out_root)
    save_videos = bool(output_cfg.get("save_videos", True))
    save_points = bool(output_cfg.get("save_points", True))
    save_frame_points = bool(output_cfg.get("save_frame_points", False))
    save_depth = bool(output_cfg.get("save_depth", True))
    save_depth_conf = bool(output_cfg.get("save_depth_conf", True))
    save_images = bool(output_cfg.get("save_images", True))
    save_plots = bool(output_cfg.get("save_plots", True))
    video_fps = float(output_cfg.get("video_fps", 30.0))
    if video_fps <= 0:
        video_fps = 30.0
    mask_sky = bool(output_cfg.get("mask_sky", False))
    point_mask_sky = bool(output_cfg.get("point_mask_sky", False))
    max_full_pointcloud_points = output_cfg.get("max_full_pointcloud_points", None)
    max_full_pointcloud_points = (
        None
        if max_full_pointcloud_points in (None, "", "null")
        else int(max_full_pointcloud_points)
    )
    max_frame_pointcloud_points = output_cfg.get("max_frame_pointcloud_points", None)
    max_frame_pointcloud_points = (
        None
        if max_frame_pointcloud_points in (None, "", "null")
        else int(max_frame_pointcloud_points)
    )
    point_voxel_size = output_cfg.get("point_voxel_size", None)
    point_voxel_size = (
        None if point_voxel_size in (None, "", "null") else float(point_voxel_size)
    )
    point_random_sample_ratio = output_cfg.get("point_random_sample_ratio", None)
    point_random_sample_ratio = (
        None
        if point_random_sample_ratio in (None, "", "null")
        else float(point_random_sample_ratio)
    )
    pointcloud_filter_kwargs = _pointcloud_filter_kwargs(output_cfg)
    point_depth_min, point_depth_max = _parse_depth_bounds(output_cfg)
    point_depth_percentile_min = output_cfg.get("point_depth_percentile_min", None)
    point_depth_percentile_max = output_cfg.get("point_depth_percentile_max", None)
    point_depth_percentile_min = (
        None
        if point_depth_percentile_min in (None, "", "null")
        else float(point_depth_percentile_min)
    )
    point_depth_percentile_max = (
        None
        if point_depth_percentile_max in (None, "", "null")
        else float(point_depth_percentile_max)
    )
    skyseg_path = output_cfg.get("skyseg_path", "checkpoints/skyseg.onnx")
    print(
        f"[horizonstream] sky mask config: depth_enabled={mask_sky} "
        f"point_enabled={point_mask_sky} skyseg_path={skyseg_path}",
        flush=True,
    )
    total_forward_time = 0.0
    total_infer_frames = 0

    with torch.no_grad():
        for seq in loader:
            images = seq.images
            if images.device.type != "cpu":
                images = images.cpu()
            B, S, _, H, W = images.shape
            if B != 1:
                raise ValueError("HorizonStream inference currently expects batch size 1.")
            print(f"[horizonstream] sequence {seq.name}: inference start ({S} frames)", flush=True)

            chunks = _chunk_schedule(S, window_size, sliding_size)
            global_depth = []
            global_depth_conf = []
            chunk_cam_maps = []
            state = model.build_sequence_state()
            forward_time = 0.0

            for chunk_idx, (start, end) in enumerate(chunks):
                chunk_images = images[:, start:end].to(device, non_blocking=True)

                current_window_size = end - start if chunk_idx == 0 else window_size
                _sync_device_for_timing(device)
                forward_t0 = time.perf_counter()
                outputs = model.forward_chunk(
                    chunk_images,
                    window_size=current_window_size,
                    chunk_idx=chunk_idx,
                    state=state,
                )
                _sync_device_for_timing(device)
                forward_time += time.perf_counter() - forward_t0

                chunk_cam_map = outputs["chunk_cam_map"].detach().to(dtype=torch.float32)
                if offload_outputs_to_cpu:
                    chunk_cam_map = chunk_cam_map.cpu()
                chunk_cam_maps.append(chunk_cam_map)
                _append_chunk_depths(outputs, global_depth, global_depth_conf)
                model.advance_sequence_state(state, is_last_chunk=chunk_idx == len(chunks) - 1)
                del outputs, chunk_images
                if offload_outputs_to_cpu and isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if len(global_depth) != S:
                raise ValueError(f"Expected {S} depth frames, got {len(global_depth)}")

            motion_maps = compute_motion_averaged_camera_maps(
                chunk_cam_maps,
                frames_num=S,
                window_size=window_size,
                dtype=torch.float32,
                enable_offline=enable_offline_motion_averaging,
            )
            total_forward_time += forward_time
            total_infer_frames += int(S)
            online_cam_map = motion_maps["online_cam_map"]
            offline_cam_map = motion_maps.get("offline_cam_map", None)
            last_cam_map = motion_maps.get("last_cam_map", None)
            lag_cam_map = motion_maps.get("lag_cam_map", None)
            if abs_pose_source not in {"online", "offline"}:
                raise ValueError(f"output.abs_pose_source must be 'online' or 'offline', got: {abs_pose_source}")
            if abs_pose_source == "offline" and offline_cam_map is not None:
                main_cam_map = offline_cam_map
            else:
                main_cam_map = online_cam_map

            print(f"[horizonstream] sequence {seq.name}: inference done", flush=True)

            seq_dir = os.path.join(out_root, seq.name)
            _ensure_dir(seq_dir)
            frame_ids = list(range(S))
            rgb = _to_uint8_rgb(images[0].permute(0, 2, 3, 1))

            pose_dir = os.path.join(seq_dir, "poses")
            _ensure_dir(pose_dir)
            online_extri_np, online_intri_np = _save_pose_variant(
                pose_dir,
                "abs",
                online_cam_map,
                (H, W),
                frame_ids,
                save_intri=False,
            )
            offline_extri_np = None
            offline_intri_np = None
            if offline_cam_map is not None:
                offline_extri_np, offline_intri_np = _save_pose_variant(
                    pose_dir,
                    "offline",
                    offline_cam_map,
                    (H, W),
                    frame_ids,
                    save_intri=False,
                )
            main_extri_np, main_intri_np = online_extri_np, online_intri_np

            save_intri_txt(os.path.join(pose_dir, "intri.txt"), main_intri_np, frame_ids)
            if offline_error:
                with open(os.path.join(pose_dir, "offline_error.txt"), "w") as f:
                    f.write(str(offline_error).strip() + "\n")

            gt_w2c = _load_gt_w2c_from_sequence(seq)
            if gt_w2c is not None:
                gt_stems = frame_stems(getattr(seq, "image_paths", []))
                gt_frame_ids = gt_stems if len(gt_stems) == len(gt_w2c) else list(range(len(gt_w2c)))
                save_w2c_txt(os.path.join(pose_dir, "gt_abs_pose.txt"), gt_w2c, gt_frame_ids)

            plot_trajectories = {
                "online": affine_padding(torch.from_numpy(online_extri_np)).numpy(),
            }
            if offline_extri_np is not None:
                plot_trajectories["offline"] = affine_padding(torch.from_numpy(offline_extri_np)).numpy()
            if save_plots:
                save_trajectory_comparison_plot(
                    os.path.join(seq_dir, "plots", "trajectory_compare.png"),
                    plot_trajectories,
                    seq=seq,
                    offline_error=offline_error,
                )

            if save_images:
                print(f"[horizonstream] sequence {seq.name}: saving rgb", flush=True)
                rgb_dir = os.path.join(seq_dir, "images", "rgb")
                save_image_sequence(rgb_dir, list(rgb[: len(frame_ids)]))
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "images", "rgb.mp4"),
                        os.path.join(rgb_dir, "frame_*.png"),
                        fps=video_fps,
                    )

            sky_masks = None
            if mask_sky or point_mask_sky:
                sky_t0 = time.perf_counter()
                print(
                    f"[horizonstream] sequence {seq.name}: computing sky mask",
                    flush=True,
                )
                raw_sky_masks = compute_sky_mask(
                    (
                        seq.image_paths[: len(frame_ids)]
                        if len(seq.image_paths) >= len(frame_ids)
                        and all(os.path.exists(p) for p in seq.image_paths[: len(frame_ids)])
                        else list(rgb[: len(frame_ids)])
                    ),
                    skyseg_path,
                    os.path.join(seq_dir, "sky_masks"),
                )
                print(
                    f"[horizonstream] sequence {seq.name}: sky mask done in "
                    f"{time.perf_counter() - sky_t0:.2f}s",
                    flush=True,
                )
                if raw_sky_masks is not None:
                    sky_masks = [
                        _prepare_mask_for_model(
                            mask,
                            size=int(data_cfg.get("size", 518)),
                            crop=bool(data_cfg.get("crop", False)),
                            patch_size=int(data_cfg.get("patch_size", 14)),
                            target_shape=(H, W),
                        )
                        for mask in raw_sky_masks
                    ]
                else:
                    print(
                        f"[horizonstream] sequence {seq.name}: sky mask not applied "
                        f"(compute_sky_mask returned None, skyseg_path={skyseg_path})",
                        flush=True,
                    )
            else:
                print(
                    f"[horizonstream] sequence {seq.name}: sky mask disabled by config",
                    flush=True,
                )

            point_chunks = []
            color_chunks = []
            if save_depth:
                print(f"[horizonstream] sequence {seq.name}: saving depth", flush=True)
                depth_dir = os.path.join(seq_dir, "depth", "dpt")
                conf_dir = os.path.join(seq_dir, "depth", "conf")
                save_depth_vis = bool(output_cfg.get("save_depth_vis", True))
                color_dir = os.path.join(seq_dir, "depth", "dpt_plasma")
                _ensure_dir(depth_dir)
                if save_depth_conf:
                    _ensure_dir(conf_dir)
                if save_depth_vis:
                    _ensure_dir(color_dir)

                color_frames = []
                for i, depth_tensor in enumerate(global_depth):
                    d = depth_tensor[:, :, 0].numpy()
                    sky_mask = None if sky_masks is None else sky_masks[i]
                    if mask_sky and sky_mask is not None:
                        d = _apply_sky_mask(d, sky_mask)
                    np.save(os.path.join(depth_dir, f"frame_{i:06d}.npy"), d)
                    if save_depth_conf:
                        depth_conf = global_depth_conf[i].numpy()
                        np.save(os.path.join(conf_dir, f"frame_{i:06d}.npy"), depth_conf)
                    if save_depth_vis:
                        colored = colorize_depth(d, cmap="plasma")
                        Image.fromarray(colored).save(os.path.join(color_dir, f"frame_{i:06d}.png"))
                        color_frames.append(colored)

                    if save_points or save_frame_points:
                        point_sky_mask = sky_mask if point_mask_sky else None
                        valid_mask = _build_point_valid_mask(
                            d,
                            sky_mask=point_sky_mask,
                            depth_min=point_depth_min,
                            depth_max=point_depth_max,
                        )
                        valid_mask = _apply_depth_percentile_bounds(
                            d,
                            valid_mask,
                            point_depth_percentile_min,
                            point_depth_percentile_max,
                        )
                        if not np.any(valid_mask):
                            mask_ratio = _mask_valid_ratio(sky_mask)
                            image_path = seq.image_paths[i] if i < len(seq.image_paths) else ""
                            print(
                                f"[horizonstream] warning: sequence {seq.name} frame {i:06d} "
                                f"has no valid points after depth/sky filtering "
                                f"(depth_min={point_depth_min}, depth_max={point_depth_max}, "
                                f"mask_valid_ratio={mask_ratio}, image={image_path})",
                                flush=True,
                            )
                            continue
                        depth_for_unproj = np.where(valid_mask, d, 0.0).astype(np.float32)
                        depth_b = torch.from_numpy(depth_for_unproj[None]).float()
                        intri_b = torch.from_numpy(main_intri_np[i : i + 1]).float()
                        pts_cam = unproject_depth_to_points(depth_b, intri_b)[0].numpy()
                        cols = rgb[i][: pts_cam.shape[0], : pts_cam.shape[1]]
                        pts, cols = _mask_points_and_colors(pts_cam, cols, valid_mask)
                        pts_world = _camera_points_to_world(pts, main_extri_np[i][:3])
                        if pts_world.shape[0] == 0:
                            mask_ratio = _mask_valid_ratio(sky_mask)
                            image_path = seq.image_paths[i] if i < len(seq.image_paths) else ""
                            print(
                                f"[horizonstream] warning: sequence {seq.name} frame {i:06d} "
                                f"has no valid world points after filtering "
                                f"(depth_min={point_depth_min}, depth_max={point_depth_max}, "
                                f"mask_valid_ratio={mask_ratio}, image={image_path})",
                                flush=True,
                            )
                        elif save_points:
                            point_chunks.append(pts_world)
                            color_chunks.append(cols)
                        if save_frame_points:
                            frame_pc_dir = os.path.join(seq_dir, "points", "frames")
                            _ensure_dir(frame_pc_dir)
                            save_pointcloud(
                                os.path.join(frame_pc_dir, f"frame_{i:06d}.ply"),
                                pts_world,
                                colors=cols,
                                max_points=max_frame_pointcloud_points,
                                voxel_size=point_voxel_size,
                                random_sample_ratio=point_random_sample_ratio,
                                **pointcloud_filter_kwargs,
                                seed=i,
                            )

                if save_videos and save_depth_vis:
                    save_video(
                        os.path.join(seq_dir, "depth", "dpt_plasma.mp4"),
                        os.path.join(color_dir, "frame_*.png"),
                        fps=video_fps,
                    )

            if save_points and point_chunks:
                print(f"[horizonstream] sequence {seq.name}: saving pointcloud", flush=True)
                pc_dir = os.path.join(seq_dir, "points")
                _ensure_dir(pc_dir)
                _save_full_pointcloud(
                    os.path.join(pc_dir, "full.ply"),
                    point_chunks,
                    color_chunks,
                    max_points=max_full_pointcloud_points,
                    voxel_size=point_voxel_size,
                    random_sample_ratio=point_random_sample_ratio,
                    filter_kwargs=pointcloud_filter_kwargs,
                )
