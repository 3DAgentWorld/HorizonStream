#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from horizonstream.io.save_points import prepare_pointcloud, save_pointcloud


def read_w2c_txt(path: Path) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"Expected 13 values in {path}, got {len(vals)}")
            frame = int(vals[0])
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = np.asarray(vals[1:10], dtype=np.float64).reshape(3, 3)
            mat[:3, 3] = np.asarray(vals[10:13], dtype=np.float64)
            poses[frame] = mat
    return poses


def read_intri_txt(path: Path) -> Dict[int, np.ndarray]:
    intri: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 5:
                raise ValueError(f"Expected 5 values in {path}, got {len(vals)}")
            frame = int(vals[0])
            fx, fy, cx, cy = vals[1:]
            k = np.eye(3, dtype=np.float64)
            k[0, 0] = fx
            k[1, 1] = fy
            k[0, 2] = cx
            k[1, 2] = cy
            intri[frame] = k
    return intri


def resolve_pose_path(sequence_dir: Path, pose_variant: str) -> Tuple[str, Path]:
    pose_dir = sequence_dir / "poses"
    variant = str(pose_variant).strip().lower()
    if variant in {"auto", ""}:
        loop_path = pose_dir / "loop_abs_pose.txt"
        if loop_path.exists():
            return "loop", loop_path
        return "online", pose_dir / "abs_pose.txt"
    if variant in {"online", "main", "abs"}:
        return "online", pose_dir / "abs_pose.txt"
    if variant in {"lc", "loop"}:
        return "loop", pose_dir / "loop_abs_pose.txt"
    return variant, pose_dir / f"{variant}_abs_pose.txt"


def frame_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def camera_to_world(points_cam: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    return (r.T @ (points_cam.T - t[:, None])).T


def world_to_camera(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    return (r @ points_world.T + t[:, None]).T


def unproject(uv: np.ndarray, depth: np.ndarray, intri: np.ndarray) -> np.ndarray:
    z = depth.astype(np.float64)
    x = (uv[:, 0] - intri[0, 2]) / intri[0, 0] * z
    y = (uv[:, 1] - intri[1, 2]) / intri[1, 1] * z
    return np.stack([x, y, z], axis=1)


def project(points_cam: np.ndarray, intri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    uv = np.empty((len(points_cam), 2), dtype=np.float64)
    uv[:, 0] = points_cam[:, 0] / z * intri[0, 0] + intri[0, 2]
    uv[:, 1] = points_cam[:, 1] / z * intri[1, 1] + intri[1, 2]
    return uv, z


def load_conf(conf_dir: Path, frame: int, shape: Tuple[int, int]) -> np.ndarray | None:
    path = conf_dir / f"frame_{frame:06d}.npy"
    if not path.exists():
        return None
    conf = np.load(path).astype(np.float32)
    if conf.shape[:2] != shape:
        conf = cv2.resize(conf, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(conf)) if conf.size else 0.0
    if max_val > 1.0:
        conf = conf / max(max_val, 1e-8)
    return np.clip(conf, 0.0, 1.0)


def source_pixels(
    depth: np.ndarray,
    conf: np.ndarray | None,
    stride: int,
    depth_min: float,
    depth_max: float | None,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    z = depth[y, x]
    valid = np.isfinite(z) & (z > depth_min)
    if depth_max is not None:
        valid &= z < depth_max
    if conf is not None:
        valid &= conf[y, x] >= conf_threshold
    uv = np.stack([x[valid], y[valid]], axis=1).astype(np.float64)
    return uv, z[valid].astype(np.float64)


def neighbor_ids(frame: int, all_frames: List[int], window: int, max_views: int) -> List[int]:
    ids = [idx for idx in all_frames if idx != frame and abs(idx - frame) <= window]
    ids.sort(key=lambda idx: (abs(idx - frame), idx))
    return ids[:max_views]


def consistency_mask(
    points_world: np.ndarray,
    src_uv: np.ndarray,
    src_depth: np.ndarray,
    src_w2c: np.ndarray,
    src_intri: np.ndarray,
    dst_depth: np.ndarray,
    dst_w2c: np.ndarray,
    dst_intri: np.ndarray,
    pixel_threshold: float,
    relative_depth_threshold: float,
) -> np.ndarray:
    h, w = dst_depth.shape[:2]
    dst_cam = world_to_camera(points_world, dst_w2c)
    dst_uv, dst_z = project(dst_cam, dst_intri)
    x = np.rint(dst_uv[:, 0]).astype(np.int64)
    y = np.rint(dst_uv[:, 1]).astype(np.int64)
    valid = (dst_z > 1e-8) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(valid):
        return valid

    sampled = np.zeros_like(dst_z)
    sampled[valid] = dst_depth[y[valid], x[valid]]
    valid &= np.isfinite(sampled) & (sampled > 1e-8)
    if not np.any(valid):
        return valid

    valid_idx = np.flatnonzero(valid)
    dst_back_cam = unproject(dst_uv[valid], sampled[valid], dst_intri)
    dst_back_world = camera_to_world(dst_back_cam, dst_w2c)
    src_back_cam = world_to_camera(dst_back_world, src_w2c)
    src_back_uv, src_back_depth = project(src_back_cam, src_intri)

    reproj_err = np.linalg.norm(src_back_uv - src_uv[valid], axis=1)
    rel_depth_err = np.abs(src_back_depth - src_depth[valid]) / np.maximum(src_depth[valid], 1e-8)
    out = np.zeros(len(points_world), dtype=bool)
    out[valid_idx] = (reproj_err <= pixel_threshold) & (rel_depth_err <= relative_depth_threshold)
    return out


def load_rgb(rgb_dir: Path, frame: int, shape: Tuple[int, int]) -> np.ndarray | None:
    path = rgb_dir / f"frame_{frame:06d}.png"
    if not path.exists():
        return None
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape[:2] != shape:
        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb


def fuse_sequence(args: argparse.Namespace) -> None:
    sequence_dir = Path(args.sequence_dir).resolve()
    depth_dir = sequence_dir / "depth" / "dpt"
    conf_dir = sequence_dir / "depth" / "conf"
    rgb_dir = sequence_dir / "images" / "rgb"
    pose_variant, pose_path = resolve_pose_path(sequence_dir, args.pose_variant)
    intri_path = sequence_dir / "poses" / "intri.txt"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose file: {pose_path}")
    if not intri_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intri_path}")

    w2c = read_w2c_txt(pose_path)
    intri = read_intri_txt(intri_path)
    frames = [
        frame_id(path)
        for path in sorted(depth_dir.glob("frame_*.npy"))
        if frame_id(path) in w2c and frame_id(path) in intri
    ]
    if not frames:
        raise RuntimeError(f"No aligned frames found in {sequence_dir}")

    point_chunks: List[np.ndarray] = []
    color_chunks: List[np.ndarray] = []
    for idx, frame in enumerate(frames, start=1):
        depth = np.load(depth_dir / f"frame_{frame:06d}.npy").astype(np.float32)
        conf = load_conf(conf_dir, frame, depth.shape[:2])
        uv, src_z = source_pixels(
            depth,
            conf,
            max(1, int(args.pixel_stride)),
            float(args.depth_min),
            args.depth_max,
            float(args.conf_threshold),
        )
        if len(uv) == 0:
            continue

        src_cam = unproject(uv, src_z, intri[frame])
        points_world = camera_to_world(src_cam, w2c[frame])
        support = np.zeros(len(points_world), dtype=np.int32)
        for dst in neighbor_ids(frame, frames, int(args.source_window), int(args.max_source_views)):
            dst_depth = np.load(depth_dir / f"frame_{dst:06d}.npy").astype(np.float32)
            support += consistency_mask(
                points_world,
                uv,
                src_z,
                w2c[frame],
                intri[frame],
                dst_depth,
                w2c[dst],
                intri[dst],
                float(args.pixel_threshold),
                float(args.relative_depth_threshold),
            ).astype(np.int32)

        keep = support >= int(args.min_consistent)
        if not np.any(keep):
            continue
        point_chunks.append(points_world[keep].astype(np.float32, copy=False))
        rgb = load_rgb(rgb_dir, frame, depth.shape[:2])
        if rgb is not None:
            x = uv[:, 0].astype(np.int64)
            y = uv[:, 1].astype(np.int64)
            color_chunks.append(rgb[y[keep], x[keep]].astype(np.uint8, copy=False))
        if idx % 25 == 0 or idx == len(frames):
            print(f"[post_depthfusion] fused {idx}/{len(frames)} frames", flush=True)

    if not point_chunks:
        raise RuntimeError("No fused points survived consistency filtering.")
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0) if len(color_chunks) == len(point_chunks) else None

    default_name = "fused_lc.ply" if pose_variant in {"loop", "lc"} else "fused.ply"
    output_path = Path(args.output).resolve() if args.output else sequence_dir / "points" / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points, colors = prepare_pointcloud(
        points,
        colors=colors,
        voxel_size=args.voxel_size,
        random_sample_ratio=args.random_sample_ratio,
        max_points=args.max_points,
        sky_color_filter=bool(args.sky_color_filter),
        sky_white_threshold=args.sky_white_threshold,
        sky_bright_any_threshold=args.sky_bright_any_threshold,
        sky_blue_b_min=args.sky_blue_b_min,
        sky_blue_r_max=args.sky_blue_r_max,
        sky_blue_g_min=args.sky_blue_g_min,
        sky_blue_rg_diff_max=args.sky_blue_rg_diff_max,
        sky_blue_bg_diff_min=args.sky_blue_bg_diff_min,
        sky_blue_br_diff_min=args.sky_blue_br_diff_min,
        outlier_filter=bool(args.outlier_filter),
        radius_outlier_radius=args.radius_outlier_radius,
        radius_outlier_min_points=args.radius_outlier_min_points,
    )
    np.save(output_path.with_suffix(".npy"), points.astype(np.float32, copy=False))
    save_pointcloud(str(output_path), points, colors=colors, outlier_filter=False)
    print(
        f"[post_depthfusion] wrote {output_path} ({len(points)} fused points, pose_variant={pose_variant})",
        flush=True,
    )


def iter_sequence_dirs(output_root: Path) -> List[Path]:
    if (output_root / "depth" / "dpt").is_dir():
        return [output_root]
    return [
        path
        for path in sorted(output_root.iterdir())
        if path.is_dir() and (path / "depth" / "dpt").is_dir()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse HorizonStream depth maps directly from native outputs.")
    parser.add_argument("--sequence-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pose-variant", default="auto", choices=["auto", "online", "main", "abs", "offline", "loop", "lc"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--source-window", type=int, default=5)
    parser.add_argument("--max-source-views", type=int, default=6)
    parser.add_argument("--min-consistent", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--depth-min", type=float, default=1e-3)
    parser.add_argument("--depth-max", type=float, default=40.0)
    parser.add_argument("--pixel-threshold", type=float, default=1.0)
    parser.add_argument("--relative-depth-threshold", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--random-sample-ratio", type=float, default=None)
    parser.add_argument("--sky-color-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sky-white-threshold", type=int, default=240)
    parser.add_argument("--sky-bright-any-threshold", type=int, default=200)
    parser.add_argument("--sky-blue-b-min", type=int, default=155)
    parser.add_argument("--sky-blue-r-max", type=int, default=195)
    parser.add_argument("--sky-blue-g-min", type=int, default=100)
    parser.add_argument("--sky-blue-rg-diff-max", type=int, default=95)
    parser.add_argument("--sky-blue-bg-diff-min", type=int, default=8)
    parser.add_argument("--sky-blue-br-diff-min", type=int, default=15)
    parser.add_argument("--outlier-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radius-outlier-radius", type=float, default=0.10)
    parser.add_argument("--radius-outlier-min-points", type=int, default=16)
    args = parser.parse_args()
    if args.sequence_dir is None and args.output_root is None:
        raise SystemExit("Provide --sequence-dir or --output-root.")
    if args.sequence_dir is not None:
        fuse_sequence(args)
        return

    sequence_dirs = iter_sequence_dirs(Path(args.output_root).resolve())
    if not sequence_dirs:
        raise RuntimeError(f"No HorizonStream sequence outputs found under {args.output_root}")
    if args.output is not None and len(sequence_dirs) > 1:
        raise ValueError("--output can only be used with one --sequence-dir or a single sequence --output-root")
    for sequence_dir in sequence_dirs:
        seq_args = copy(args)
        seq_args.sequence_dir = str(sequence_dir)
        fuse_sequence(seq_args)


if __name__ == "__main__":
    main()
