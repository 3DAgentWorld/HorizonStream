import argparse
import colorsys
import math
import os
import pickle
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import viser
import viser.transforms as vt
from PIL import Image, ImageDraw
from plyfile import PlyData
from matplotlib import colormaps

from demo_utils import viser_wrapper


def _to_numpy_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_numpy_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_numpy_recursive(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_recursive(v) for v in value)
    return value


def _load_pt(path: str) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Loading .pt requires torch. Either install torch, or convert to .npz/.pkl first."
        ) from exc

    payload = torch.load(path, map_location="cpu", weights_only=False)

    def _convert(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        if isinstance(v, dict):
            return {k: _convert(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_convert(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_convert(x) for x in v)
        return v

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported .pt payload type: {type(payload)}")
    return _convert(payload)


def _assign_nested(target: dict, key: str, value: Any) -> None:
    parts = key.split("/")
    cur = target
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_npz(path: str) -> dict:
    loaded = np.load(path, allow_pickle=True)
    keys = list(loaded.files)

    if len(keys) == 1:
        only_key = keys[0]
        obj = loaded[only_key]
        if obj.dtype == object and obj.shape == ():
            data = obj.item()
            if not isinstance(data, dict):
                raise ValueError(f"Unsupported object payload in {path}: {type(data)}")
            return _to_numpy_recursive(data)

    out: dict = {}
    for k in keys:
        v = loaded[k]
        if v.dtype == object and v.shape == ():
            v = v.item()
        _assign_nested(out, k, v)
    return _to_numpy_recursive(out)


def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported .pkl payload type: {type(payload)}")
    return _to_numpy_recursive(payload)


def _pack_point_cloud_to_pred_dict(points: np.ndarray, colors: np.ndarray) -> dict:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must be (N, 3), got {colors.shape}")
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same N")
    if points.shape[0] == 0:
        raise ValueError("Empty point cloud in .ply")

    n = points.shape[0]
    h = int(np.ceil(np.sqrt(n)))
    w = int(np.ceil(n / h))
    grid_n = h * w

    points_grid = np.zeros((1, h, w, 3), dtype=np.float32)
    images_grid = np.zeros((1, h, w, 3), dtype=np.uint8)
    conf_grid = np.zeros((1, h, w), dtype=np.float32)

    points_grid.reshape(grid_n, 3)[:n] = points.astype(np.float32, copy=False)
    images_grid.reshape(grid_n, 3)[:n] = colors.astype(np.uint8, copy=False)
    conf_grid.reshape(grid_n)[:n] = 1.0

    camera_poses = np.eye(4, dtype=np.float32)[None, ...]
    return {
        "images": images_grid,
        "points": points_grid,
        "conf": conf_grid,
        "camera_poses": camera_poses,
    }


def _extract_frame_id(path_like: str) -> Optional[int]:
    m = re.search(r"frame_(\d+)", Path(path_like).stem)
    if m:
        return int(m.group(1))
    return None


def _read_ply_points_colors(
    path: str,
    ply_stride: int,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(ply_stride))
    max_points_int = None if max_points in (None, "", "null") else int(max_points)
    try:
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    break
                header_lines.append(line.decode("ascii", errors="ignore").strip())
                if line.strip() == b"end_header":
                    break
            data_offset = f.tell()
        fmt = ""
        vertex_count = None
        properties = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["element", "vertex"] and len(parts) >= 3:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            elif in_vertex and parts[0] == "property" and len(parts) >= 3:
                properties.append((parts[1], parts[2]))
        if fmt == "binary_little_endian" and vertex_count is not None:
            dtype_map = {
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "float64": "<f8",
                "uchar": "u1",
                "uint8": "u1",
                "char": "i1",
                "int8": "i1",
                "ushort": "<u2",
                "uint16": "<u2",
                "short": "<i2",
                "int16": "<i2",
                "uint": "<u4",
                "uint32": "<u4",
                "int": "<i4",
                "int32": "<i4",
            }
            if properties and all(kind in dtype_map for kind, _ in properties):
                dtype = np.dtype([(name, dtype_map[kind]) for kind, name in properties])
                names = dtype.names or ()
                if all(k in names for k in ("x", "y", "z")):
                    step = stride
                    if max_points_int is not None and max_points_int > 0:
                        step = max(step, int(np.ceil(vertex_count / max_points_int)))
                    indices = np.arange(0, vertex_count, step, dtype=np.int64)
                    if max_points_int is not None and max_points_int > 0:
                        indices = indices[:max_points_int]
                    vertex = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,))
                    sampled = np.asarray(vertex[indices])
                    points = np.stack(
                        [
                            np.asarray(sampled["x"], dtype=np.float32),
                            np.asarray(sampled["y"], dtype=np.float32),
                            np.asarray(sampled["z"], dtype=np.float32),
                        ],
                        axis=1,
                    )
                    color_names = ("red", "green", "blue")
                    if all(k in names for k in color_names):
                        colors = np.stack(
                            [
                                np.asarray(sampled["red"]),
                                np.asarray(sampled["green"]),
                                np.asarray(sampled["blue"]),
                            ],
                            axis=1,
                        )
                        colors = np.clip(colors, 0, 255).astype(np.uint8)
                    else:
                        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
                    return points, colors
    except Exception as exc:
        print(f"[viser] warning: fast PLY read failed for {path}: {exc}", flush=True)

    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")

    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    required = ("x", "y", "z")
    if any(k not in names for k in required):
        raise ValueError(f"{path} vertex is missing one of {required}")

    points = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )

    color_names = ("red", "green", "blue")
    if all(k in names for k in color_names):
        colors = np.stack(
            [
                np.asarray(vertex["red"]),
                np.asarray(vertex["green"]),
                np.asarray(vertex["blue"]),
            ],
            axis=1,
        )
        if np.issubdtype(colors.dtype, np.floating):
            max_val = float(np.nanmax(colors)) if colors.size > 0 else 0.0
            if max_val <= 1.0:
                colors = np.clip(colors * 255.0, 0, 255)
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

    if stride > 1:
        points = points[::stride]
        colors = colors[::stride]
    if max_points_int is not None:
        points, colors = _limit_points(points, colors, max_points_int)
    return points, colors


def _load_ply(path: str) -> dict:
    points, colors = _read_ply_points_colors(path, ply_stride=1)
    return _pack_point_cloud_to_pred_dict(points, colors)


def _invert_w2c_pose_robust(t_w2c: np.ndarray) -> np.ndarray:
    r = t_w2c[:3, :3].astype(np.float64, copy=False)
    trans = t_w2c[:3, 3].astype(np.float64, copy=False)
    if not np.isfinite(r).all() or not np.isfinite(trans).all():
        raise ValueError("pose contains non-finite values")

    # Project to the closest proper rotation to avoid instability on slightly non-rigid matrices.
    u, _, vt_mat = np.linalg.svd(r, full_matrices=False)
    r_ortho = u @ vt_mat
    if np.linalg.det(r_ortho) < 0.0:
        u[:, -1] *= -1.0
        r_ortho = u @ vt_mat
    if not np.isfinite(r_ortho).all():
        raise ValueError("invalid rotation after orthogonalization")
    r_inv = r_ortho.T

    t_c2w = np.eye(4, dtype=np.float64)
    t_c2w[:3, :3] = r_inv.astype(np.float64, copy=False)
    t_c2w[:3, 3] = -r_inv @ trans
    if not np.isfinite(t_c2w).all():
        raise ValueError("inverted pose contains non-finite values")
    return t_c2w.astype(np.float32)


def _orthogonalize_pose_rotation(t_in: np.ndarray) -> np.ndarray:
    t = t_in.astype(np.float64, copy=True)
    r = t[:3, :3]
    u, _, vt_mat = np.linalg.svd(r, full_matrices=False)
    r_ortho = u @ vt_mat
    if np.linalg.det(r_ortho) < 0.0:
        u[:, -1] *= -1.0
        r_ortho = u @ vt_mat
    t[:3, :3] = r_ortho
    return t.astype(np.float32)


def _build_pose_matrix(vals: np.ndarray, pose_layout: str) -> np.ndarray:
    t = np.eye(4, dtype=np.float32)
    if pose_layout == "row":
        t[:3, :] = vals.reshape(3, 4)
        return t
    if pose_layout == "col":
        t[:3, :] = vals.reshape(4, 3).T
        return t
    if pose_layout == "r9t3":
        t[:3, :3] = vals[:9].reshape(3, 3)
        t[:3, 3] = vals[9:12]
        return t
    raise ValueError(f"Unsupported pose layout: {pose_layout}")


def _rotation_orthogonality_error(t: np.ndarray) -> float:
    r = t[:3, :3].astype(np.float64, copy=False)
    return float(np.linalg.norm(r.T @ r - np.eye(3)))


def _infer_pose_layout(vals_samples: List[np.ndarray]) -> str:
    # Pick the layout with smallest average rotation orthogonality error.
    # Prefer "col" on ties, because many sequence outputs store 3x4 in column-major.
    candidates = ["col", "r9t3", "row"]
    best_layout = candidates[0]
    best_err = float("inf")
    for layout in candidates:
        errs = []
        for vals in vals_samples:
            t = _build_pose_matrix(vals, layout)
            errs.append(_rotation_orthogonality_error(t))
        mean_err = float(np.mean(errs)) if errs else float("inf")
        if mean_err < best_err:
            best_err = mean_err
            best_layout = layout
    return best_layout


def _parse_pose_file(path: str, pose_convention: str, pose_layout: str) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    header_mode: Optional[str] = None
    auto_idx = 0
    skipped = 0
    vals_buffer: List[Tuple[int, np.ndarray]] = []
    with open(path, "r") as f:
        for line_idx, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "w2c" in line.lower():
                    header_mode = "w2c"
                elif "c2w" in line.lower():
                    header_mode = "c2w"
                continue
            parts = line.split()
            if len(parts) == 13:
                frame_idx = int(float(parts[0]))
                vals = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            elif len(parts) == 12:
                frame_idx = auto_idx
                auto_idx += 1
                vals = np.asarray([float(x) for x in parts], dtype=np.float32)
            else:
                continue
            vals_buffer.append((frame_idx, vals))

    if not vals_buffer:
        raise ValueError(f"No valid poses parsed from: {path}")

    effective_layout = pose_layout
    if pose_layout == "auto":
        sample_vals = [x[1] for x in vals_buffer[: min(64, len(vals_buffer))]]
        effective_layout = _infer_pose_layout(sample_vals)

    for frame_idx, vals in vals_buffer:
        try:
            t = _build_pose_matrix(vals, effective_layout)
            effective_convention = pose_convention
            if effective_convention == "auto":
                effective_convention = header_mode if header_mode in {"w2c", "c2w"} else "w2c"
            if effective_convention == "w2c":
                t = _invert_w2c_pose_robust(t)
            elif effective_convention != "c2w":
                raise ValueError(f"Unsupported pose convention: {effective_convention}")
            t = _orthogonalize_pose_rotation(t)
            if not np.isfinite(t).all():
                raise ValueError("non-finite pose")
        except Exception as exc:
            skipped += 1
            print(f"[pose] skip frame {frame_idx}: {exc}")
            continue
        poses[frame_idx] = t

    if not poses:
        raise ValueError(f"No valid poses parsed after conversion: {path}")
    if skipped > 0:
        print(f"[pose] parsed {len(poses)} poses, skipped {skipped} invalid rows from {path}")
    return poses


def _parse_intri_file(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    intri: Dict[int, Tuple[float, float, float, float]] = {}
    auto_idx = 0
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 5:
                frame_idx = int(float(parts[0]))
                fx, fy, cx, cy = [float(x) for x in parts[1:]]
            elif len(parts) == 4:
                frame_idx = auto_idx
                auto_idx += 1
                fx, fy, cx, cy = [float(x) for x in parts]
            else:
                continue
            intri[frame_idx] = (fx, fy, cx, cy)
    return intri


def _read_image_preview(path: str, video_width: int) -> np.ndarray:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w <= 0 or h <= 0:
                return np.zeros((64, 64, 3), dtype=np.uint8)
            new_w = int(video_width)
            new_h = max(1, int(h * (new_w / w)))
            im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return np.zeros((64, 64, 3), dtype=np.uint8)


def _discover_sequence_paths(
    sequence_dir: str,
    points_dir: Optional[str],
    poses_dir: Optional[str],
    rgb_dir: Optional[str],
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    base = Path(sequence_dir)
    points_frames_final = points_dir or str(base / "points" / "frames")
    points_full_final = str(base / "points" / "full.ply")
    poses_final = poses_dir or str(base / "poses")
    rgb_final = rgb_dir or str(base / "images" / "rgb")
    if not os.path.isdir(points_frames_final):
        points_frames_final = None
    if not os.path.isfile(points_full_final):
        points_full_final = None
    if not os.path.isdir(poses_final):
        raise FileNotFoundError(f"Poses directory not found: {poses_final}")
    if not os.path.isdir(rgb_final):
        rgb_final = None
    return points_frames_final, points_full_final, poses_final, rgb_final


def _discover_frame_plys(points_frames_path: Optional[str]) -> Dict[int, str]:
    if not points_frames_path or not os.path.isdir(points_frames_path):
        return {}
    ply_by_id: Dict[int, str] = {}
    for p in Path(points_frames_path).glob("*.ply"):
        fid = _extract_frame_id(str(p))
        if fid is not None:
            ply_by_id[fid] = str(p)
    return ply_by_id


def _compute_fov_aspect(
    intri: Optional[Tuple[float, float, float, float]],
    image_for_shape: np.ndarray,
) -> Tuple[float, float]:
    h, w = image_for_shape.shape[:2]
    if h <= 0 or w <= 0:
        return 1.047, 1.0
    if intri is None:
        return 1.047, float(w) / float(h)
    fx, fy, _, _ = intri
    if fx <= 1e-6:
        return 1.047, float(w) / float(h)
    fov = 2.0 * math.atan((w * 0.5) / fx)
    aspect = float(w) / float(h)
    return float(fov), float(aspect)


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return points[keep_idx], colors[keep_idx]


def _limit_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    step = int(np.ceil(points.shape[0] / max_points))
    return points[::step][:max_points], colors[::step][:max_points]


def _filter_white_points(
    points: np.ndarray,
    colors: np.ndarray,
    white_threshold: int = 245,
    bright_channel_threshold: int = 200,
    black_threshold: int = 8,
    sky_blue_min_b: int = 170,
    sky_blue_max_r: int = 160,
    sky_blue_min_g: int = 120,
    sky_blue_max_rg_diff: int = 80,
    sky_blue_min_bg_diff: int = 8,
    sky_blue_min_br_diff: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors
    # Remove near-white points, near-black points, and sky-blue artifacts.
    is_white = (
        (colors[:, 0] >= white_threshold)
        & (colors[:, 1] >= white_threshold)
        & (colors[:, 2] >= white_threshold)
    )
    is_bright_any = (
        (colors[:, 0] > bright_channel_threshold)
        | (colors[:, 1] > bright_channel_threshold)
        | (colors[:, 2] > bright_channel_threshold)
    )
    is_black = (
        (colors[:, 0] <= black_threshold)
        & (colors[:, 1] <= black_threshold)
        & (colors[:, 2] <= black_threshold)
    )
    rg_diff = np.abs(colors[:, 0].astype(np.int16) - colors[:, 1].astype(np.int16))
    bg_diff = colors[:, 2].astype(np.int16) - colors[:, 1].astype(np.int16)
    br_diff = colors[:, 2].astype(np.int16) - colors[:, 0].astype(np.int16)
    is_sky_blue = (
        (colors[:, 2] >= sky_blue_min_b)
        & (colors[:, 0] <= sky_blue_max_r)
        & (colors[:, 1] >= sky_blue_min_g)
        & (rg_diff <= sky_blue_max_rg_diff)
        & (bg_diff >= sky_blue_min_bg_diff)
        & (br_diff >= sky_blue_min_br_diff)
        & (colors[:, 2] > colors[:, 1])
    )
    # Additional mask for pale cyan haze (high value, low saturation).
    rgb_i16 = colors.astype(np.int16)
    cmax = np.max(rgb_i16, axis=1)
    cmin = np.min(rgb_i16, axis=1)
    sat = (cmax - cmin).astype(np.float32) / np.maximum(cmax, 1).astype(np.float32)
    is_cyan_haze = (
        (colors[:, 1] >= 170)
        & (colors[:, 2] >= 170)
        & (colors[:, 0] <= 220)
        & ((colors[:, 1].astype(np.int16) - colors[:, 0].astype(np.int16)) >= 8)
        & ((colors[:, 2].astype(np.int16) - colors[:, 0].astype(np.int16)) >= 8)
        & (np.abs(colors[:, 2].astype(np.int16) - colors[:, 1].astype(np.int16)) <= 30)
        & (cmax >= 170)
        & (sat <= 0.23)
    )

    keep = ~(is_white | is_bright_any | is_black | is_sky_blue | is_cyan_haze)
    return points[keep], colors[keep]


def _depth_to_rgb(depth: np.ndarray, out_width: int = 320) -> np.ndarray:
    if depth.ndim != 2:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((64, 64, 3), dtype=np.uint8)
    dv = depth[valid]
    lo = float(np.percentile(dv, 2.0))
    hi = float(np.percentile(dv, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    # Invert so near is warm (orange) and far is cool (blue).
    norm = 1.0 - norm
    norm[~valid] = 0.0
    # Use a vivid depth colormap close to the requested palette.
    cmap = colormaps["turbo"]
    rgb = cmap(norm)[..., :3]
    img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    new_w = int(out_width)
    new_h = max(1, int(h * (new_w / w)))
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.uint8)


# Output size for the pose trajectory plot (independent of video_width).
_POSE_PLOT_SIZE: int = 512
# Supersampling factor: render internally at SS× to get anti-aliased lines via LANCZOS downscale.
_POSE_PLOT_SS: int = 2

# Incremental pose plot cache — avoids O(N) PIL redraw every frame during playback.
# bg_img is stored at the supersampled resolution (SS × output_size).
_pose_plot_cache: Dict[str, Any] = {
    "pred_pts": None,   # List[Tuple[float,float]] — pixel coords at SS resolution
    "gt_pts": None,
    "bg_img": None,     # PIL Image at SS resolution with trajectory drawn up to bg_idx
    "bg_idx": -1,
    "width": 0,         # output width (before downscaling)
    "height": 0,
}

# Background / foreground colours for the plot.
_PLOT_BG = (255, 255, 255)       # white background
_PLOT_GT_COL = (220, 40, 40)     # red for ground-truth
_PLOT_GT_WIDTH = 4               # GT line width at output resolution
_PLOT_PRED_WIDTH = 3             # pred line width at output resolution
_PLOT_DOT_R = 6                  # current-position dot radius at output resolution


def _render_pose_plot(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    current_idx: int,
    playing: bool,
    width: int = _POSE_PLOT_SIZE,
    height: int = _POSE_PLOT_SIZE,
) -> np.ndarray:
    """Render a top-down trajectory plot.

    Internally draws at SS× resolution then downscales with LANCZOS for smooth,
    anti-aliased lines.  During forward playback only one new segment is added
    per frame (O(1)) to avoid progressive slowdown.
    """
    cache = _pose_plot_cache
    SS = _POSE_PLOT_SS
    sw, sh = width * SS, height * SS  # supersampled canvas size

    # ------------------------------------------------------------------
    # Rebuild coordinate mapping when size changes or on first call.
    # ------------------------------------------------------------------
    if cache["pred_pts"] is None or cache["width"] != width or cache["height"] != height:
        blank = Image.new("RGB", (sw, sh), _PLOT_BG)
        if pred_xyz.size == 0 and gt_xyz.size == 0:
            return np.asarray(blank.resize((width, height), Image.LANCZOS), dtype=np.uint8)

        pred_xz = pred_xyz[:, [0, 2]] if pred_xyz.size > 0 else np.zeros((0, 2), dtype=np.float32)
        gt_xz = gt_xyz[:, [0, 2]] if gt_xyz.size > 0 else np.zeros((0, 2), dtype=np.float32)
        all_xz = (
            np.concatenate([pred_xz, gt_xz], axis=0)
            if pred_xz.size > 0 and gt_xz.size > 0
            else (pred_xz if gt_xz.size == 0 else gt_xz)
        )

        min_xy = all_xz.min(axis=0)
        max_xy = all_xz.max(axis=0)
        center = 0.5 * (min_xy + max_xy)
        span_scalar = float(max(*(np.maximum(max_xy - min_xy, 1e-6).tolist()))) * 1.18
        min_xy = center - 0.5 * span_scalar
        span = np.array([span_scalar, span_scalar], dtype=np.float32)

        def _map_pts(xz: np.ndarray) -> List[Tuple[float, float]]:
            if xz.size == 0:
                return []
            u = (xz[:, 0] - min_xy[0]) / span[0]
            v = (xz[:, 1] - min_xy[1]) / span[1]
            px = u * (sw - 1)
            py = (1.0 - v) * (sh - 1)
            return list(zip(px.tolist(), py.tolist()))

        pred_pts = _map_pts(pred_xz)
        gt_pts = _map_pts(gt_xz)

        # Pre-draw GT on the static base layer (never redrawn during playback).
        base = Image.new("RGB", (sw, sh), _PLOT_BG)
        if len(gt_pts) >= 2:
            ImageDraw.Draw(base).line(gt_pts, fill=_PLOT_GT_COL, width=_PLOT_GT_WIDTH * SS)

        cache["pred_pts"] = pred_pts
        cache["gt_pts"] = gt_pts
        cache["bg_img"] = base
        cache["bg_idx"] = -1
        cache["width"] = width
        cache["height"] = height

    pred_pts = cache["pred_pts"]

    if not pred_pts:
        out = cache["bg_img"].resize((width, height), Image.LANCZOS)
        return np.asarray(out, dtype=np.uint8)

    # end_idx: show full trajectory when paused; up to current frame during play.
    end_idx = len(pred_pts) - 1 if not playing else max(0, min(current_idx, len(pred_pts) - 1))

    # ------------------------------------------------------------------
    # Incremental update: O(1) per frame during forward playback.
    # ------------------------------------------------------------------
    if playing and cache["bg_idx"] >= 0 and end_idx == cache["bg_idx"] + 1:
        draw = ImageDraw.Draw(cache["bg_img"])
        i = end_idx
        hue = 0.0 if len(pred_pts) <= 1 else i / float(len(pred_pts) - 1)
        col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        draw.line([pred_pts[i - 1], pred_pts[i]], fill=col, width=_PLOT_PRED_WIDTH * SS)
        cache["bg_idx"] = end_idx
    elif cache["bg_idx"] != end_idx:
        # Full redraw (first frame, backward jump, pause → play transition).
        base = Image.new("RGB", (sw, sh), _PLOT_BG)
        draw = ImageDraw.Draw(base)
        gt_pts = cache["gt_pts"]
        if len(gt_pts) >= 2:
            draw.line(gt_pts, fill=_PLOT_GT_COL, width=_PLOT_GT_WIDTH * SS)
        for i in range(1, end_idx + 1):
            hue = 0.0 if len(pred_pts) <= 1 else i / float(len(pred_pts) - 1)
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
            draw.line([pred_pts[i - 1], pred_pts[i]], fill=col, width=_PLOT_PRED_WIDTH * SS)
        cache["bg_img"] = base
        cache["bg_idx"] = end_idx

    # ------------------------------------------------------------------
    # Compose final frame: downsample background, then overlay the
    # current-position marker (drawn at output resolution for sharpness).
    # ------------------------------------------------------------------
    canvas = cache["bg_img"].resize((width, height), Image.LANCZOS)
    if 0 <= end_idx < len(pred_pts):
        # Map back from SS coords to output coords.
        cx = pred_pts[end_idx][0] / SS
        cy = pred_pts[end_idx][1] / SS
        r = _PLOT_DOT_R
        draw = ImageDraw.Draw(canvas)
        # Outer border ring (dark gray, visible on white bg).
        draw.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill=(60, 60, 60))
        # Coloured fill matching current hue.
        hue = 0.0 if len(pred_pts) <= 1 else end_idx / float(len(pred_pts) - 1)
        dot_col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=dot_col)
        # Small white highlight.
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(255, 255, 255))

    return np.asarray(canvas, dtype=np.uint8)


def _filter_frame_distance(
    points: np.ndarray,
    colors: np.ndarray,
    cam_center: np.ndarray,
    min_dist: float,
    max_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors
    d = np.linalg.norm(points - cam_center[None, :], axis=1)
    keep = d >= max(0.0, float(min_dist))
    if max_dist > 0.0:
        keep &= d <= float(max_dist)
    if np.any(keep):
        return points[keep], colors[keep]
    return points, colors


def _filter_points_by_camera_frustum(
    points_world: np.ndarray,
    colors: np.ndarray,
    pose_c2w: np.ndarray,
    intri: Optional[Tuple[float, float, float, float]],
    near_dist: float = 0.0,
    far_dist: float = 0.0,
    fov_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    keep = _camera_frustum_visible_mask(
        points_world,
        pose_c2w=pose_c2w,
        intri=intri,
        near_dist=near_dist,
        far_dist=far_dist,
        fov_scale=fov_scale,
    )
    return points_world[keep], colors[keep]


def _camera_frustum_visible_mask(
    points_world: np.ndarray,
    pose_c2w: np.ndarray,
    intri: Optional[Tuple[float, float, float, float]],
    near_dist: float = 0.0,
    far_dist: float = 0.0,
    fov_scale: float = 1.0,
) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    fov_scale = max(1e-6, float(fov_scale))
    r = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    points_cam = (points_world - t[None, :]) @ r
    z = points_cam[:, 2]
    # Use z-depth (along camera optical axis) for near/far range — more intuitive
    # than Euclidean distance and matches what users expect from "depth limit".
    keep = z > max(1e-4, float(near_dist))
    if far_dist > 0.0:
        keep &= z <= float(far_dist)
    if not np.any(keep):
        return keep

    if intri is not None:
        fx, fy, cx, cy = intri
        if fx > 1e-6 and fy > 1e-6:
            fx_eff = fx / fov_scale
            fy_eff = fy / fov_scale
            x = points_cam[:, 0]
            y = points_cam[:, 1]
            u = fx_eff * (x / np.maximum(z, 1e-6)) + cx
            v = fy_eff * (y / np.maximum(z, 1e-6)) + cy
            width = max(1.0, 2.0 * float(cx))
            height = max(1.0, 2.0 * float(cy))
            keep &= (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
            return keep

    # Fallback with a generic frustum when intrinsics are unavailable.
    fov = float(np.clip(1.047 * fov_scale, 0.05, 3.1))
    aspect = 1.0
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    x_lim = np.tan(fov * 0.5) * aspect * np.maximum(z, 1e-6)
    y_lim = np.tan(fov * 0.5) * np.maximum(z, 1e-6)
    keep &= (np.abs(x) <= x_lim) & (np.abs(y) <= y_lim)
    return keep


def _compose_panel_frame(rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> np.ndarray:
    def _to_rgb_u8(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = np.repeat(x[:, :, None], 3, axis=2)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    rgb = _to_rgb_u8(rgb)
    depth = _to_rgb_u8(depth)
    pose = _to_rgb_u8(pose)
    target_w = max(rgb.shape[1], depth.shape[1], pose.shape[1], 1)

    def _resize_to_w(img: np.ndarray, w: int) -> np.ndarray:
        h, iw = img.shape[:2]
        if iw == w:
            return img
        nh = max(1, int(round(h * (w / max(1, iw)))))
        return np.asarray(Image.fromarray(img).resize((w, nh), Image.Resampling.BILINEAR), dtype=np.uint8)

    rgb = _resize_to_w(rgb, target_w)
    depth = _resize_to_w(depth, target_w)
    pose = _resize_to_w(pose, target_w)
    gap = 4
    canvas_h = rgb.shape[0] + depth.shape[0] + pose.shape[0] + gap * 2
    canvas = np.zeros((canvas_h, target_w, 3), dtype=np.uint8)
    y = 0
    canvas[y : y + rgb.shape[0]] = rgb
    y += rgb.shape[0] + gap
    canvas[y : y + depth.shape[0]] = depth
    y += depth.shape[0] + gap
    canvas[y : y + pose.shape[0]] = pose
    return canvas


def _ensure_hwc_rgb_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        # Handle CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _ensure_hwc_rgba_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate([_ensure_hwc_rgb_u8(rgb), alpha], axis=2))
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim for RGBA: {arr.ndim}")
    if arr.shape[2] == 4:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    rgb = _ensure_hwc_rgb_u8(arr)
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))


def _rotate_vec_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = axis / (np.linalg.norm(axis) + 1e-8)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis_n, v) * s + axis_n * np.dot(axis_n, v) * (1.0 - c)


def _visualize_sequence_dir_simple(
    sequence_dir: str,
    pose_file: str,
    gt_pose_file: Optional[str],
    pose_convention: str,
    pose_layout: str,
    points_dir: Optional[str],
    poses_dir: Optional[str],
    rgb_dir: Optional[str],
    intri_file: Optional[str],
    port: int,
    share: bool,
    background_mode: bool,
    point_size: float,
    ply_stride: int,
    video_width: int,
    global_max_points: int = 2_000_000,
    cleanup_paths: Optional[List[str]] = None,
) -> None:
    points_frames_path, points_full_path, poses_path, rgb_path = _discover_sequence_paths(sequence_dir, points_dir, poses_dir, rgb_dir)
    loop_points_path = os.path.join(sequence_dir, "points", "full_lc.ply")
    if os.path.isfile(loop_points_path):
        points_full_path = loop_points_path
    if points_full_path is None:
        raise FileNotFoundError(f"Global point cloud not found: {os.path.join(sequence_dir, 'points', 'full.ply')}")
    pose_path = os.path.join(poses_path, pose_file)
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    if intri_file is None:
        intri_guess = pose_file.replace("abs_pose", "intri")
        intri_path = os.path.join(poses_path, intri_guess)
        if not os.path.isfile(intri_path):
            intri_path = os.path.join(poses_path, "intri.txt")
    else:
        intri_path = os.path.join(poses_path, intri_file)
    if not os.path.isfile(intri_path):
        intri_path = None

    pose_by_id = _parse_pose_file(pose_path, pose_convention=pose_convention, pose_layout=pose_layout)
    gt_by_id = {}
    gt_candidate_paths: List[str] = []
    if gt_pose_file:
        gt_candidate_paths.append(os.path.join(poses_path, gt_pose_file))
    else:
        for name in ("gt_pose.txt", "abs_pose_gt.txt", "gt.txt"):
            gt_candidate_paths.append(os.path.join(poses_path, name))
    for gt_path in gt_candidate_paths:
        if not os.path.isfile(gt_path):
            continue
        try:
            gt_by_id = _parse_pose_file(gt_path, pose_convention=pose_convention, pose_layout=pose_layout)
            break
        except Exception as exc:
            print(f"[gt] failed to parse {gt_path}: {exc}")

    intri_by_id = _parse_intri_file(intri_path) if intri_path else {}
    rgb_by_id: Dict[int, str] = {}
    if rgb_path is not None:
        for p in Path(rgb_path).glob("*.png"):
            fid = _extract_frame_id(str(p))
            if fid is not None:
                rgb_by_id[fid] = str(p)
    depth_by_id: Dict[int, str] = {}
    depth_dir = os.path.join(sequence_dir, "depth", "dpt")
    if os.path.isdir(depth_dir):
        for p in Path(depth_dir).glob("*.npy"):
            fid = _extract_frame_id(str(p))
            if fid is not None:
                depth_by_id[fid] = str(p)

    frame_ids = sorted(pose_by_id.keys())
    if not frame_ids:
        raise ValueError("No frame ids found in pose file.")
    pred_xyz_all = np.array([pose_by_id[k][:3, 3] for k in frame_ids], dtype=np.float32)
    if gt_by_id:
        gt_xyz_all = np.array([gt_by_id[k][:3, 3] if k in gt_by_id else pose_by_id[k][:3, 3] for k in frame_ids], dtype=np.float32)
    else:
        gt_xyz_all = pred_xyz_all

    server = viser.ViserServer(host="0.0.0.0", port=port)
    _share_url: Optional[str] = None
    if share:
        _share_url = server.request_share_url()
    server.scene.set_up_direction("-y")

    with server.gui.add_folder("Preview", order=0, expand_by_default=True):
        preview_img = np.zeros((64, 64, 3), dtype=np.uint8)
        preview_handle = server.gui.add_image(preview_img, format="jpeg", label="RGB")
        depth_handle = server.gui.add_image(preview_img, format="jpeg", label="Depth")
        pose_plot_handle = server.gui.add_image(
            np.full((_POSE_PLOT_SIZE, _POSE_PLOT_SIZE, 3), 255, dtype=np.uint8),
            format="jpeg",
            label="Pose",
        )

    with server.gui.add_folder("Playback", order=10, expand_by_default=True):
        gui_play = server.gui.add_checkbox("Playing", False)
        gui_pause = server.gui.add_button("Pause")
        gui_frame = server.gui.add_slider("Frame", 0, len(frame_ids) - 1, 1, 0)
        gui_prev = server.gui.add_button("Prev")
        gui_next = server.gui.add_button("Next")
        gui_fps = server.gui.add_slider("FPS", 1, 60, 0.1, 20)
        gui_play_max_points = server.gui.add_slider("Playback Max Points", 5000, 500000, 5000, 120000)

    with server.gui.add_folder("Scene", order=20, expand_by_default=True):
        gui_point_size = server.gui.add_slider("Point Size", 0.0001, 5.0, 0.0001, point_size)
        gui_display_mode = server.gui.add_dropdown(
            "Display Mode",
            options=("cumulative", "single", "all"),
            initial_value="cumulative",
        )
        gui_cloud_status = server.gui.add_markdown("Global cloud: loading")
        gui_show_cam = server.gui.add_checkbox("Show Cameras", True)
        gui_cam_accumulate = server.gui.add_checkbox("Accumulate Cameras", True)
        gui_cam_size = server.gui.add_slider("Camera Size", 0.01, 5.0, 0.01, 0.04)
        gui_cam_stride = server.gui.add_slider("Camera Stride", 1, 64, 1, 1)
        gui_cam_window = server.gui.add_slider("Camera Window (0=all)", 0, len(frame_ids) - 1, 1, 0)
        gui_fov_scale = server.gui.add_slider("FOV Scale", 0.25, 3.0, 0.05, 1.0)
        gui_depth_occlusion = server.gui.add_checkbox("Depth Visibility", True)
        gui_depth_tol_rel = server.gui.add_slider("Depth Tol Rel", 0.0, 0.5, 0.005, 0.04)
        gui_depth_tol_abs = server.gui.add_slider("Depth Tol Abs", 0.0, 2.0, 0.01, 0.08)
        gui_follow_cam = server.gui.add_checkbox("Follow Camera", False)
        gui_follow_distance = server.gui.add_slider("Follow Distance", 0.0, 200.0, 0.1, 1.5)
        gui_follow_lag = server.gui.add_slider("Follow Lag", 0, len(frame_ids) - 1, 1, 0)

    with server.gui.add_folder("Save Video", order=30, expand_by_default=True):
        default_capture_dir = os.path.join(sequence_dir, "captures")
        gui_capture_dir = server.gui.add_text("Save Dir", initial_value=default_capture_dir)
        gui_capture_prefix = server.gui.add_text("Prefix", initial_value="viser_capture")
        gui_capture_fps = server.gui.add_slider("Video FPS", 1, 60, 1, 20)
        gui_start_video = server.gui.add_button("Start Save Video")
        gui_stop_video = server.gui.add_button("Stop Save Video")
        gui_capture_status = server.gui.add_markdown("Video: idle")

    pts, cols = _read_ply_points_colors(
        points_full_path,
        max(1, int(ply_stride)),
        max_points=int(global_max_points),
    )
    print(f"[viser] loaded global point cloud: {points_full_path} points={pts.shape[0]} point_size={point_size}")
    gui_cloud_status.content = f"Global cloud: {pts.shape[0]:,} points"
    pcd_handle = server.scene.add_point_cloud(
        "/sequence/pc_global",
        pts.astype(np.float32, copy=False),
        cols.astype(np.uint8, copy=False),
        point_size=point_size,
        visible=pts.shape[0] > 0,
    )
    rgb_preview_cache: Dict[int, np.ndarray] = {}
    depth_preview_cache: Dict[int, np.ndarray] = {}
    depth_map_cache: Dict[int, Optional[np.ndarray]] = {}
    follow_state: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    prev_cam_visible_set: set = set()
    frustum_mask_cache: Dict[int, np.ndarray] = {}
    play_index_cache: Dict[int, np.ndarray] = {}
    accum_mask_state: Dict[str, Any] = {
        "indices": (),
        "mask": np.zeros((pts.shape[0],), dtype=bool),
    }
    cloud_state: Dict[str, Any] = {
        "key": None,
        "point_size": None,
    }
    capture_state: Dict[str, Any] = {
        "last_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "last_depth": np.zeros((64, 64, 3), dtype=np.uint8),
        "last_pose": np.zeros((64, 64, 3), dtype=np.uint8),
        "writer": None,
        "record_dir": None,
        "frame_idx": 0,
        "frame_shape": None,
        "recording": False,
    }
    record_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
    record_stop = threading.Event()

    current_frustum = server.scene.add_camera_frustum(
        "/sequence/cam",
        fov=1.047,
        aspect=1.0,
        scale=gui_cam_size.value,
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        color=(1.0, 1.0, 1.0),
    )
    history_frustums = []
    history_fov_aspects: List[Tuple[float, float]] = []
    for idx, fid in enumerate(frame_ids):
        pose = pose_by_id[fid]
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        wxyz = vt.SO3.from_matrix(rot).wxyz
        hue = 0.0 if len(frame_ids) <= 1 else idx / float(len(frame_ids) - 1)
        hist_fov, hist_aspect = _compute_fov_aspect(intri_by_id.get(fid, None), preview_img)
        history_fov_aspects.append((hist_fov, hist_aspect))
        history_frustums.append(
            server.scene.add_camera_frustum(
                f"/sequence/cam_hist/{idx}",
                fov=hist_fov,
                aspect=hist_aspect,
                scale=gui_cam_size.value,
                wxyz=wxyz,
                position=trans,
                color=colorsys.hsv_to_rgb(hue, 1.0, 1.0),
                line_width=1.2,
                visible=False,
            )
        )

    def _set_capture_status(msg: str) -> None:
        gui_capture_status.content = f"Video: {msg}"

    def _ensure_capture_dir() -> str:
        out_dir = os.path.abspath(os.path.expanduser(str(gui_capture_dir.value).strip()))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _capture_frame_image() -> np.ndarray:
        return _ensure_hwc_rgb_u8(
            _compose_panel_frame(
                capture_state["last_rgb"],
                capture_state["last_depth"],
                capture_state["last_pose"],
            )
        )

    def _close_capture_writer() -> None:
        writer = capture_state.get("writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        capture_state["writer"] = None
        capture_state["frame_shape"] = None

    def _append_capture_frame_from_frame(frame: np.ndarray) -> None:
        out_dir = _ensure_capture_dir()
        prefix = str(gui_capture_prefix.value).strip() or "viser_capture"
        frame = _ensure_hwc_rgb_u8(frame)
        expected_shape = capture_state.get("frame_shape", None)
        if expected_shape is None:
            capture_state["frame_shape"] = frame.shape
        elif tuple(frame.shape) != tuple(expected_shape):
            target_h, target_w = int(expected_shape[0]), int(expected_shape[1])
            frame = np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8)
        if capture_state["writer"] is None:
            import imageio.v2 as imageio

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mp4_path = os.path.join(out_dir, f"{prefix}_{ts}.mp4")
            capture_state["writer"] = imageio.get_writer(mp4_path, fps=int(gui_capture_fps.value), codec="libx264")
            capture_state["record_dir"] = mp4_path
            capture_state["frame_idx"] = 0
            _set_capture_status(f"recording {mp4_path}")
        capture_state["writer"].append_data(frame)
        capture_state["frame_idx"] += 1

    def _record_worker() -> None:
        while not record_stop.is_set():
            try:
                frame = record_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                _append_capture_frame_from_frame(frame)
            except Exception as exc:
                _set_capture_status(f"write failed: {exc}")
            finally:
                record_queue.task_done()

    def _get_rgb_preview(fid: int) -> np.ndarray:
        if fid not in rgb_preview_cache:
            rgb_file = rgb_by_id.get(fid, None)
            rgb_preview_cache[fid] = _read_image_preview(rgb_file, video_width) if rgb_file else np.zeros((64, 64, 3), dtype=np.uint8)
        return rgb_preview_cache[fid]

    def _get_depth_preview(fid: int) -> np.ndarray:
        if fid not in depth_preview_cache:
            depth_file = depth_by_id.get(fid, None)
            if depth_file and os.path.isfile(depth_file):
                try:
                    depth_preview_cache[fid] = _depth_to_rgb(np.load(depth_file), out_width=video_width)
                except Exception:
                    depth_preview_cache[fid] = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                depth_preview_cache[fid] = np.zeros((64, 64, 3), dtype=np.uint8)
        return depth_preview_cache[fid]

    def _get_depth_map(fid: int) -> Optional[np.ndarray]:
        if fid not in depth_map_cache:
            depth_file = depth_by_id.get(fid, None)
            if depth_file and os.path.isfile(depth_file):
                try:
                    depth = np.asarray(np.load(depth_file), dtype=np.float32)
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    depth_map_cache[fid] = depth if depth.ndim == 2 else None
                except Exception:
                    depth_map_cache[fid] = None
            else:
                depth_map_cache[fid] = None
        return depth_map_cache[fid]

    def _update_camera_visibility(frame_idx: int) -> None:
        nonlocal prev_cam_visible_set
        cam_stride = max(1, int(gui_cam_stride.value))
        if not bool(gui_show_cam.value):
            new_visible: set = set()
            want_current = False
        elif bool(gui_play.value) and not bool(gui_cam_accumulate.value):
            snapped_idx = (max(0, frame_idx) // cam_stride) * cam_stride
            new_visible = {snapped_idx} if snapped_idx < len(history_frustums) else set()
            want_current = True
        else:
            if bool(gui_play.value) and bool(gui_cam_accumulate.value):
                cam_window = int(gui_cam_window.value)
                start_idx = max(0, frame_idx - cam_window + 1) if cam_window > 0 else 0
                stop_idx = min(len(history_frustums), frame_idx + 1)
            else:
                start_idx = 0
                stop_idx = len(history_frustums)
            new_visible = {i for i in range(start_idx, stop_idx) if i % cam_stride == 0}
            want_current = True
        for i in prev_cam_visible_set - new_visible:
            history_frustums[i].visible = False
        for i in new_visible - prev_cam_visible_set:
            history_frustums[i].visible = True
        prev_cam_visible_set = new_visible
        current_frustum.visible = want_current

    def _clear_visibility_cache() -> None:
        frustum_mask_cache.clear()
        accum_mask_state["indices"] = ()
        accum_mask_state["mask"] = np.zeros((pts.shape[0],), dtype=bool)
        cloud_state["key"] = None

    def _play_candidate_indices() -> np.ndarray:
        max_points = max(1, int(gui_play_max_points.value))
        if max_points not in play_index_cache:
            if pts.shape[0] <= max_points:
                play_index_cache[max_points] = np.arange(pts.shape[0], dtype=np.int64)
            else:
                step = int(np.ceil(pts.shape[0] / max_points))
                play_index_cache[max_points] = np.arange(0, pts.shape[0], step, dtype=np.int64)[:max_points]
        return play_index_cache[max_points]

    def _depth_visibility_mask(frame_idx: int, base_mask: np.ndarray) -> np.ndarray:
        if not bool(gui_depth_occlusion.value) or not np.any(base_mask):
            return base_mask
        fid = frame_ids[frame_idx]
        depth = _get_depth_map(fid)
        intri = intri_by_id.get(fid, None)
        if depth is None or intri is None:
            return base_mask

        fx, fy, cx, cy = intri
        if fx <= 1e-6 or fy <= 1e-6:
            return base_mask

        idx = np.flatnonzero(base_mask)
        points_world = pts[idx]
        pose = pose_by_id[fid]
        r = pose[:3, :3]
        t = pose[:3, 3]
        points_cam = (points_world - t[None, :]) @ r
        z = points_cam[:, 2]
        valid = z > 1e-4
        if not np.any(valid):
            out = np.zeros_like(base_mask)
            return out

        h, w = depth.shape[:2]
        u = np.rint(fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + cx).astype(np.int64)
        v = np.rint(fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + cy).astype(np.int64)
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            out = np.zeros_like(base_mask)
            return out

        sampled_depth = np.zeros_like(z, dtype=np.float32)
        sampled_depth[valid] = depth[v[valid], u[valid]]
        valid_depth = np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
        tol = float(gui_depth_tol_abs.value) + float(gui_depth_tol_rel.value) * np.maximum(sampled_depth, 0.0)
        visible = valid & valid_depth & (np.abs(z - sampled_depth) <= tol)

        out = np.zeros_like(base_mask)
        out[idx[visible]] = True
        return out

    def _limited_mask(mask: np.ndarray, max_points: int) -> np.ndarray:
        count = int(np.count_nonzero(mask))
        if max_points <= 0 or count <= max_points:
            return mask
        keep_idx = np.flatnonzero(mask)
        step = int(np.ceil(count / max_points))
        limited = np.zeros_like(mask)
        limited[keep_idx[::step][:max_points]] = True
        return limited

    def _set_cloud_from_mask(mask: np.ndarray, key: Tuple[Any, ...], max_points: Optional[int] = None) -> None:
        if max_points is not None:
            mask = _limited_mask(mask, int(max_points))
            key = (*key, "limit", int(max_points))
        point_size_value = float(gui_point_size.value)
        if cloud_state.get("key") == key:
            if cloud_state.get("point_size") != point_size_value:
                pcd_handle.point_size = point_size_value
                cloud_state["point_size"] = point_size_value
            return
        pcd_handle.points = pts[mask]
        pcd_handle.colors = cols[mask]
        pcd_handle.visible = bool(np.any(mask))
        pcd_handle.point_size = point_size_value
        cloud_state["key"] = key
        cloud_state["point_size"] = point_size_value

    def _global_frustum_mask(frame_idx: int) -> np.ndarray:
        if frame_idx in frustum_mask_cache:
            return frustum_mask_cache[frame_idx]
        fid = frame_ids[frame_idx]
        candidate_idx = _play_candidate_indices()
        candidate_mask = _camera_frustum_visible_mask(
            pts[candidate_idx],
            pose_c2w=pose_by_id[fid],
            intri=intri_by_id.get(fid, None),
            fov_scale=float(gui_fov_scale.value),
        )
        mask = np.zeros((pts.shape[0],), dtype=bool)
        mask[candidate_idx[candidate_mask]] = True
        mask = _depth_visibility_mask(frame_idx, mask)
        frustum_mask_cache[frame_idx] = mask
        return mask

    def _cloud_visible_indices(frame_idx: int) -> Tuple[int, ...]:
        if not bool(gui_cam_accumulate.value):
            return (frame_idx,)
        cam_stride = max(1, int(gui_cam_stride.value))
        cam_window = int(gui_cam_window.value)
        start_idx = max(0, frame_idx - cam_window + 1) if cam_window > 0 else 0
        indices = [i for i in range(start_idx, frame_idx + 1) if i % cam_stride == 0]
        if frame_idx not in indices:
            indices.append(frame_idx)
        return tuple(sorted(set(indices)))

    def _update_point_cloud_display(frame_idx: int) -> None:
        mode = str(gui_display_mode.value)
        full_mask = np.ones((pts.shape[0],), dtype=bool)

        if (not bool(gui_play.value)) or mode == "all":
            _set_cloud_from_mask(full_mask, ("full",))
            return

        if mode == "single":
            mask = _global_frustum_mask(frame_idx)
            _set_cloud_from_mask(mask, ("single", frame_idx), max_points=int(gui_play_max_points.value))
            return

        indices = _cloud_visible_indices(frame_idx)
        cached_indices = tuple(accum_mask_state.get("indices", ()))
        if indices == cached_indices:
            mask = accum_mask_state["mask"]
        elif cached_indices and indices[: len(cached_indices)] == cached_indices:
            mask = accum_mask_state["mask"].copy()
            for i in indices[len(cached_indices) :]:
                mask |= _global_frustum_mask(i)
            accum_mask_state["indices"] = indices
            accum_mask_state["mask"] = mask
        else:
            mask = np.zeros((pts.shape[0],), dtype=bool)
            for i in indices:
                mask |= _global_frustum_mask(i)
            accum_mask_state["indices"] = indices
            accum_mask_state["mask"] = mask

        _set_cloud_from_mask(mask, ("cumulative", indices), max_points=int(gui_play_max_points.value))

    def _apply_follow_camera(frame_idx: int) -> None:
        if not bool(gui_follow_cam.value):
            follow_state.clear()
            return
        follow_idx = max(0, frame_idx - int(gui_follow_lag.value))
        pose = pose_by_id[frame_ids[follow_idx]]
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cam_pos = trans - forward * float(gui_follow_distance.value)
        look_at = trans + forward * 3.0
        for client_id, client in server.get_clients().items():
            prev = follow_state.get(client_id)
            if prev is not None:
                cam_pos = 0.75 * prev[0] + 0.25 * cam_pos
                look_at = 0.75 * prev[1] + 0.25 * look_at
            follow_state[client_id] = (cam_pos, look_at)
            client.camera.position = cam_pos
            client.camera.look_at = look_at
            client.camera.up_direction = (0.0, -1.0, 0.0)

    def _update_frame(frame_idx: int) -> None:
        fid = frame_ids[frame_idx]
        preview = _get_rgb_preview(fid)
        depth_img = _get_depth_preview(fid)
        pose_img = _render_pose_plot(
            pred_xyz=pred_xyz_all,
            gt_xyz=gt_xyz_all,
            current_idx=frame_idx,
            playing=bool(gui_play.value),
            width=_POSE_PLOT_SIZE,
            height=_POSE_PLOT_SIZE,
        )
        preview_handle.image = preview
        depth_handle.image = depth_img
        pose_plot_handle.image = pose_img
        capture_state["last_rgb"] = preview
        capture_state["last_depth"] = depth_img
        capture_state["last_pose"] = pose_img

        pose = pose_by_id[fid]
        current_frustum.wxyz = vt.SO3.from_matrix(pose[:3, :3]).wxyz
        current_frustum.position = pose[:3, 3]
        current_frustum.scale = gui_cam_size.value
        fov, aspect = _compute_fov_aspect(intri_by_id.get(fid, None), preview)
        fov_scale = float(gui_fov_scale.value)
        current_frustum.fov = float(np.clip(fov * fov_scale, 0.05, 3.1))
        current_frustum.aspect = aspect
        for h, (hist_fov, hist_aspect) in zip(history_frustums, history_fov_aspects):
            h.scale = gui_cam_size.value
            h.fov = float(np.clip(hist_fov * fov_scale, 0.05, 3.1))
            h.aspect = hist_aspect
        _update_camera_visibility(frame_idx)
        _apply_follow_camera(frame_idx)
        _update_point_cloud_display(frame_idx)
        if capture_state.get("recording"):
            try:
                record_queue.put_nowait(_capture_frame_image())
            except queue.Full:
                pass

    @gui_next.on_click
    def _(_):
        gui_frame.value = (gui_frame.value + 1) % len(frame_ids)

    @gui_prev.on_click
    def _(_):
        gui_frame.value = (gui_frame.value - 1 + len(frame_ids)) % len(frame_ids)

    @gui_pause.on_click
    def _(_):
        gui_play.value = False
        _update_frame(int(gui_frame.value))

    @gui_play.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_frame.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    for handle in [
        gui_point_size,
        gui_show_cam,
        gui_cam_size,
        gui_follow_cam,
        gui_follow_distance,
        gui_follow_lag,
    ]:
        @handle.on_update
        def _(_):
            _update_frame(int(gui_frame.value))

    for handle in [
        gui_cam_accumulate,
        gui_cam_stride,
        gui_cam_window,
        gui_fov_scale,
        gui_play_max_points,
        gui_depth_occlusion,
        gui_depth_tol_rel,
        gui_depth_tol_abs,
    ]:
        @handle.on_update
        def _(_):
            _clear_visibility_cache()
            _update_frame(int(gui_frame.value))

    @gui_display_mode.on_update
    def _(_):
        _clear_visibility_cache()
        _update_frame(int(gui_frame.value))

    @gui_start_video.on_click
    def _(_):
        _close_capture_writer()
        capture_state["recording"] = True
        capture_state["frame_idx"] = 0
        _set_capture_status("recording started")
        try:
            record_queue.put_nowait(_capture_frame_image())
        except queue.Full:
            pass

    @gui_stop_video.on_click
    def _(_):
        capture_state["recording"] = False
        record_queue.join()
        saved_target = capture_state.get("record_dir", None)
        count = int(capture_state.get("frame_idx", 0))
        _close_capture_writer()
        _set_capture_status(f"saved {count} frames: {saved_target}" if saved_target else "stopped")

    worker_thread = threading.Thread(target=_record_worker, daemon=True)
    worker_thread.start()
    _update_frame(0)

    def _cleanup_loaded_files() -> None:
        if not cleanup_paths:
            return
        # The Viser scene now owns the point cloud and poses; keep lazy UI assets in memory too.
        for fid in frame_ids:
            _get_rgb_preview(fid)
            _get_depth_preview(fid)
            _get_depth_map(fid)
        for cleanup_path in cleanup_paths:
            if not cleanup_path:
                continue
            path = os.path.abspath(os.path.expanduser(cleanup_path))
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.exists(path):
                    os.remove(path)
            except Exception as exc:
                print(f"[viser] warning: failed to cleanup {path}: {exc}", flush=True)

    _cleanup_loaded_files()

    def _play_loop() -> None:
        prev_time = time.time()
        while True:
            if gui_play.value:
                now = time.time()
                if now - prev_time >= 1.0 / max(float(gui_fps.value), 1e-6):
                    gui_frame.value = (gui_frame.value + 1) % len(frame_ids)
                    prev_time = now
            time.sleep(0.005)

    if background_mode:
        threading.Thread(target=_play_loop, daemon=True).start()
        print(f"Simple Viser server running in background on port {port}")
        return _share_url
    else:
        print(f"Simple Viser server running in foreground on port {port}. Press Ctrl+C to stop.")
        try:
            _play_loop()
        except KeyboardInterrupt:
            record_stop.set()
            _close_capture_writer()
            print("\n(viser) Stopped by user.")
        return None


def _visualize_sequence_dir(
    sequence_dir: str,
    pose_file: str,
    gt_pose_file: Optional[str],
    pose_convention: str,
    pose_layout: str,
    points_dir: Optional[str],
    poses_dir: Optional[str],
    rgb_dir: Optional[str],
    intri_file: Optional[str],
    port: int,
    share: bool,
    background_mode: bool,
    point_size: float,
    ply_stride: int,
    video_width: int,
    max_cache_frames: int,
    load_from: Optional[int],
    load_to: Optional[int],
    auto_load_cloud: bool = False,
) -> None:
    points_frames_path, points_full_path, poses_path, rgb_path = _discover_sequence_paths(sequence_dir, points_dir, poses_dir, rgb_dir)
    loop_points_path = os.path.join(sequence_dir, "points", "full_lc.ply")
    if os.path.isfile(loop_points_path):
        points_full_path = loop_points_path
    pose_path = os.path.join(poses_path, pose_file)
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    if intri_file is None:
        intri_guess = pose_file.replace("abs_pose", "intri")
        intri_path = os.path.join(poses_path, intri_guess)
        if not os.path.isfile(intri_path):
            intri_path = os.path.join(poses_path, "intri.txt")
    else:
        intri_path = os.path.join(poses_path, intri_file)
    if not os.path.isfile(intri_path):
        intri_path = None

    pose_by_id = _parse_pose_file(pose_path, pose_convention=pose_convention, pose_layout=pose_layout)
    gt_by_id = {}
    gt_candidate_paths: List[str] = []
    if gt_pose_file:
        gt_candidate_paths.append(os.path.join(poses_path, gt_pose_file))
    else:
        for name in ("gt_pose.txt", "abs_pose_gt.txt", "gt.txt"):
            gt_candidate_paths.append(os.path.join(poses_path, name))
    for gt_path in gt_candidate_paths:
        if not os.path.isfile(gt_path):
            continue
        try:
            gt_by_id = _parse_pose_file(gt_path, pose_convention=pose_convention, pose_layout=pose_layout)
            print(f"[gt] loaded: {gt_path}")
            break
        except Exception as exc:
            print(f"[gt] failed to parse {gt_path}: {exc}")
    intri_by_id = _parse_intri_file(intri_path) if intri_path else {}
    gt_status = gt_pose_file if gt_pose_file else ("auto-found" if gt_by_id else "N/A")

    ply_by_id = _discover_frame_plys(points_frames_path)

    rgb_by_id: Dict[int, str] = {}
    if rgb_path is not None:
        for p in Path(rgb_path).glob("*.png"):
            fid = _extract_frame_id(str(p))
            if fid is not None:
                rgb_by_id[fid] = str(p)
    depth_by_id: Dict[int, str] = {}
    depth_dir = os.path.join(sequence_dir, "depth", "dpt")
    if os.path.isdir(depth_dir):
        for p in Path(depth_dir).glob("*.npy"):
            fid = _extract_frame_id(str(p))
            if fid is not None:
                depth_by_id[fid] = str(p)

    frame_ids = sorted(pose_by_id.keys())
    if not frame_ids:
        raise ValueError("No frame ids found in pose file.")
    pred_xyz_all = np.array([pose_by_id[k][:3, 3] for k in frame_ids], dtype=np.float32)
    if gt_by_id:
        gt_xyz_all = np.array([gt_by_id[k][:3, 3] if k in gt_by_id else pose_by_id[k][:3, 3] for k in frame_ids], dtype=np.float32)
    else:
        gt_xyz_all = pred_xyz_all

    server = viser.ViserServer(host="0.0.0.0", port=port)
    if share:
        server.request_share_url()
    server.scene.set_up_direction("-y")
    with server.gui.add_folder("Left Views Panel", order=0, expand_by_default=True):
        preview_img = np.zeros((64, 64, 3), dtype=np.uint8)
        preview_handle = server.gui.add_image(preview_img, format="jpeg", label="RGB")
        depth_handle = server.gui.add_image(preview_img, format="jpeg", label="Depth")
        pose_plot_handle = server.gui.add_image(
            np.full((_POSE_PLOT_SIZE, _POSE_PLOT_SIZE, 3), 255, dtype=np.uint8),
            format="jpeg",
            label="Pose Plot",
        )

    with server.gui.add_folder("Playback", order=10):
        gui_play = server.gui.add_checkbox("Playing", False)
        gui_pause = server.gui.add_button("Pause")
        gui_frame = server.gui.add_slider("Frame", 0, len(frame_ids) - 1, 1, 0)
        gui_prev = server.gui.add_button("Prev")
        gui_next = server.gui.add_button("Next")
        gui_fps = server.gui.add_slider("FPS", 1, 60, 0.1, 20)

    with server.gui.add_folder("Cloud Build", order=20):
        gui_frustum_cull = server.gui.add_checkbox("Frustum Cull Frame Clouds", True)
        gui_frustum_near = server.gui.add_slider("Frustum Near Dist", 0.0, 20.0, 0.1, 0.0)
        gui_frustum_far = server.gui.add_slider("Frustum Far Dist (0=off)", 0.0, 300.0, 0.5, 0.0)
        gui_ply_stride = server.gui.add_slider("PLY stride", 1, 32, 1, max(1, ply_stride))
        gui_global_max_points = server.gui.add_slider("Global Max Points", 50000, 5000000, 5000, 2000000)
        gui_global_voxel = server.gui.add_slider("Global Voxel Size", 0.0, 2.0, 0.001, 0.0)
        gui_apply_cloud = server.gui.add_button("Load / Rebuild Cloud")
        gui_clear_cloud = server.gui.add_button("Clear Cloud")
        gui_cloud_status = server.gui.add_markdown("Cloud: not loaded")

    with server.gui.add_folder("Visualization", order=30):
        gui_point_size = server.gui.add_slider("Point Size", 0.0001, 5.0, 0.0001, point_size)
        gui_cloud_opacity = server.gui.add_slider("Cloud Opacity", 0.0, 1.0, 0.01, 1.0)
        gui_cloud_grayscale = server.gui.add_slider("Cloud Grayscale", 0.0, 1.0, 0.01, 0.0)
        gui_cloud_brightness = server.gui.add_slider("Cloud Brightness", 0.0, 2.0, 0.01, 1.0)
        gui_cloud_saturation = server.gui.add_slider("Cloud Saturation", 0.0, 2.0, 0.01, 1.0)
        gui_accumulate = server.gui.add_checkbox("Accumulate Previous", False)
        gui_accum_window = server.gui.add_slider("Accum Window (0=all)", 0, len(frame_ids) - 1, 1, min(120, len(frame_ids) - 1))
        gui_show_cam = server.gui.add_checkbox("Show Camera Frustum", True)
        gui_cam_size = server.gui.add_slider("Camera Size", 0.01, 5.0, 0.01, 0.04)
        gui_cam_stride = server.gui.add_slider("Camera Stride", 1, 64, 1, 1)
        gui_cam_accumulate = server.gui.add_checkbox("Accumulate Cameras", True)
        gui_cam_window = server.gui.add_slider("Camera Window (0=all)", 0, len(frame_ids) - 1, 1, 0)
        gui_follow_cam = server.gui.add_checkbox("Follow Camera", False)
        gui_follow_distance = server.gui.add_slider("Follow Distance", 0.0, 200.0, 0.1, 1.5)
        gui_follow_lag = server.gui.add_slider("Follow Lag (frames)", 0, len(frame_ids) - 1, 1, 0)
        gui_follow_yaw = server.gui.add_slider("Follow Yaw (deg)", -180.0, 180.0, 1.0, 0.0)
        gui_follow_pitch = server.gui.add_slider("Follow Pitch (deg)", -80.0, 80.0, 1.0, 0.0)
        gui_follow_lookahead = server.gui.add_slider("Follow Lookahead", 0.1, 100.0, 0.1, 3.0)
        gui_follow_smooth = server.gui.add_slider("Follow Smooth", 0.0, 0.99, 0.01, 0.2)

    with server.gui.add_folder("Filtering", order=40):
        gui_white_thresh = server.gui.add_slider("White Thresh", 180, 255, 1, 245)
        gui_black_thresh = server.gui.add_slider("Black Thresh", 0, 80, 1, 8)
        gui_sky_blue_strength = server.gui.add_slider("SkyBlue Strength", 0.0, 3.0, 0.05, 1.0)
        gui_sky_blue_min_b = server.gui.add_slider("SkyBlue B Min", 80, 255, 1, 160)
        gui_sky_blue_max_r = server.gui.add_slider("SkyBlue R Max", 50, 255, 1, 190)
        gui_sky_blue_min_g = server.gui.add_slider("SkyBlue G Min", 50, 255, 1, 120)
        gui_sky_blue_max_rg_diff = server.gui.add_slider("SkyBlue |R-G| Max", 0, 140, 1, 95)
        gui_sky_blue_min_bg_diff = server.gui.add_slider("SkyBlue B-G Min", 0, 120, 1, 8)
        gui_sky_blue_min_br_diff = server.gui.add_slider("SkyBlue B-R Min", 0, 200, 1, 20)

    with server.gui.add_folder("Session", order=50):
        gui_reload = server.gui.add_button("Reload Sequence")
        gui_reload_current = server.gui.add_button("Reload Current Dir")
        gui_browse_seq = server.gui.add_button("Browse Sequence Dir")
        gui_seq_dir_input = None
        if hasattr(server.gui, "add_text"):
            try:
                gui_seq_dir_input = server.gui.add_text("Sequence Dir", initial_value=sequence_dir)
            except Exception:
                gui_seq_dir_input = None
        if gui_seq_dir_input is None and hasattr(server.gui, "add_text_input"):
            try:
                gui_seq_dir_input = server.gui.add_text_input("Sequence Dir", initial_value=sequence_dir)
            except Exception:
                gui_seq_dir_input = None
        if gui_seq_dir_input is None:
            print("Session dir input not available in this viser version; using Reload Current Dir button only.")

    with server.gui.add_folder("Capture", order=55):
        default_capture_dir = os.path.join(sequence_dir, "captures")
        gui_capture_dir = server.gui.add_text("Capture Dir", initial_value=default_capture_dir)
        gui_capture_prefix = server.gui.add_text("Capture Prefix", initial_value="viser_capture")
        gui_scene_w = server.gui.add_slider("Scene Capture W", 320, 4096, 1, 1280)
        gui_scene_h = server.gui.add_slider("Scene Capture H", 240, 4096, 1, 720)
        gui_scene_bg_tol = server.gui.add_slider("Scene BG Key Tol", 0, 64, 1, 6)
        gui_scene_transparent = server.gui.add_checkbox("Scene Transparent BG", True)
        gui_scene_snapshot = server.gui.add_button("Save Scene PNG")
        gui_capture_fps = server.gui.add_slider("Capture FPS", 1, 60, 1, 20)
        gui_capture_mode = server.gui.add_dropdown(
            "Capture Mode",
            options=("mp4", "png_seq"),
            initial_value="mp4",
        )
        gui_snapshot = server.gui.add_button("Save Snapshot")
        gui_record = server.gui.add_checkbox("Record Panels", False)
        gui_capture_status = server.gui.add_markdown("Capture: idle")

    pcd_handle = server.scene.add_point_cloud(
        "/sequence/pc_global",
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint8),
        point_size=point_size,
        visible=False,
    )
    pcd_frame_handle = server.scene.add_point_cloud(
        "/sequence/pc_frame_current",
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint8),
        point_size=point_size,
        visible=False,
    )
    pcd_accum_handle = server.scene.add_point_cloud(
        "/sequence/pc_frame_accum",
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint8),
        point_size=point_size,
        visible=False,
    )
    current_frustum = server.scene.add_camera_frustum(
        "/sequence/cam",
        fov=1.047,
        aspect=1.0,
        scale=gui_cam_size.value,
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        color=(1.0, 1.0, 1.0),
    )
    history_frustums = []
    for idx, fid in enumerate(frame_ids):
        pose = pose_by_id[fid]
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        wxyz = vt.SO3.from_matrix(rot).wxyz
        hue = 0.0 if len(frame_ids) <= 1 else idx / float(len(frame_ids) - 1)
        color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        hist_intri = intri_by_id.get(fid, None)
        hist_fov, hist_aspect = _compute_fov_aspect(hist_intri, preview_img)
        history_frustums.append(
            server.scene.add_camera_frustum(
                f"/sequence/cam_hist/{idx}",
                fov=hist_fov,
                aspect=hist_aspect,
                scale=gui_cam_size.value,
                wxyz=wxyz,
                position=trans,
                color=color,
                line_width=1.2,
                visible=False,
            )
        )

    cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
    global_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
    cache_order: List[Tuple[Any, ...]] = []
    max_cache = max(1, int(max_cache_frames))
    follow_state: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    frame_cloud_cache_raw: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    frame_cloud_cache_styled: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    selected_cloud_indices: List[int] = []
    accum_cache_state: Dict[str, Any] = {"last_idx": None, "window": None, "chunks": []}
    # Incremental camera visibility diff — only send WebSocket messages for changed handles.
    prev_cam_visible_set: set = set()
    cloud_loaded = False
    cloud_dirty = True
    global_raw_points = np.zeros((0, 3), dtype=np.float32)
    global_raw_colors = np.zeros((0, 3), dtype=np.uint8)
    frustum_cloud_cache_raw: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    frustum_cloud_cache_styled: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    # Per-frame viser handles for accumulation: data uploaded ONCE, only visibility toggled during play
    frame_accum_handles: Dict[int, Any] = {}
    prev_accum_visible_set: List[int] = []
    rgb_preview_cache: Dict[int, np.ndarray] = {}
    depth_preview_cache: Dict[int, np.ndarray] = {}
    cloud_style_dirty = False
    pause_view_active = False
    capture_state: Dict[str, Any] = {
        "last_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "last_depth": np.zeros((64, 64, 3), dtype=np.uint8),
        "last_pose": np.zeros((64, 64, 3), dtype=np.uint8),
        "writer": None,
        "mode": None,
        "record_dir": None,
        "frame_idx": 0,
        "frame_shape": None,
    }
    record_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
    record_stop = threading.Event()

    def _filter_cfg() -> Tuple[int, int, float, int, int, int, int, int, int]:
        return (
            int(gui_white_thresh.value),
            int(gui_black_thresh.value),
            float(gui_sky_blue_strength.value),
            int(gui_sky_blue_min_b.value),
            int(gui_sky_blue_max_r.value),
            int(gui_sky_blue_min_g.value),
            int(gui_sky_blue_max_rg_diff.value),
            int(gui_sky_blue_min_bg_diff.value),
            int(gui_sky_blue_min_br_diff.value),
        )

    def _apply_sky_strength(fcfg: Tuple[int, int, float, int, int, int, int, int, int]) -> Tuple[int, int, int, int, int, int, int, int]:
        white_thr, black_thr, strength, bmin, rmax, gmin, rgmax, bgmin, brmin = fcfg
        s = max(0.0, float(strength))
        bmin_eff = int(np.clip(round(bmin - 20.0 * (s - 1.0)), 80, 255))
        rmax_eff = int(np.clip(round(rmax + 35.0 * (s - 1.0)), 40, 255))
        gmin_eff = int(np.clip(round(gmin - 20.0 * (s - 1.0)), 40, 255))
        rgmax_eff = int(np.clip(round(rgmax + 40.0 * (s - 1.0)), 0, 180))
        bgmin_eff = int(np.clip(round(bgmin - 8.0 * (s - 1.0)), 0, 120))
        brmin_eff = int(np.clip(round(brmin - 16.0 * (s - 1.0)), 0, 240))
        return white_thr, black_thr, bmin_eff, rmax_eff, gmin_eff, rgmax_eff, bgmin_eff, brmin_eff

    def _dist_cfg() -> Tuple[float, float]:
        return (0.0, 0.0)

    def _get_points_colors(fid: int, stride_val: int, frame_max_points: int, frame_voxel: float) -> Tuple[np.ndarray, np.ndarray]:
        if fid not in ply_by_id:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        fcfg = _filter_cfg()
        eff = _apply_sky_strength(fcfg)
        dcfg = _dist_cfg()
        key = (fid, stride_val, frame_max_points, round(frame_voxel, 6), round(dcfg[0], 6), round(dcfg[1], 6), *fcfg)
        if key in cache:
            return cache[key]
        pts, cols = _read_ply_points_colors(ply_by_id[fid], stride_val)
        pts, cols = _filter_white_points(
            pts,
            cols,
            white_threshold=eff[0],
            bright_channel_threshold=200,
            black_threshold=eff[1],
            sky_blue_min_b=eff[2],
            sky_blue_max_r=eff[3],
            sky_blue_min_g=eff[4],
            sky_blue_max_rg_diff=eff[5],
            sky_blue_min_bg_diff=eff[6],
            sky_blue_min_br_diff=eff[7],
        )
        cam_center = pose_by_id[fid][:3, 3]
        pts, cols = _filter_frame_distance(pts, cols, cam_center, dcfg[0], dcfg[1])
        pts, cols = _voxel_downsample(pts, cols, frame_voxel)
        pts, cols = _limit_points(pts, cols, frame_max_points)
        cache[key] = (pts, cols)
        cache_order.append(key)
        while len(cache_order) > max_cache:
            old = cache_order.pop(0)
            if old in cache:
                del cache[old]
        return pts, cols

    def _build_global_cloud(
        stride_val: int,
        frame_max_points: int,
        frame_voxel: float,
        global_max_points: int,
        global_voxel: float,
        selected_indices: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fcfg = _filter_cfg()
        dcfg = _dist_cfg()
        selected_key = selected_indices if selected_indices is not None else tuple(range(len(frame_ids)))
        key = (
            stride_val,
            frame_max_points,
            round(frame_voxel, 6),
            round(dcfg[0], 6),
            round(dcfg[1], 6),
            global_max_points,
            round(global_voxel, 6),
            selected_key,
            *fcfg,
        )
        if key in global_cache:
            return global_cache[key]
        pts_list: List[np.ndarray] = []
        cols_list: List[np.ndarray] = []
        idx_iter = selected_indices if selected_indices is not None else tuple(range(len(frame_ids)))
        for idx in idx_iter:
            fid = frame_ids[int(idx)]
            pts, cols = _get_points_colors(fid, stride_val, frame_max_points, frame_voxel)
            pts_list.append(pts)
            cols_list.append(cols)
        if not pts_list:
            out = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
            global_cache[key] = out
            return out
        pts = np.concatenate(pts_list, axis=0)
        cols = np.concatenate(cols_list, axis=0)
        pts, cols = _voxel_downsample(pts, cols, global_voxel)
        pts, cols = _limit_points(pts, cols, global_max_points)
        global_cache[key] = (pts, cols)
        return pts, cols

    def _set_cloud_status(msg: str) -> None:
        gui_cloud_status.content = f"Cloud: {msg}"

    def _style_cloud_colors(colors: np.ndarray) -> np.ndarray:
        if colors.shape[0] == 0:
            return colors
        c = colors.astype(np.float32) / 255.0
        lum = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2])[:, None]
        gray_mix = float(gui_cloud_grayscale.value)
        sat = float(gui_cloud_saturation.value)
        bri = float(gui_cloud_brightness.value)
        op = float(gui_cloud_opacity.value)
        c = (1.0 - gray_mix) * c + gray_mix * lum
        c = lum + sat * (c - lum)
        c = np.clip(c * bri * op, 0.0, 1.0)
        return (c * 255.0).astype(np.uint8)

    def _selected_frame_indices() -> List[int]:
        return list(range(len(frame_ids)))

    def _set_capture_status(msg: str) -> None:
        gui_capture_status.content = f"Capture: {msg}"

    def _mark_cloud_style_dirty() -> None:
        nonlocal cloud_style_dirty
        cloud_style_dirty = True

    def _clear_accum_cache() -> None:
        accum_cache_state.clear()
        accum_cache_state.update(
            {
                "start": None,
                "end": None,
                "window": None,
                "points": np.zeros((0, 3), dtype=np.float32),
                "colors": np.zeros((0, 3), dtype=np.uint8),
            }
        )

    def _ensure_frame_accum_handle(frame_idx: int) -> None:
        """Create a per-frame point cloud handle for accumulation (data sent once)."""
        if frame_idx in frame_accum_handles:
            return
        pts, cols = _get_global_frustum_cloud(frame_idx)
        handle = server.scene.add_point_cloud(
            f"/sequence/pc_accum/{frame_idx}",
            pts, cols,
            point_size=gui_point_size.value,
            visible=False,
        )
        frame_accum_handles[frame_idx] = handle

    def _clear_frame_accum_handles() -> None:
        """Remove all per-frame accumulation handles and reset visibility tracking."""
        nonlocal prev_accum_visible_set
        for h in frame_accum_handles.values():
            try:
                h.remove()
            except Exception:
                pass
        frame_accum_handles.clear()
        prev_accum_visible_set = []

    def _close_capture_writer() -> None:
        writer = capture_state.get("writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        capture_state["writer"] = None
        capture_state["mode"] = None
        capture_state["record_dir"] = None
        capture_state["frame_shape"] = None

    def _ensure_capture_dir() -> str:
        out_dir = os.path.abspath(os.path.expanduser(str(gui_capture_dir.value).strip()))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _capture_frame_image() -> np.ndarray:
        frame = _compose_panel_frame(
            capture_state["last_rgb"],
            capture_state["last_depth"],
            capture_state["last_pose"],
        )
        return _ensure_hwc_rgb_u8(frame)

    def _save_snapshot() -> None:
        out_dir = _ensure_capture_dir()
        prefix = str(gui_capture_prefix.value).strip() or "viser_capture"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"{prefix}_{ts}.png")
        Image.fromarray(_capture_frame_image()).save(path)
        _set_capture_status(f"snapshot saved: {path}")

    def _save_scene_snapshot() -> None:
        clients = server.get_clients()
        if len(clients) == 0:
            _set_capture_status("scene capture failed: no connected client")
            return
        client = clients[sorted(clients.keys())[0]]
        out_dir = _ensure_capture_dir()
        prefix = str(gui_capture_prefix.value).strip() or "viser_capture"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"{prefix}_{ts}_scene.png")

        try:
            img = client.get_render(
                int(gui_scene_h.value),
                int(gui_scene_w.value),
                transport_format="png",
            )
            rgba = _ensure_hwc_rgba_u8(img)
            if bool(gui_scene_transparent.value):
                alpha = rgba[:, :, 3]
                if int(alpha.min()) == 255:
                    # Fallback: key out a uniform background color when PNG alpha is unavailable.
                    rgb = rgba[:, :, :3].astype(np.int16)
                    corners = np.array(
                        [
                            rgb[0, 0],
                            rgb[0, -1],
                            rgb[-1, 0],
                            rgb[-1, -1],
                        ],
                        dtype=np.int16,
                    )
                    bg = np.mean(corners, axis=0, dtype=np.float64)
                    tol = int(gui_scene_bg_tol.value)
                    dist = np.max(np.abs(rgb - bg[None, None, :]), axis=2)
                    mask_bg = dist <= tol
                    rgba[mask_bg, 3] = 0
            Image.fromarray(rgba, mode="RGBA").save(path)
            _set_capture_status(f"scene png saved: {path}")
        except Exception as exc:
            _set_capture_status(f"scene capture failed: {exc}")

    def _append_capture_frame() -> None:
        frame = _capture_frame_image()
        _append_capture_frame_from_frame(frame)

    def _record_worker() -> None:
        while not record_stop.is_set():
            try:
                frame = record_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame.size == 0:
                record_queue.task_done()
                continue
            try:
                _append_capture_frame_from_frame(frame)
            except Exception as exc:
                _set_capture_status(f"record worker error: {exc}")
            finally:
                record_queue.task_done()

    def _append_capture_frame_from_frame(frame: np.ndarray) -> None:
        # Same as _append_capture_frame but using already captured frame.
        out_dir = _ensure_capture_dir()
        prefix = str(gui_capture_prefix.value).strip() or "viser_capture"
        mode = str(gui_capture_mode.value)
        frame = _ensure_hwc_rgb_u8(frame)
        expected_shape = capture_state.get("frame_shape", None)
        if expected_shape is None:
            capture_state["frame_shape"] = frame.shape
        elif tuple(frame.shape) != tuple(expected_shape):
            target_h, target_w = int(expected_shape[0]), int(expected_shape[1])
            frame = np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8)
            frame = _ensure_hwc_rgb_u8(frame)

        if mode == "mp4":
            if capture_state["writer"] is None or capture_state.get("mode") != "mp4":
                try:
                    import imageio.v2 as imageio
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    mp4_path = os.path.join(out_dir, f"{prefix}_{ts}.mp4")
                    capture_state["writer"] = imageio.get_writer(mp4_path, fps=int(gui_capture_fps.value), codec="libx264")
                    capture_state["mode"] = "mp4"
                    capture_state["record_dir"] = mp4_path
                    capture_state["frame_idx"] = 0
                    _set_capture_status(f"recording mp4: {mp4_path}")
                except Exception as exc:
                    _set_capture_status(f"mp4 unavailable ({exc}), fallback to png_seq")
                    mode = "png_seq"
            if mode == "mp4" and capture_state["writer"] is not None:
                try:
                    capture_state["writer"].append_data(frame)
                    capture_state["frame_idx"] += 1
                    return
                except Exception as exc:
                    _set_capture_status(f"mp4 write failed ({exc}), fallback to png_seq")
                    try:
                        capture_state["writer"].close()
                    except Exception:
                        pass
                    capture_state["writer"] = None
                    capture_state["mode"] = None

        if capture_state.get("mode") != "png_seq":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seq_dir = os.path.join(out_dir, f"{prefix}_{ts}_frames")
            os.makedirs(seq_dir, exist_ok=True)
            capture_state["mode"] = "png_seq"
            capture_state["record_dir"] = seq_dir
            capture_state["frame_idx"] = 0
            _set_capture_status(f"recording png sequence: {seq_dir}")
        seq_dir = str(capture_state["record_dir"])
        idx = int(capture_state["frame_idx"])
        frame_path = os.path.join(seq_dir, f"frame_{idx:06d}.png")
        Image.fromarray(frame).save(frame_path)
        capture_state["frame_idx"] = idx + 1

    def _mark_cloud_dirty() -> None:
        nonlocal cloud_dirty
        cloud_dirty = True

    def _clear_cloud_handles() -> None:
        nonlocal cloud_loaded, global_raw_points, global_raw_colors
        frame_cloud_cache_raw.clear()
        frame_cloud_cache_styled.clear()
        frustum_cloud_cache_raw.clear()
        frustum_cloud_cache_styled.clear()
        selected_cloud_indices.clear()
        _clear_accum_cache()
        _clear_frame_accum_handles()
        pcd_handle.points = np.zeros((0, 3), np.float32)
        pcd_handle.colors = np.zeros((0, 3), np.uint8)
        pcd_frame_handle.points = np.zeros((0, 3), np.float32)
        pcd_frame_handle.colors = np.zeros((0, 3), np.uint8)
        pcd_accum_handle.points = np.zeros((0, 3), np.float32)
        pcd_accum_handle.colors = np.zeros((0, 3), np.uint8)
        pcd_handle.visible = False
        pcd_frame_handle.visible = False
        pcd_accum_handle.visible = False
        global_raw_points = np.zeros((0, 3), dtype=np.float32)
        global_raw_colors = np.zeros((0, 3), dtype=np.uint8)
        cloud_loaded = False

    def _apply_cloud_style_to_loaded_clouds() -> None:
        nonlocal cloud_style_dirty
        pcd_handle.point_size = gui_point_size.value
        pcd_frame_handle.point_size = gui_point_size.value
        pcd_accum_handle.point_size = gui_point_size.value
        for h in frame_accum_handles.values():
            h.point_size = gui_point_size.value
        if global_raw_colors.shape[0] > 0:
            pcd_handle.colors = _style_cloud_colors(global_raw_colors)
        frame_cloud_cache_styled.clear()
        for idx, (pts, raw_cols) in frame_cloud_cache_raw.items():
            frame_cloud_cache_styled[idx] = (pts, _style_cloud_colors(raw_cols))
        frustum_cloud_cache_styled.clear()
        for idx, (pts, raw_cols) in frustum_cloud_cache_raw.items():
            frustum_cloud_cache_styled[idx] = (pts, _style_cloud_colors(raw_cols))
        _clear_accum_cache()
        _clear_frame_accum_handles()  # Style changed, rebuild handles with new colors
        cloud_style_dirty = False

    def _get_global_frustum_cloud(frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if frame_idx in frustum_cloud_cache_styled:
            return frustum_cloud_cache_styled[frame_idx]
        if global_raw_points.shape[0] == 0:
            out = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
            frustum_cloud_cache_raw[frame_idx] = out
            frustum_cloud_cache_styled[frame_idx] = out
            return out

        fid = frame_ids[frame_idx]
        if bool(gui_frustum_cull.value):
            pts, raw_cols = _filter_points_by_camera_frustum(
                global_raw_points,
                global_raw_colors,
                pose_c2w=pose_by_id[fid],
                intri=intri_by_id.get(fid, None),
                near_dist=float(gui_frustum_near.value),
                far_dist=float(gui_frustum_far.value),
            )
        else:
            pts, raw_cols = global_raw_points, global_raw_colors
        out_raw = (pts, raw_cols)
        out_styled = (pts, _style_cloud_colors(raw_cols))
        frustum_cloud_cache_raw[frame_idx] = out_raw
        frustum_cloud_cache_styled[frame_idx] = out_styled
        return out_styled

    def _get_accumulated_global_frustum_cloud(frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        window = int(gui_accum_window.value)
        if window > 0:
            start_idx = max(0, frame_idx - window + 1)
        else:
            start_idx = min(selected_cloud_indices) if selected_cloud_indices else 0
        picked = [i for i in selected_cloud_indices if start_idx <= i <= frame_idx]
        if not picked:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        start = picked[0]
        end = picked[-1]
        cached_start = accum_cache_state.get("start", None)
        cached_end = accum_cache_state.get("end", None)
        cached_window = accum_cache_state.get("window", None)
        if cached_start == start and cached_end == end and cached_window == window:
            return accum_cache_state["points"], accum_cache_state["colors"]

        can_append = (
            window == 0
            and cached_window == 0
            and cached_start == start
            and cached_end is not None
            and int(cached_end) < end
        )
        if can_append:
            new_chunks = [_get_global_frustum_cloud(i) for i in picked if int(cached_end) < i <= end]
            old_pts = accum_cache_state["points"]
            old_cols = accum_cache_state["colors"]
            if new_chunks:
                pts = np.concatenate([old_pts, *[x[0] for x in new_chunks]], axis=0)
                cols = np.concatenate([old_cols, *[x[1] for x in new_chunks]], axis=0)
            else:
                pts, cols = old_pts, old_cols
        else:
            chunks = [_get_global_frustum_cloud(i) for i in picked]
            pts = np.concatenate([x[0] for x in chunks], axis=0)
            cols = np.concatenate([x[1] for x in chunks], axis=0)

        accum_cache_state["start"] = start
        accum_cache_state["end"] = end
        accum_cache_state["window"] = window
        accum_cache_state["points"] = pts
        accum_cache_state["colors"] = cols
        return pts, cols

    def _get_rgb_preview(fid: int) -> np.ndarray:
        if fid in rgb_preview_cache:
            return rgb_preview_cache[fid]
        rgb_file = rgb_by_id.get(fid, None)
        if rgb_file:
            preview = _read_image_preview(rgb_file, video_width)
        else:
            preview = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb_preview_cache[fid] = preview
        return preview

    def _get_depth_preview(fid: int) -> np.ndarray:
        if fid in depth_preview_cache:
            return depth_preview_cache[fid]
        depth_file = depth_by_id.get(fid, None)
        if depth_file and os.path.isfile(depth_file):
            try:
                depth_arr = np.load(depth_file)
                depth_img = _depth_to_rgb(depth_arr, out_width=video_width)
            except Exception:
                depth_img = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            depth_img = np.zeros((64, 64, 3), dtype=np.uint8)
        depth_preview_cache[fid] = depth_img
        return depth_img

    def _update_cloud_visibility(frame_idx: int) -> None:
        nonlocal prev_accum_visible_set
        if not cloud_loaded:
            pcd_handle.visible = False
            pcd_frame_handle.visible = False
            pcd_accum_handle.visible = False
            for i in prev_accum_visible_set:
                if i in frame_accum_handles:
                    frame_accum_handles[i].visible = False
            prev_accum_visible_set = []
            return
        if not bool(gui_play.value) and not pause_view_active:
            pcd_handle.visible = pcd_handle.points.shape[0] > 0
            pcd_frame_handle.visible = False
            pcd_accum_handle.visible = False
            for i in prev_accum_visible_set:
                if i in frame_accum_handles:
                    frame_accum_handles[i].visible = False
            prev_accum_visible_set = []
            return
        # Playback/pause: render frustum-culled slice (current or accumulated).
        pcd_handle.visible = False
        pcd_accum_handle.visible = False
        if bool(gui_accumulate.value):
            # Per-frame handles: data uploaded once, only visibility toggled.
            window = int(gui_accum_window.value)
            if window > 0:
                start_idx = max(0, frame_idx - window + 1)
            else:
                start_idx = min(selected_cloud_indices) if selected_cloud_indices else 0
            new_visible = [i for i in selected_cloud_indices if start_idx <= i <= frame_idx]
            new_visible_set = set(new_visible)
            prev_set = set(prev_accum_visible_set)
            to_hide = prev_set - new_visible_set
            to_show = new_visible_set - prev_set
            for i in to_hide:
                if i in frame_accum_handles:
                    frame_accum_handles[i].visible = False
            for i in to_show:
                _ensure_frame_accum_handle(i)
                frame_accum_handles[i].visible = True
            prev_accum_visible_set = new_visible
            pcd_frame_handle.visible = False
        else:
            # Single frame: show only current frustum slice.
            for i in prev_accum_visible_set:
                if i in frame_accum_handles:
                    frame_accum_handles[i].visible = False
            prev_accum_visible_set = []
            pts, cols = _get_global_frustum_cloud(frame_idx)
            pcd_frame_handle.points = pts
            pcd_frame_handle.colors = cols
            pcd_frame_handle.visible = pts.shape[0] > 0

    def _build_cloud_assets() -> None:
        nonlocal cloud_loaded, cloud_dirty, global_raw_points, global_raw_colors
        _clear_cloud_handles()
        stride_val = int(gui_ply_stride.value)
        selected_indices = _selected_frame_indices()
        selected_cloud_indices.clear()
        selected_cloud_indices.extend(selected_indices)
        if points_full_path is None:
            _set_cloud_status("points/full.ply not found; run inference with point saving enabled")
            cloud_loaded = False
            return
        pts, cols = _read_ply_points_colors(
            points_full_path,
            stride_val,
            max_points=int(gui_global_max_points.value),
        )
        fcfg = _filter_cfg()
        eff = _apply_sky_strength(fcfg)
        pts, cols = _filter_white_points(
            pts,
            cols,
            white_threshold=eff[0],
            bright_channel_threshold=200,
            black_threshold=eff[1],
            sky_blue_min_b=eff[2],
            sky_blue_max_r=eff[3],
            sky_blue_min_g=eff[4],
            sky_blue_max_rg_diff=eff[5],
            sky_blue_min_bg_diff=eff[6],
            sky_blue_min_br_diff=eff[7],
        )
        pts, cols = _voxel_downsample(pts, cols, float(gui_global_voxel.value))
        pcd_handle.points = pts
        global_raw_points = pts
        global_raw_colors = cols
        pcd_handle.colors = _style_cloud_colors(cols)
        built_any = pts.shape[0] > 0
        cloud_loaded = built_any
        cloud_dirty = False
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))
        global_n = int(pcd_handle.points.shape[0])
        _set_cloud_status(
            f"loaded points/full.ply (global={global_n})"
        )

    def _update_camera_visibility(frame_idx: int) -> None:
        nonlocal prev_cam_visible_set
        cam_stride = max(1, int(gui_cam_stride.value))
        n = len(history_frustums)

        if not gui_show_cam.value:
            new_visible: set = set()
            want_current = False
        elif not gui_play.value:
            # Static view: show all strided cameras.
            new_visible = set(range(0, n, cam_stride))
            want_current = True
        elif gui_cam_accumulate.value:
            start = max(0, frame_idx - int(gui_cam_window.value) + 1) if gui_cam_window.value > 0 else 0
            new_visible = {i for i in range(start, min(frame_idx + 1, n)) if i % cam_stride == 0}
            want_current = True
        else:
            snapped = (frame_idx // cam_stride) * cam_stride
            new_visible = {snapped} if snapped < n else set()
            want_current = True

        # Only send WebSocket messages for handles whose state actually changed.
        for i in prev_cam_visible_set - new_visible:
            if i < n:
                history_frustums[i].visible = False
        for i in new_visible - prev_cam_visible_set:
            if i < n:
                history_frustums[i].visible = True
        prev_cam_visible_set = new_visible

        current_frustum.visible = want_current

    def _apply_follow_camera(frame_idx: int) -> None:
        if not gui_follow_cam.value:
            follow_state.clear()
            return
        lag = int(gui_follow_lag.value)
        follow_idx = max(0, frame_idx - lag)
        follow_fid = frame_ids[follow_idx]
        follow_pose = pose_by_id[follow_fid]
        rot = follow_pose[:3, :3]
        trans = follow_pose[:3, 3]

        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        direction = forward.copy()
        # Yaw first in world-up frame.
        direction = _rotate_vec_axis(direction, world_up, math.radians(float(gui_follow_yaw.value)))
        # Then pitch around the current right axis so pitch is true up/down tilt.
        right = np.cross(direction, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        direction = _rotate_vec_axis(direction, right, math.radians(float(gui_follow_pitch.value)))
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        cam_pos_target = trans - direction * float(gui_follow_distance.value)
        # Lookahead follows raw camera forward, making this control clearly observable.
        look_at_target = trans + forward * float(gui_follow_lookahead.value)
        alpha = float(gui_follow_smooth.value)

        for client_id, client in server.get_clients().items():
            prev = follow_state.get(client_id, None)
            if prev is None:
                cam_pos = cam_pos_target
                look_at = look_at_target
            else:
                cam_pos = (1.0 - alpha) * cam_pos_target + alpha * prev[0]
                look_at = (1.0 - alpha) * look_at_target + alpha * prev[1]
            follow_state[client_id] = (cam_pos, look_at)
            client.camera.position = cam_pos
            client.camera.look_at = look_at
            client.camera.up_direction = (0.0, -1.0, 0.0)

    def _update_frame(frame_idx: int) -> None:
        fid = frame_ids[frame_idx]
        _update_cloud_visibility(frame_idx)
        if cloud_style_dirty:
            _apply_cloud_style_to_loaded_clouds()

        preview = _get_rgb_preview(fid)
        preview_handle.image = preview
        capture_state["last_rgb"] = preview

        depth_img = _get_depth_preview(fid)
        depth_handle.image = depth_img
        capture_state["last_depth"] = depth_img

        pose = pose_by_id[fid]
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        wxyz = vt.SO3.from_matrix(rot).wxyz
        intri = intri_by_id.get(fid, None)
        fov, aspect = _compute_fov_aspect(intri, preview)

        hue = 0.0 if len(frame_ids) <= 1 else frame_idx / float(len(frame_ids) - 1)
        current_frustum.color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        current_frustum.wxyz = wxyz
        current_frustum.position = trans
        current_frustum.scale = gui_cam_size.value
        current_frustum.fov = fov
        current_frustum.aspect = aspect
        _update_camera_visibility(frame_idx)
        _apply_follow_camera(frame_idx)

        pose_img = _render_pose_plot(
            pred_xyz=pred_xyz_all,
            gt_xyz=gt_xyz_all,
            current_idx=frame_idx,
            playing=bool(gui_play.value),
            width=_POSE_PLOT_SIZE,
            height=_POSE_PLOT_SIZE,
        )
        pose_plot_handle.image = pose_img
        capture_state["last_pose"] = pose_img

        if bool(gui_record.value):
            try:
                frame = _capture_frame_image()
                try:
                    record_queue.put_nowait(frame)
                except queue.Full:
                    # Drop frame to keep interactive playback responsive.
                    pass
            except Exception as exc:
                gui_record.value = False
                _set_capture_status(f"recording stopped by error: {exc}")

    @gui_next.on_click
    def _(_):
        gui_frame.value = (gui_frame.value + 1) % len(frame_ids)

    @gui_prev.on_click
    def _(_):
        gui_frame.value = (gui_frame.value - 1 + len(frame_ids)) % len(frame_ids)

    @gui_pause.on_click
    def _(_):
        nonlocal pause_view_active
        gui_play.value = False
        pause_view_active = True
        _update_frame(int(gui_frame.value))

    @gui_play.on_update
    def _(_):
        nonlocal pause_view_active
        if bool(gui_play.value):
            pause_view_active = False
        else:
            pause_view_active = False
            _update_frame(int(gui_frame.value))

    @gui_frame.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_point_size.on_update
    def _(_):
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))

    @gui_cloud_opacity.on_update
    def _(_):
        _mark_cloud_style_dirty()
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))

    @gui_cloud_grayscale.on_update
    def _(_):
        _mark_cloud_style_dirty()
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))

    @gui_cloud_brightness.on_update
    def _(_):
        _mark_cloud_style_dirty()
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))

    @gui_cloud_saturation.on_update
    def _(_):
        _mark_cloud_style_dirty()
        _apply_cloud_style_to_loaded_clouds()
        _update_cloud_visibility(int(gui_frame.value))

    def _on_cloud_param_changed() -> None:
        cache.clear()
        global_cache.clear()
        cache_order.clear()
        frustum_cloud_cache_raw.clear()
        frustum_cloud_cache_styled.clear()
        _clear_accum_cache()
        _mark_cloud_dirty()
        _set_cloud_status("params changed, click 'Load / Rebuild Cloud'")

    def _on_frustum_param_changed() -> None:
        frustum_cloud_cache_raw.clear()
        frustum_cloud_cache_styled.clear()
        _clear_accum_cache()
        _clear_frame_accum_handles()
        _update_cloud_visibility(int(gui_frame.value))

    @gui_ply_stride.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_global_max_points.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_global_voxel.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_accumulate.on_update
    def _(_):
        nonlocal prev_accum_visible_set
        _clear_accum_cache()
        if not bool(gui_accumulate.value):
            for i in prev_accum_visible_set:
                if i in frame_accum_handles:
                    frame_accum_handles[i].visible = False
            prev_accum_visible_set = []
        _update_frame(int(gui_frame.value))

    @gui_accum_window.on_update
    def _(_):
        _clear_accum_cache()
        _update_frame(int(gui_frame.value))

    @gui_frustum_cull.on_update
    def _(_):
        _on_frustum_param_changed()

    @gui_frustum_near.on_update
    def _(_):
        _on_frustum_param_changed()

    @gui_frustum_far.on_update
    def _(_):
        _on_frustum_param_changed()

    @gui_apply_cloud.on_click
    def _(_):
        _build_cloud_assets()

    @gui_clear_cloud.on_click
    def _(_):
        _clear_cloud_handles()
        _set_cloud_status("cleared")
        _update_frame(int(gui_frame.value))

    @gui_snapshot.on_click
    def _(_):
        _save_snapshot()

    @gui_scene_snapshot.on_click
    def _(_):
        _save_scene_snapshot()

    @gui_record.on_update
    def _(_):
        if bool(gui_record.value):
            _close_capture_writer()
            _set_capture_status("recording started")
            try:
                record_queue.put_nowait(_capture_frame_image())
            except queue.Full:
                pass
        else:
            saved_target = capture_state.get("record_dir", None)
            mode = capture_state.get("mode", None)
            count = int(capture_state.get("frame_idx", 0))
            while not record_queue.empty():
                try:
                    record_queue.get_nowait()
                    record_queue.task_done()
                except Exception:
                    break
            _close_capture_writer()
            if saved_target is not None:
                _set_capture_status(f"saved {count} frames ({mode}): {saved_target}")
            else:
                _set_capture_status("recording stopped")

    @gui_show_cam.on_update
    def _(_):
        _update_camera_visibility(int(gui_frame.value))

    @gui_cam_size.on_update
    def _(_):
        current_frustum.scale = gui_cam_size.value
        for h in history_frustums:
            h.scale = gui_cam_size.value
        _update_frame(int(gui_frame.value))

    @gui_cam_stride.on_update
    def _(_):
        _update_camera_visibility(int(gui_frame.value))

    @gui_cam_accumulate.on_update
    def _(_):
        _update_camera_visibility(int(gui_frame.value))

    @gui_cam_window.on_update
    def _(_):
        _update_camera_visibility(int(gui_frame.value))

    @gui_follow_distance.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_lag.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_yaw.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_pitch.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_lookahead.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_smooth.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_follow_cam.on_update
    def _(_):
        _update_frame(int(gui_frame.value))

    @gui_white_thresh.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_black_thresh.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_strength.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_min_b.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_max_r.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_min_g.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_max_rg_diff.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_min_bg_diff.on_update
    def _(_):
        _on_cloud_param_changed()

    @gui_sky_blue_min_br_diff.on_update
    def _(_):
        _on_cloud_param_changed()

    def _restart_with_sequence(target_dir: str) -> None:
        target_dir = os.path.abspath(os.path.expanduser(target_dir))
        if not os.path.isdir(target_dir):
            print(f"[reload] directory not found: {target_dir}")
            return
        record_stop.set()
        _close_capture_writer()
        argv = [
            sys.executable,
            os.path.abspath(__file__),
            "--sequence_dir",
            target_dir,
            "--pose_file",
            pose_file,
            "--pose_convention",
            pose_convention,
            "--pose_layout",
            pose_layout,
            "--port",
            str(port),
            "--point_size",
            str(float(gui_point_size.value)),
            "--ply_stride",
            str(int(gui_ply_stride.value)),
            "--video_width",
            str(video_width),
        ]
        print(f"[reload] restarting with sequence_dir={target_dir}")
        os.execv(sys.executable, argv)

    @gui_reload_current.on_click
    def _(_):
        _restart_with_sequence(sequence_dir)

    @gui_reload.on_click
    def _(_):
        if gui_seq_dir_input is None:
            print("[reload] sequence input unavailable; use Reload Current Dir")
            return
        _restart_with_sequence(str(gui_seq_dir_input.value).strip())

    @gui_browse_seq.on_click
    def _(_):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update()
            chosen = filedialog.askdirectory(
                initialdir=os.path.abspath(os.path.expanduser(sequence_dir)),
                title="Select sequence directory",
            )
            root.destroy()
        except Exception as exc:
            print(f"[browse] failed to open folder picker: {exc}")
            return

        if not chosen:
            return
        chosen = os.path.abspath(os.path.expanduser(chosen))
        if gui_seq_dir_input is not None:
            gui_seq_dir_input.value = chosen
        _restart_with_sequence(chosen)

    worker_thread = threading.Thread(target=_record_worker, daemon=True)
    worker_thread.start()
    if auto_load_cloud:
        try:
            _build_cloud_assets()
        except Exception as exc:
            _set_cloud_status(f"auto load failed: {exc}")
            print(f"[auto-load] failed: {exc}")
    _update_frame(0)

    def _play_loop() -> None:
        prev_time = time.time()
        while True:
            if gui_play.value:
                now = time.time()
                if now - prev_time >= 1.0 / gui_fps.value:
                    gui_frame.value = (gui_frame.value + 1) % len(frame_ids)
                    prev_time = now
            time.sleep(0.005)

    if background_mode:
        threading.Thread(target=_play_loop, daemon=True).start()
        print(f"Viser server running in background on port {port}")
    else:
        print(f"Viser server running in foreground on port {port}. Press Ctrl+C to stop.")
        try:
            _play_loop()
        except KeyboardInterrupt:
            record_stop.set()
            _close_capture_writer()
            print("\n(viser) Stopped by user.")


def load_predictions(path: str) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        return _load_pt(path)
    if suffix == ".npz":
        return _load_npz(path)
    if suffix in {".pkl", ".pickle"}:
        return _load_pkl(path)
    if suffix == ".ply":
        return _load_ply(path)
    raise ValueError(f"Unsupported file format: {suffix}. Use .npz/.pkl/.pt/.ply")


def validate_predictions(pred_dict: dict) -> None:
    required = ["images", "points", "conf"]
    missing = [k for k in required if k not in pred_dict]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Viser visualization for HorizonStream outputs.")
    parser.add_argument("--predictions", type=str, default=None, help="Path to prediction file (.npz/.pkl/.pt/.ply).")
    parser.add_argument("--sequence_dir", type=str, default=None, help="Sequence directory with points/poses/images structure.")
    parser.add_argument("--pose_file", type=str, default="abs_pose.txt", help="Pose txt filename under poses dir.")
    parser.add_argument("--gt_pose_file", type=str, default=None, help="Optional GT pose txt filename under poses dir (red trajectory).")
    parser.add_argument(
        "--pose_convention",
        type=str,
        default="c2w",
        choices=["auto", "w2c", "c2w"],
        help="Pose convention in pose file. c2w means direct read without inversion.",
    )
    parser.add_argument(
        "--pose_layout",
        type=str,
        default="r9t3",
        choices=["auto", "row", "col", "r9t3"],
        help="How 12 pose numbers are laid out in file. default r9t3.",
    )
    parser.add_argument("--intri_file", type=str, default=None, help="Optional intrinsics txt filename under poses dir.")
    parser.add_argument("--points_dir", type=str, default=None, help="Override points directory (expects .ply per frame).")
    parser.add_argument("--poses_dir", type=str, default=None, help="Override poses directory.")
    parser.add_argument("--rgb_dir", type=str, default=None, help="Override RGB image directory.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--share", action="store_true", help="Share viser server with others.")
    parser.add_argument("--background_mode", action="store_true", help="Run viser server in background mode.")
    parser.add_argument("--conf_threshold", type=float, default=20.0, help="Initial confidence threshold (percentage).")
    parser.add_argument("--subsample", type=int, default=2, help="Subsample factor for dense tensor mode.")
    parser.add_argument("--video_width", type=int, default=320, help="Video preview width.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation in dense tensor mode.")
    parser.add_argument(
        "--image_folder_for_sky_mask",
        type=str,
        default=None,
        help="Image folder used for sky mask inference when --mask_sky is enabled.",
    )
    parser.add_argument(
        "--canonical_first_frame",
        action="store_true",
        default=True,
        help="Use first frame as canonical frame (identity pose) in dense tensor mode.",
    )
    parser.add_argument(
        "--no_canonical_first_frame",
        action="store_false",
        dest="canonical_first_frame",
        help="Disable canonical first frame transform in dense tensor mode.",
    )
    parser.add_argument("--point_size", type=float, default=0.002, help="Initial point size for sequence mode.")
    parser.add_argument("--ply_stride", type=int, default=1, help="PLY downsample stride for sequence mode.")
    parser.add_argument("--max_cache_frames", type=int, default=32, help="LRU cache size for sequence mode.")
    parser.add_argument("--auto_load_cloud", action="store_true", help="Automatically load/rebuild the point cloud on startup.")
    args = parser.parse_args()

    # Backward-compatible UX: allow passing a sequence directory through --predictions.
    if args.sequence_dir is None and args.predictions and os.path.isdir(args.predictions):
        args.sequence_dir = args.predictions
        args.predictions = None

    if args.sequence_dir:
        if not os.path.isdir(args.sequence_dir):
            raise FileNotFoundError(f"Sequence directory not found: {args.sequence_dir}")
        _visualize_sequence_dir(
            sequence_dir=args.sequence_dir,
            pose_file=args.pose_file,
            gt_pose_file=args.gt_pose_file,
            pose_convention=args.pose_convention,
            pose_layout=args.pose_layout,
            points_dir=args.points_dir,
            poses_dir=args.poses_dir,
            rgb_dir=args.rgb_dir,
            intri_file=args.intri_file,
            port=args.port,
            share=args.share,
            background_mode=args.background_mode,
            point_size=args.point_size,
            ply_stride=args.ply_stride,
            video_width=args.video_width,
            max_cache_frames=args.max_cache_frames,
            load_from=None,
            load_to=None,
            auto_load_cloud=args.auto_load_cloud,
        )
        return

    if not args.predictions:
        raise ValueError("Provide either --predictions or --sequence_dir")
    if not os.path.isfile(args.predictions):
        raise FileNotFoundError(f"Prediction file not found: {args.predictions}")
    if args.mask_sky and args.image_folder_for_sky_mask is None:
        raise ValueError("--mask_sky requires --image_folder_for_sky_mask")

    pred_dict = load_predictions(args.predictions)
    validate_predictions(pred_dict)

    print(f"Loaded predictions from: {args.predictions}")
    print("Starting viser visualization...")
    viser_wrapper(
        pred_dict,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=args.image_folder_for_sky_mask,
        subsample=args.subsample,
        video_width=args.video_width,
        share=args.share,
        canonical_first_frame=args.canonical_first_frame,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
