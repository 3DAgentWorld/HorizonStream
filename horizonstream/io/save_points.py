import os
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _radius_outlier_filter(
    points,
    colors=None,
    enabled=True,
    radius=0.10,
    min_points=16,
):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if not enabled or cKDTree is None or pts.shape[0] == 0:
        return pts, cols
    radius = float(radius)
    min_points = int(min_points)
    if radius <= 0.0 or min_points <= 1 or pts.shape[0] < min_points:
        return pts, cols
    tree = cKDTree(pts)
    try:
        counts = tree.query_ball_point(pts, r=radius, return_length=True, workers=-1)
    except TypeError:
        counts = tree.query_ball_point(pts, r=radius, return_length=True)
    keep = np.asarray(counts) >= min_points
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def _sky_color_filter(
    points,
    colors=None,
    enabled=True,
    white_threshold=240,
    bright_any_threshold=200,
    blue_b_min=155,
    blue_r_max=195,
    blue_g_min=100,
    blue_rg_diff_max=95,
    blue_bg_diff_min=8,
    blue_br_diff_min=15,
):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if not enabled or cols is None or pts.shape[0] == 0:
        return pts, cols
    if cols.size == 0:
        return pts, cols
    if cols.max() <= 1.0:
        rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(cols, 0, 255).astype(np.uint8)

    white_threshold = int(white_threshold)
    is_white = (
        (rgb[:, 0] >= white_threshold)
        & (rgb[:, 1] >= white_threshold)
        & (rgb[:, 2] >= white_threshold)
    )

    bright_any_threshold = int(bright_any_threshold)
    if bright_any_threshold > 0:
        is_bright_any = (
            (rgb[:, 0] >= bright_any_threshold)
            | (rgb[:, 1] >= bright_any_threshold)
            | (rgb[:, 2] >= bright_any_threshold)
        )
    else:
        is_bright_any = np.zeros(rgb.shape[0], dtype=bool)

    c = rgb.astype(np.int16)
    rg_diff = np.abs(c[:, 0] - c[:, 1])
    bg_diff = c[:, 2] - c[:, 1]
    br_diff = c[:, 2] - c[:, 0]
    is_sky_blue = (
        (rgb[:, 2] >= int(blue_b_min))
        & (rgb[:, 0] <= int(blue_r_max))
        & (rgb[:, 1] >= int(blue_g_min))
        & (rg_diff <= int(blue_rg_diff_max))
        & (bg_diff >= int(blue_bg_diff_min))
        & (br_diff >= int(blue_br_diff_min))
        & (rgb[:, 2] > rgb[:, 1])
    )

    cmax = np.max(c, axis=1)
    cmin = np.min(c, axis=1)
    sat = (cmax - cmin).astype(np.float32) / np.maximum(cmax, 1).astype(np.float32)
    is_cyan_haze = (
        (rgb[:, 1] >= 170)
        & (rgb[:, 2] >= 170)
        & (rgb[:, 0] <= 220)
        & ((c[:, 1] - c[:, 0]) >= 8)
        & ((c[:, 2] - c[:, 0]) >= 8)
        & (np.abs(c[:, 2] - c[:, 1]) <= 30)
        & (cmax >= 170)
        & (sat <= 0.23)
    )

    keep = ~(is_white | is_bright_any | is_sky_blue | is_cyan_haze)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def _voxel_downsample(points, colors=None, voxel_size=None):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if voxel_size in (None, "", "null") or pts.shape[0] == 0:
        return pts, cols
    voxel_size = float(voxel_size)
    if voxel_size <= 0.0:
        return pts, cols
    coords = np.floor(pts / voxel_size).astype(np.int64)
    _, keep = np.unique(coords, axis=0, return_index=True)
    keep = np.sort(keep)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def _random_sample(points, colors=None, ratio=None, seed=0):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if ratio in (None, "", "null") or pts.shape[0] == 0:
        return pts, cols
    ratio = float(ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        return pts, cols
    keep_count = max(1, int(round(pts.shape[0] * ratio)))
    rng = np.random.default_rng(seed)
    keep = rng.choice(pts.shape[0], size=keep_count, replace=False)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def _limit_points(points, colors=None, max_points=None, seed=0):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if max_points in (None, "", "null") or pts.shape[0] <= int(max_points):
        return pts, cols
    if int(max_points) <= 0:
        return pts[:0], None if cols is None else cols[:0]
    rng = np.random.default_rng(seed)
    keep = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def prepare_pointcloud(
    points,
    colors=None,
    *,
    voxel_size=None,
    random_sample_ratio=None,
    max_points=None,
    sky_color_filter=False,
    sky_white_threshold=240,
    sky_bright_any_threshold=200,
    sky_blue_b_min=155,
    sky_blue_r_max=195,
    sky_blue_g_min=100,
    sky_blue_rg_diff_max=95,
    sky_blue_bg_diff_min=8,
    sky_blue_br_diff_min=15,
    outlier_filter=False,
    radius_outlier_radius=0.10,
    radius_outlier_min_points=16,
    seed=0,
):
    pts, cols = _voxel_downsample(points, colors=colors, voxel_size=voxel_size)
    pts, cols = _random_sample(pts, colors=cols, ratio=random_sample_ratio, seed=seed)
    pts, cols = _limit_points(pts, colors=cols, max_points=max_points, seed=seed)
    pts, cols = _sky_color_filter(
        pts,
        colors=cols,
        enabled=sky_color_filter,
        white_threshold=sky_white_threshold,
        bright_any_threshold=sky_bright_any_threshold,
        blue_b_min=sky_blue_b_min,
        blue_r_max=sky_blue_r_max,
        blue_g_min=sky_blue_g_min,
        blue_rg_diff_max=sky_blue_rg_diff_max,
        blue_bg_diff_min=sky_blue_bg_diff_min,
        blue_br_diff_min=sky_blue_br_diff_min,
    )
    pts, cols = _radius_outlier_filter(
        pts,
        colors=cols,
        enabled=outlier_filter,
        radius=radius_outlier_radius,
        min_points=radius_outlier_min_points,
    )
    return pts, cols


def save_pointcloud(
    path,
    points,
    colors=None,
    max_points=None,
    seed=0,
    voxel_size=None,
    random_sample_ratio=None,
    sky_color_filter=False,
    sky_white_threshold=240,
    sky_bright_any_threshold=200,
    sky_blue_b_min=155,
    sky_blue_r_max=195,
    sky_blue_g_min=100,
    sky_blue_rg_diff_max=95,
    sky_blue_bg_diff_min=8,
    sky_blue_br_diff_min=15,
    outlier_filter=False,
    radius_outlier_radius=0.10,
    radius_outlier_min_points=16,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts, cols = prepare_pointcloud(
        points,
        colors=colors,
        voxel_size=voxel_size,
        random_sample_ratio=random_sample_ratio,
        max_points=max_points,
        sky_color_filter=sky_color_filter,
        sky_white_threshold=sky_white_threshold,
        sky_bright_any_threshold=sky_bright_any_threshold,
        sky_blue_b_min=sky_blue_b_min,
        sky_blue_r_max=sky_blue_r_max,
        sky_blue_g_min=sky_blue_g_min,
        sky_blue_rg_diff_max=sky_blue_rg_diff_max,
        sky_blue_bg_diff_min=sky_blue_bg_diff_min,
        sky_blue_br_diff_min=sky_blue_br_diff_min,
        outlier_filter=outlier_filter,
        radius_outlier_radius=radius_outlier_radius,
        radius_outlier_min_points=radius_outlier_min_points,
        seed=seed,
    )
    pts = pts.astype(np.float32, copy=False)
    if cols is not None:
        if cols.size == 0:
            cols = cols.astype(np.uint8, copy=False)
        elif cols.max() <= 1.0:
            cols = (cols * 255.0).astype(np.uint8)
        else:
            cols = cols.astype(np.uint8)
        has_color = True
    else:
        cols = None
        has_color = False

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n".encode("ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if has_color:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        if has_color:
            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
            vertex_data = np.empty(pts.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = pts[:, 0]
            vertex_data["y"] = pts[:, 1]
            vertex_data["z"] = pts[:, 2]
            vertex_data["red"] = cols[:, 0]
            vertex_data["green"] = cols[:, 1]
            vertex_data["blue"] = cols[:, 2]
            vertex_data.tofile(f)
        else:
            vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            vertex_data = np.empty(pts.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = pts[:, 0]
            vertex_data["y"] = pts[:, 1]
            vertex_data["z"] = pts[:, 2]
            vertex_data.tofile(f)
