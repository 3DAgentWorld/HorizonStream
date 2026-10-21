"""Post-unprojection point filters."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from ..core.configuration import FilterSettings


LogFn = Callable[[str], None]


def filter_sky_colored_points(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    frames: torch.Tensor,
    settings: FilterSettings,
    log: LogFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not settings.color_filter_enabled or len(rgb) == 0:
        return xyz, rgb, frames

    colors = (rgb.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    c = colors.astype(np.int16)
    is_white = np.all(colors >= settings.white_threshold, axis=1)
    if settings.bright_channel_threshold > 0:
        is_bright = np.any(colors >= settings.bright_channel_threshold, axis=1)
    else:
        is_bright = np.zeros(len(colors), dtype=bool)

    rg_diff = np.abs(c[:, 0] - c[:, 1])
    bg_diff = c[:, 2] - c[:, 1]
    br_diff = c[:, 2] - c[:, 0]
    is_blue = (
        (colors[:, 2] >= settings.blue_channel_min)
        & (colors[:, 0] <= settings.red_channel_max)
        & (colors[:, 1] >= settings.green_channel_min)
        & (rg_diff <= settings.red_green_delta_max)
        & (bg_diff >= settings.blue_green_delta_min)
        & (br_diff >= settings.blue_red_delta_min)
        & (colors[:, 2] > colors[:, 1])
    )

    cmax = np.max(c, axis=1)
    cmin = np.min(c, axis=1)
    sat = (cmax - cmin).astype(np.float32) / np.maximum(cmax, 1).astype(np.float32)
    is_cyan_haze = (
        (colors[:, 1] >= 170)
        & (colors[:, 2] >= 170)
        & (colors[:, 0] <= 220)
        & ((c[:, 1] - c[:, 0]) >= 8)
        & ((c[:, 2] - c[:, 0]) >= 8)
        & (np.abs(c[:, 2] - c[:, 1]) <= 30)
        & (cmax >= 170)
        & (sat <= 0.23)
    )

    keep = ~(is_white | is_bright | is_blue | is_cyan_haze)
    removed = int((~keep).sum())
    log(f"[sky-color] removed {removed:,} residual sky points")
    keep_t = torch.from_numpy(keep).to(xyz.device)
    return xyz[keep_t], rgb[keep_t], frames[keep_t]


def filter_statistical_outliers(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    frames: torch.Tensor,
    settings: FilterSettings,
    log: LogFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not settings.statistical_filter_enabled or len(xyz) == 0:
        return xyz, rgb, frames
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        log("[outlier] scipy not installed, skipped")
        return xyz, rgb, frames

    pts = xyz.cpu().numpy()
    k = min(settings.statistical_neighbors + 1, len(pts))
    if k <= 2:
        return xyz, rgb, frames
    t0 = time.time()
    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=k)
    mean_dist = distances[:, 1:].mean(axis=1)
    finite = np.isfinite(mean_dist)
    sigma = float(np.std(mean_dist[finite]))
    mu = float(np.mean(mean_dist[finite]))
    keep = finite & (mean_dist <= mu + settings.statistical_std_ratio * sigma) if sigma > 0 else finite
    log(f"[outlier] removed {(~keep).sum():,} points in {time.time() - t0:.1f}s")
    keep_t = torch.from_numpy(keep).to(xyz.device)
    return xyz[keep_t], rgb[keep_t], frames[keep_t]


def filter_radius_outliers(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    frames: torch.Tensor,
    settings: FilterSettings,
    log: LogFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not settings.radius_filter_enabled or len(xyz) == 0:
        return xyz, rgb, frames
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        log("[radius-outlier] scipy not installed, skipped")
        return xyz, rgb, frames

    pts = xyz.cpu().numpy()
    if len(pts) < settings.radius_filter_min_neighbors:
        return xyz, rgb, frames
    t0 = time.time()
    tree = cKDTree(pts)
    counts = tree.query_ball_point(
        pts,
        r=float(settings.radius_filter_radius),
        return_length=True,
        workers=-1,
    )
    keep = np.asarray(counts) >= int(settings.radius_filter_min_neighbors)
    log(f"[radius-outlier] removed {(~keep).sum():,} points in {time.time() - t0:.1f}s")
    keep_t = torch.from_numpy(keep).to(xyz.device)
    return xyz[keep_t], rgb[keep_t], frames[keep_t]
