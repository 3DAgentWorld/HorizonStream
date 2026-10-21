"""Depth unprojection into world-space colored points."""

from __future__ import annotations

import torch


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unproject_depth_batch(
    depths: torch.Tensor,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    min_depth: float,
    max_depth: float,
    downsample: int,
    jitter: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened valid world points, RGB values, and per-frame counts."""
    device = depths.device
    B, H, W = depths.shape
    step = max(int(downsample), 1)
    d = depths[:, ::step, ::step]
    rgb = images[:, ::step, ::step]
    _, Hd, Wd = d.shape

    us = torch.arange(0, W, step, dtype=torch.float32, device=device)
    vs = torch.arange(0, H, step, dtype=torch.float32, device=device)
    u_grid = us[None, None, :].expand(B, Hd, Wd)
    v_grid = vs[None, :, None].expand(B, Hd, Wd)
    if jitter and step > 1:
        span = step * 0.8
        u_grid = u_grid + (torch.rand(B, Hd, Wd, device=device) - 0.5) * span
        v_grid = v_grid + (torch.rand(B, Hd, Wd, device=device) - 0.5) * span

    fx = intrinsics[:, 0, 0:1, None]
    fy = intrinsics[:, 1, 1:2, None]
    cx = intrinsics[:, 0, 2:3, None]
    cy = intrinsics[:, 1, 2:3, None]

    pts_cam = torch.stack([
        (u_grid - cx) * d / fx,
        (v_grid - cy) * d / fy,
        d,
    ], dim=-1).reshape(B, -1, 3)

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    pts_world = torch.bmm(pts_cam, R.transpose(1, 2)) + t[:, None, :]
    valid = (d > min_depth) & (d < max_depth)
    counts = valid.reshape(B, -1).sum(dim=1).to(torch.int32)
    flat_valid = valid.reshape(-1)
    return (
        pts_world.reshape(-1, 3)[flat_valid],
        rgb.reshape(-1, 3)[flat_valid].float().div(255.0),
        counts,
    )
