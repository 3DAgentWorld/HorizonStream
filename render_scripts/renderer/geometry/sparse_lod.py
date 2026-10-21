"""Sparse LOD point hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .depth_points import choose_device


def _child_offsets(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=torch.long,
        device=device,
    )


@dataclass
class SparseLodTree:
    max_level: int = 16
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.max_level > 21:
            raise ValueError("lod_levels > 21 exceeds int64 cell id capacity")
        self.device = self.device or choose_device()
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0
        self.level_xyz = []
        self.level_rgb = []
        self.level_frames = []
        self.level_cell_ids = []
        self.sorted_xyz = np.empty((0, 3), dtype=np.float32)
        self.sorted_rgb = np.empty((0, 3), dtype=np.float32)
        self.sorted_frames = np.empty((0,), dtype=np.int32)

    def build(self, xyz: torch.Tensor, rgb: torch.Tensor, frames: torch.Tensor, log=print) -> None:
        xyz = xyz.to(self.device, dtype=torch.float32).contiguous()
        rgb = rgb.to(self.device, dtype=torch.float32).contiguous()
        frames = frames.to(self.device, dtype=torch.int32).contiguous()
        if len(xyz) == 0:
            self._set_empty()
            return

        mins = xyz.min(dim=0).values
        maxs = xyz.max(dim=0).values
        center_t = (mins + maxs) * 0.5
        extent = float((maxs - mins).max().item())
        self.scale = max(extent * 0.5 * 1.01, 1e-6)
        self.center = center_t.detach().cpu().numpy().astype(np.float32)
        normalized = (xyz - center_t) / self.scale
        log(f"[lod] building level-{self.max_level} sparse tree on {self.device.type}")
        self._compute_levels(xyz, rgb, frames, normalized)
        self._sort_finest()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_empty(self) -> None:
        self.level_xyz = [torch.empty((0, 3), dtype=torch.float32, device=self.device)]
        self.level_rgb = [torch.empty((0, 3), dtype=torch.float32, device=self.device)]
        self.level_frames = [torch.empty((0,), dtype=torch.int32, device=self.device)]
        self.level_cell_ids = [torch.empty((0,), dtype=torch.int64, device=self.device)]

    def _compute_levels(self, xyz, rgb, frames, normalized):
        max_lv = self.max_level
        grid = 2 ** max_lv
        cell_id = ((normalized[:, 0] + 1) * 0.5 * grid).long().clamp(0, grid - 1)
        cell_id = cell_id * grid + ((normalized[:, 1] + 1) * 0.5 * grid).long().clamp(0, grid - 1)
        cell_id = cell_id * grid + ((normalized[:, 2] + 1) * 0.5 * grid).long().clamp(0, grid - 1)

        unique_cells, inverse = torch.unique(cell_id, return_inverse=True)
        n = len(unique_cells)
        inv3 = inverse[:, None].expand(-1, 3)
        counts = torch.zeros(n, dtype=torch.float32, device=self.device)
        sum_xyz = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        sum_rgb = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
        sum_xyz.scatter_add_(0, inv3, xyz)
        sum_rgb.scatter_add_(0, inv3, rgb)
        min_frames = torch.full((n,), 2**30, dtype=torch.int32, device=self.device)
        min_frames.scatter_reduce_(0, inverse, frames, reduce="amin", include_self=True)

        self.level_xyz = [None] * (max_lv + 1)
        self.level_rgb = [None] * (max_lv + 1)
        self.level_frames = [None] * (max_lv + 1)
        self.level_cell_ids = [None] * (max_lv + 1)
        self.level_xyz[max_lv] = sum_xyz / counts[:, None]
        self.level_rgb[max_lv] = sum_rgb / counts[:, None]
        self.level_frames[max_lv] = min_frames
        self.level_cell_ids[max_lv] = unique_cells

        iz = unique_cells % grid
        iy = (unique_cells // grid) % grid
        ix = unique_cells // (grid * grid)
        prev_coords = torch.stack([ix, iy, iz], dim=1)
        prev_xyz = self.level_xyz[max_lv]
        prev_rgb = self.level_rgb[max_lv]
        prev_frames = min_frames
        prev_counts = counts

        for lv in range(max_lv - 1, -1, -1):
            parent_coords = prev_coords >> 1
            size = 2 ** lv if lv > 0 else 1
            parent_id = parent_coords[:, 0] * size * size + parent_coords[:, 1] * size + parent_coords[:, 2]
            unique_parents, inv = torch.unique(parent_id, return_inverse=True)
            nparts = len(unique_parents)
            inv3 = inv[:, None].expand(-1, 3)
            weights = prev_counts[:, None]
            p_counts = torch.zeros(nparts, dtype=torch.float32, device=self.device)
            p_xyz = torch.zeros(nparts, 3, dtype=torch.float32, device=self.device)
            p_rgb = torch.zeros(nparts, 3, dtype=torch.float32, device=self.device)
            p_frames = torch.full((nparts,), 2**30, dtype=torch.int32, device=self.device)
            p_counts.scatter_add_(0, inv, prev_counts)
            p_xyz.scatter_add_(0, inv3, prev_xyz * weights)
            p_rgb.scatter_add_(0, inv3, prev_rgb * weights)
            p_frames.scatter_reduce_(0, inv, prev_frames, reduce="amin", include_self=True)

            self.level_xyz[lv] = p_xyz / p_counts[:, None]
            self.level_rgb[lv] = p_rgb / p_counts[:, None]
            self.level_frames[lv] = p_frames
            self.level_cell_ids[lv] = unique_parents
            if lv > 0:
                pz = unique_parents % size
                py = (unique_parents // size) % size
                px = unique_parents // (size * size)
                prev_coords = torch.stack([px, py, pz], dim=1)
            prev_xyz = self.level_xyz[lv]
            prev_rgb = self.level_rgb[lv]
            prev_frames = p_frames
            prev_counts = p_counts

    def _sort_finest(self) -> None:
        lv = self.max_level
        order = torch.argsort(self.level_frames[lv].long(), stable=True)
        self.sorted_xyz = self.level_xyz[lv][order].detach().cpu().numpy()
        self.sorted_rgb = self.level_rgb[lv][order].detach().cpu().numpy()
        self.sorted_frames = self.level_frames[lv][order].detach().cpu().numpy()

    def select_visible(
        self,
        w2c: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near: float,
        far: float,
        reveal_frame: int,
        target_pixels: float = 1.5,
        frame_start: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.level_xyz or len(self.level_xyz[0]) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        R = torch.from_numpy(w2c[:3, :3].astype(np.float32)).to(self.device)
        t = torch.from_numpy(w2c[:3, 3].astype(np.float32)).to(self.device)
        scene_extent = self.scale * 2.0
        focal = float(fy)
        active = torch.arange(len(self.level_xyz[0]), dtype=torch.long, device=self.device)
        child_offsets = _child_offsets(self.device)
        out_xyz = []
        out_rgb = []

        for lv in range(self.max_level + 1):
            if len(active) == 0:
                break
            xyz = self.level_xyz[lv][active]
            rgb = self.level_rgb[lv][active]
            frames = self.level_frames[lv][active]
            cell_size = scene_extent / (2 ** lv)
            half_diag = cell_size * 0.866
            cam = (R @ xyz.T + t[:, None]).T
            z = cam[:, 2]
            safe_z = z.clamp(min=near * 0.5)
            u = cam[:, 0] * fx / safe_z + cx
            v = cam[:, 1] * fy / safe_z + cy
            margin = half_diag * focal / safe_z
            visible = (
                (frames >= frame_start)
                & (frames <= reveal_frame)
                & (z > near - half_diag)
                & (z < far + half_diag)
                & (u > -margin)
                & (u < width + margin)
                & (v > -margin)
                & (v < height + margin)
            )
            proj_px = cell_size * focal / safe_z
            emit = visible & ((proj_px <= target_pixels) | (lv == self.max_level))
            refine = visible & (proj_px > target_pixels) & (lv < self.max_level)
            if emit.any():
                out_xyz.append(xyz[emit])
                out_rgb.append(rgb[emit])
            if lv < self.max_level and refine.any():
                parent_idx = active[refine]
                parent_ids = self.level_cell_ids[lv][parent_idx]
                size = 2 ** lv if lv > 0 else 1
                zc = parent_ids % size
                yc = (parent_ids // size) % size
                xc = parent_ids // (size * size)
                coords = torch.stack([xc, yc, zc], dim=1)
                child_coords = (coords[:, None, :] * 2 + child_offsets[None]).reshape(-1, 3)
                child_size = 2 ** (lv + 1)
                child_ids = (
                    child_coords[:, 0] * child_size * child_size
                    + child_coords[:, 1] * child_size
                    + child_coords[:, 2]
                )
                next_ids = self.level_cell_ids[lv + 1]
                pos = torch.searchsorted(next_ids, child_ids)
                valid = pos < len(next_ids)
                pos = pos.clamp(max=max(len(next_ids) - 1, 0))
                valid = valid & (next_ids[pos] == child_ids)
                active = pos[valid]
            else:
                active = active[:0]

        if not out_xyz:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        return torch.cat(out_xyz).cpu().numpy(), torch.cat(out_rgb).cpu().numpy()
