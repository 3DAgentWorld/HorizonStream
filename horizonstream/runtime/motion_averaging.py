from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from horizonstream.utils.vendor.models.components.utils.rotation import mat_to_quat, quat_to_mat
from horizonstream.runtime.averaging.median_rotations import get_median_rotations
from horizonstream.runtime.averaging.position_averaging import position_averaging
from horizonstream.runtime.averaging.rotation_averaging import rotation_averaging


def affine_padding(c2w: torch.Tensor) -> torch.Tensor:
    if c2w.shape[-2] == 4:
        return c2w
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))
    pad = torch.cat([pad0, pad1], dim=-1)
    return torch.cat([c2w, pad], dim=-2)


def affine_inverse(A: torch.Tensor) -> torch.Tensor:
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def get_4d_anti_diagonal_medians(tensor_4d: torch.Tensor) -> torch.Tensor:
    B, Sli, Win, C = tensor_4d.shape
    num_diagonals = Sli + Win - 1
    medians = tensor_4d.new_zeros((B, num_diagonals, C))
    for c in range(num_diagonals):
        i_start = max(0, c - Win + 1)
        i_end = min(c, Sli - 1)
        i_indices = torch.arange(i_start, i_end + 1, device=tensor_4d.device)
        j_indices = c - i_indices
        diag_elements = tensor_4d[:, i_indices, j_indices, :]
        medians[:, c, :] = diag_elements.median(dim=1).values
    return medians


def _get_median_positions(
    positions: torch.Tensor,
    translations: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    B, W = positions.shape[:2]
    device = positions.device
    scale = torch.ones((B, 1), device=device, dtype=dtype)
    median_pos = torch.median(
        positions.view(B, W, 3) + translations.view(B, W, 3),
        dim=1,
    ).values
    return median_pos.view(B, 1, 3, 1), scale


@torch.no_grad()
def online_motion_averaging(
    chunk_cam_map: torch.Tensor,
    chunk_idx: int,
    B: int,
    Win: int,
    online_absolute_poses: torch.Tensor,
    online_S: torch.Tensor,
    valid_relative_poses: torch.Tensor,
    valid_focals_scales_shifts: torch.Tensor,
    online_ptr: int,
    valid_ptr: int,
    last_absolute_poses: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[int, int]:
    cam_dim = chunk_cam_map.shape[-1]
    chunk_cam_map = chunk_cam_map.reshape(B, -1, Win, cam_dim).to(dtype=dtype)

    FN = chunk_cam_map.shape[1]
    T = chunk_cam_map[..., 0:3].unsqueeze(-1)
    R = quat_to_mat(chunk_cam_map[..., 3:7])
    w2c = torch.cat([R, T], dim=-1)

    if chunk_idx == 0:
        valid_relative_poses[:, valid_ptr] = w2c[:, -1]
        if chunk_cam_map.shape[-1] <= 9:
            valid_focals_scales_shifts[:, valid_ptr, :, :2] = chunk_cam_map[:, -1, :, 7:9]
            default_ss = chunk_cam_map.new_tensor([1.0, 0.0])
            valid_focals_scales_shifts[:, valid_ptr, :, 2:] = default_ss
        else:
            valid_focals_scales_shifts[:, valid_ptr] = chunk_cam_map[:, -1, :, 7:11]
        valid_ptr += 1
        online_absolute_poses[:, :Win] = w2c[:, -1]
        if last_absolute_poses is not None:
            last_absolute_poses[:, :Win] = w2c[:, -1]
        online_ptr = Win
        return online_ptr, valid_ptr

    for i in range(FN):
        tmp_w2c = w2c[:, i]
        valid_relative_poses[:, valid_ptr] = tmp_w2c
        if chunk_cam_map.shape[-1] <= 9:
            valid_focals_scales_shifts[:, valid_ptr, :, :2] = chunk_cam_map[:, i, :, 7:9]
            default_ss = chunk_cam_map.new_tensor([1.0, 0.0])
            valid_focals_scales_shifts[:, valid_ptr, :, 2:] = default_ss
        else:
            valid_focals_scales_shifts[:, valid_ptr] = chunk_cam_map[:, i, :, 7:11]
        valid_ptr += 1

        start_ptr = online_ptr - Win + 1
        cache_poses = online_absolute_poses[:, start_ptr:online_ptr]

        cache_rotations = cache_poses[:, :, :3, :3]
        tmp_rotations = tmp_w2c[:, : Win - 1, :3, :3].mT @ cache_rotations
        median_rotation, _ = get_median_rotations(tmp_rotations)

        cache_translations = cache_poses[:, :, :3, 3:]
        cache_positions = -cache_rotations.mT @ cache_translations
        tmp_translations = cache_rotations.mT @ tmp_w2c[:, : Win - 1, :3, 3:]
        median_position, online_S[:, start_ptr] = _get_median_positions(
            cache_positions,
            tmp_translations,
        )
        median_translation = -median_rotation @ median_position
        online_absolute_poses[:, online_ptr] = torch.cat([median_rotation, median_translation], dim=-1)

        if last_absolute_poses is not None:
            last_cache_poses = last_absolute_poses[:, start_ptr:online_ptr]
            last_cache_rotations = last_cache_poses[:, :, :3, :3]
            last_cache_translations = last_cache_poses[:, :, :3, 3:]
            last_rotation = tmp_w2c[:, -2, :3, :3].mT @ last_cache_rotations[:, -1]
            last_position = -last_cache_rotations[:, -1].mT @ last_cache_translations[:, -1]
            last_position = last_position + last_cache_rotations[:, -1].mT @ tmp_w2c[:, -2, :3, 3:]
            last_translation = -last_rotation @ last_position
            last_absolute_poses[:, online_ptr] = torch.cat([last_rotation, last_translation], dim=-1)

        online_ptr += 1

    return online_ptr, valid_ptr


@torch.no_grad()
def reconstruct_selected_edge_absolute_poses(
    valid_relative_poses: torch.Tensor,
    Win: int,
    edge_idx: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B, num_valid = valid_relative_poses.shape[:2]
    frames_num = num_valid + Win - 1
    device = valid_relative_poses.device

    edge_absolute_poses = torch.empty(B, frames_num, 3, 4, device=device, dtype=dtype)
    edge_absolute_poses[:, :Win] = valid_relative_poses[:, 0].to(dtype=dtype)

    for row_idx in range(1, num_valid):
        past_idx = row_idx + edge_idx
        cur_idx = row_idx + Win - 1
        rel_edge = valid_relative_poses[:, row_idx, edge_idx].to(dtype=dtype)
        rel_edge4 = affine_padding(rel_edge)
        past_w2c4 = affine_padding(edge_absolute_poses[:, past_idx])
        cur_w2c4 = affine_inverse(rel_edge4) @ past_w2c4
        edge_absolute_poses[:, cur_idx] = cur_w2c4[:, :3]

    return edge_absolute_poses


@torch.no_grad()
def w2c_absolute_poses_to_cam_map(
    w2c_abs: torch.Tensor,
    focal_map: torch.Tensor,
) -> torch.Tensor:
    abs_R = w2c_abs[..., :3, :3]
    abs_C = -abs_R.mT @ w2c_abs[..., :3, 3:]
    rel_T = (-abs_R @ (abs_C - abs_C[:, 0])).squeeze(-1)
    rel_Q = mat_to_quat(abs_R @ abs_R[:, 0].mT)
    return torch.cat([rel_T, rel_Q, focal_map], dim=-1)


@torch.no_grad()
def compute_motion_averaged_camera_maps(
    chunk_cam_maps: List[torch.Tensor],
    frames_num: int,
    window_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    enable_offline: bool = False,
) -> Dict[str, torch.Tensor | str]:
    if not chunk_cam_maps:
        raise ValueError("chunk_cam_maps must not be empty.")

    first_chunk = chunk_cam_maps[0]
    cam_remain_shape = first_chunk.shape[-2:]
    B = 1

    if len(chunk_cam_maps) == 1:
        online_cam_map = first_chunk.reshape(B, -1, *cam_remain_shape)[:, -1, :, :9]
        results = {
            "online_cam_map": online_cam_map,
            "last_cam_map": online_cam_map,
            "last_cam_map_label": "last",
        }
        if enable_offline:
            results["offline_cam_map_error"] = "single_chunk_no_offline_averaging"
        return results

    valid_relative_poses = torch.empty(
        B,
        frames_num - (window_size - 1),
        window_size,
        3,
        4,
        device=first_chunk.device,
        dtype=dtype,
    )
    valid_focals_scales_shifts = torch.empty(
        B,
        frames_num - (window_size - 1),
        window_size,
        4,
        device=first_chunk.device,
        dtype=dtype,
    )
    online_absolute_poses = torch.empty(B, frames_num, 3, 4, device=first_chunk.device, dtype=dtype)
    last_absolute_poses = torch.empty(B, frames_num, 3, 4, device=first_chunk.device, dtype=dtype)
    online_S = torch.ones(B, frames_num - (window_size - 1), 1, device=first_chunk.device, dtype=dtype)
    online_ptr = 0
    valid_ptr = 0

    for chunk_idx, chunk_cam_map in enumerate(chunk_cam_maps):
        online_ptr, valid_ptr = online_motion_averaging(
            chunk_cam_map=chunk_cam_map,
            chunk_idx=chunk_idx,
            B=B,
            Win=window_size,
            online_absolute_poses=online_absolute_poses,
            online_S=online_S,
            valid_relative_poses=valid_relative_poses,
            valid_focals_scales_shifts=valid_focals_scales_shifts,
            online_ptr=online_ptr,
            valid_ptr=valid_ptr,
            last_absolute_poses=last_absolute_poses,
            dtype=dtype,
        )

    online_R = online_absolute_poses[..., :3, :3]
    online_C = -online_R.mT @ online_absolute_poses[..., :3, 3:]
    online_T = (-online_R @ (online_C - online_C[:, 0])).squeeze(-1)
    online_Q = mat_to_quat(online_R @ online_R[:, 0].mT)
    median_focal = get_4d_anti_diagonal_medians(valid_focals_scales_shifts[..., :2])
    online_cam_map = torch.cat([online_T, online_Q, median_focal], dim=-1)

    lag_absolute_poses = reconstruct_selected_edge_absolute_poses(
        valid_relative_poses,
        window_size,
        edge_idx=0,
        dtype=dtype,
    )
    lag_cam_map = w2c_absolute_poses_to_cam_map(lag_absolute_poses, median_focal)

    last_R = last_absolute_poses[..., :3, :3]
    last_C = -last_R.mT @ last_absolute_poses[..., :3, 3:]
    last_T = (-last_R @ (last_C - last_C[:, 0])).squeeze(-1)
    last_Q = mat_to_quat(last_R @ last_R[:, 0].mT)
    last_cam_map = torch.cat([last_T, last_Q, median_focal], dim=-1)

    results: Dict[str, torch.Tensor | str] = {
        "online_cam_map": online_cam_map,
        "last_cam_map": last_cam_map,
        "last_cam_map_label": "last",
        "lag_cam_map": lag_cam_map,
        "lag_cam_map_label": f"t->{window_size}",
    }

    if enable_offline:
        try:
            offline_rel_R = valid_relative_poses[0, :, :, :3, :3]
            offline_rel_T = valid_relative_poses[0, :, :, :3, 3:]
            offline_R = rotation_averaging(offline_rel_R, online_R[0])
            offline_R = offline_R @ offline_R[0].mT
            offline_Q = mat_to_quat(offline_R[None, ...])
            offline_C, offline_S = position_averaging(offline_rel_T, offline_R)
            offline_C = offline_C[None, :, :, None]
            offline_R_batch = offline_R[None, ...]
            offline_T = (-offline_R_batch @ (offline_C - offline_C[:, 0])).squeeze(-1)
            results["offline_cam_map"] = torch.cat([offline_T, offline_Q, median_focal], dim=-1)
            results["offline_scales"] = offline_S
        except Exception as e:
            results["offline_cam_map_error"] = f"{type(e).__name__}:{e}"

    return results
