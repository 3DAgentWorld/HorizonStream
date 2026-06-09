import torch
from horizonstream.utils.vendor.models.components.utils.rotation import (
    quat_to_mat,
    mat_to_quat,
)


def compose_abs_from_rel(
    rel_pose_enc: torch.Tensor, keyframe_indices: torch.Tensor
) -> torch.Tensor:
    squeeze_batch = False
    if rel_pose_enc.ndim == 2:
        rel_pose_enc = rel_pose_enc.unsqueeze(0)
        squeeze_batch = True
    if keyframe_indices.ndim == 1:
        keyframe_indices = keyframe_indices.unsqueeze(0)
    if rel_pose_enc.ndim != 3 or keyframe_indices.ndim != 2:
        raise ValueError(
            f"Expected rel_pose_enc [B,S,D] or [S,D] and keyframe_indices [B,S] or [S], "
            f"got {tuple(rel_pose_enc.shape)} and {tuple(keyframe_indices.shape)}"
        )

    B, S, _ = rel_pose_enc.shape
    device = rel_pose_enc.device
    dtype = rel_pose_enc.dtype

    rel_t = rel_pose_enc[..., :3]
    rel_q = rel_pose_enc[..., 3:7]
    rel_f = rel_pose_enc[..., 7:9]
    rel_R = quat_to_mat(rel_q.reshape(-1, 4)).reshape(B, S, 3, 3)

    abs_R = torch.zeros(B, S, 3, 3, device=device, dtype=dtype)
    abs_t = torch.zeros(B, S, 3, device=device, dtype=dtype)
    abs_f = torch.zeros(B, S, 2, device=device, dtype=dtype)

    for b in range(B):
        abs_R[b, 0] = rel_R[b, 0]
        abs_t[b, 0] = rel_t[b, 0]
        abs_f[b, 0] = rel_f[b, 0]
        for s in range(1, S):
            ref_idx = int(keyframe_indices[b, s].item())
            abs_R[b, s] = rel_R[b, s] @ abs_R[b, ref_idx]
            abs_t[b, s] = rel_t[b, s] + rel_R[b, s] @ abs_t[b, ref_idx]
            abs_f[b, s] = rel_f[b, s]

    abs_q = mat_to_quat(abs_R.reshape(-1, 3, 3)).reshape(B, S, 4)
    abs_pose_enc = torch.cat([abs_t, abs_q, abs_f], dim=-1)
    if squeeze_batch:
        return abs_pose_enc[0]
    return abs_pose_enc
