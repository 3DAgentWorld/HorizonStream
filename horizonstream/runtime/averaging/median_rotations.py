import math
import torch
from typing import Tuple, Optional


def _median_midpoint(x: torch.Tensor, dim: int) -> torch.Tensor:
    xs, _ = torch.sort(x, dim=dim)
    n = xs.size(dim)
    mid = n // 2
    if n % 2 == 1:
        return xs.select(dim, mid)
    else:
        return 0.5 * (xs.select(dim, mid - 1) + xs.select(dim, mid))


def _hat(w: torch.Tensor) -> torch.Tensor:
    """
    w(...,3) -> W(...,3,3)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    return torch.stack(
        (
            torch.stack((O, -wz,  wy), dim=-1),
            torch.stack((wz,  O, -wx), dim=-1),
            torch.stack((-wy, wx,  O), dim=-1),
        ),
        dim=-2,
    )


def _so3_exp(w: torch.Tensor, eye3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    exp: so(3) 向量 w(...,3) -> SO(3) 矩阵 R(...,3,3)
    Rodrigues 公式 + 小角度稳定
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    W = _hat(w)  # (...,3,3)

    # broadcast I to (...,3,3)
    I = eye3.view((1,) * (W.ndim - 2) + (3, 3)).expand_as(W)

    theta2 = theta * theta
    small = theta < eps

    # a = sinθ/θ, b = (1-cosθ)/θ^2 with stable series for small θ
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )

    return I + a[..., None] * W + b[..., None] * (W @ W)


def _so3_log(R: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    log: SO(3) 矩阵 R(...,3,3) -> (w(...,3), theta(...))
    w = angle * axis
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...)

    vee = torch.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        dim=-1,
    )  # (...,3)

    sin_theta = torch.sin(theta)
    k = theta / (2.0 * sin_theta + eps)
    w_general = k[..., None] * vee

    # small-angle: w ≈ 0.5 * vee
    small = theta < 1e-4
    w_small = 0.5 * vee

    # near pi: recover axis from diagonal (avoid division by tiny sinθ)
    near_pi = (math.pi - theta) < 1e-4
    x = torch.sqrt(torch.clamp((R[..., 0, 0] + 1.0) * 0.5, min=0.0))
    y = torch.sqrt(torch.clamp((R[..., 1, 1] + 1.0) * 0.5, min=0.0))
    z = torch.sqrt(torch.clamp((R[..., 2, 2] + 1.0) * 0.5, min=0.0))
    axis = torch.stack((x, y, z), dim=-1)

    sx = torch.sign(R[..., 2, 1] - R[..., 1, 2])
    sy = torch.sign(R[..., 0, 2] - R[..., 2, 0])
    sz = torch.sign(R[..., 1, 0] - R[..., 0, 1])
    axis = axis * torch.stack((sx, sy, sz), dim=-1)

    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    w_pi = theta[..., None] * axis

    w = torch.where(small[..., None], w_small, w_general)
    w = torch.where(near_pi[..., None], w_pi, w)
    return w, theta


@torch.no_grad()
def get_median_rotations(
    rotations: torch.Tensor,                 # (B,W,3,3)
    times: int = 10,
    max_iters: int = 1000,
    eps: float = 1e-5,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched median rotation for input rotations (B,W,3,3), taking median over dim=1 (W).

    Returns:
      best_R: (B,1,3,3)
      final_angle_errors: (B,W)
    """
    R = torch.as_tensor(rotations).to(dtype=dtype)
    assert R.ndim == 4 and R.shape[-2:] == (3, 3), "rotations must be (B,W,3,3)"
    B, W = R.shape[0], R.shape[1]
    assert W > 0

    device = R.device
    eye3 = torch.eye(3, device=device, dtype=R.dtype)

    # ids: (B,times)  每个batch独立随机初值（允许重复，和C++一致）
    ids = torch.randint(0, W, (B, times), device=device, generator=generator)

    # gather initial R_med: (B,times,3,3)
    R_med = R.gather(
        dim=1,
        index=ids[..., None, None].expand(B, times, 3, 3),
    ).clone()

    # record median error for each candidate when it converges
    median_err = torch.full((B, times), float("inf"), device=device, dtype=R.dtype)
    converged = torch.zeros((B, times), device=device, dtype=torch.bool)

    # main loop
    for _ in range(max_iters):
        # residual_R: (B,times,W,3,3) = R[b,w] @ R_med[b,t]^T
        R_med_T = R_med.transpose(-1, -2)  # (B,times,3,3)
        residual_R = torch.einsum("bwij,btjk->btwik", R, R_med_T)

        residual_vec, angle_errors = _so3_log(residual_R)  # (B,times,W,3), (B,times,W)

        # median residual vector over W (dim=2): -> (B,times,3)
        med0 = _median_midpoint(residual_vec[..., 0], dim=2)
        med1 = _median_midpoint(residual_vec[..., 1], dim=2)
        med2 = _median_midpoint(residual_vec[..., 2], dim=2)
        med_vec = torch.stack((med0, med1, med2), dim=-1)  # (B,times,3)

        norm = torch.linalg.norm(med_vec, dim=-1)  # (B,times)
        newly_converged = (norm < eps) & (~converged)

        # 对 newly_converged 记录 score：median(angle_errors)
        if newly_converged.any():
            med_ang = _median_midpoint(angle_errors, dim=2)  # (B,times)
            median_err = torch.where(newly_converged, med_ang, median_err)
            converged = converged | newly_converged

        # 全部收敛则提前退出
        if converged.all():
            break

        # update only for not-yet-converged
        active = ~converged
        if active.any():
            dR = _so3_exp(med_vec, eye3=eye3)  # (B,times,3,3)
            R_med_new = torch.einsum("btij,btjk->btik", dR, R_med)
            R_med = torch.where(active[..., None, None], R_med_new, R_med)

    # choose best per batch among converged candidates
    min_err, best_idx = torch.min(median_err, dim=1)  # (B,), (B,)
    has_solution = torch.isfinite(min_err)

    best_R_cand = R_med[torch.arange(B, device=device), best_idx]  # (B,3,3)
    fallback_R = R[:, 0]  # (B,3,3)
    best_R = torch.where(has_solution[:, None, None], best_R_cand, fallback_R)

    # final angle errors for chosen best_R: (B,W)
    best_R_T = best_R.transpose(-1, -2)  # (B,3,3)
    residual_final = torch.einsum("bwij,bjk->bwik", R, best_R_T)  # (B,W,3,3)
    _, final_angle_errors = _so3_log(residual_final)  # (B,W)

    return best_R[:, None], final_angle_errors
