from typing import Callable, Optional, Sequence

import numpy as np
import pypose as pp
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation


def _matrix_to_se3_data(mat: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()
    return np.concatenate([mat[:3, 3], quat], axis=0).astype(np.float32, copy=False)


def _project_to_rotation(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(mat)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = u @ vt
    return rot.astype(np.float64, copy=False)


def _matrix_to_sim3_components(mat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(mat[:3, :3], dtype=np.float64)
    trans = np.asarray(mat[:3, 3], dtype=np.float64)
    col_norms = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(col_norms))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    rot = _project_to_rotation(linear / scale)
    return scale, rot, trans


def _matrix_to_sim3_data(mat: np.ndarray) -> np.ndarray:
    scale, rot, trans = _matrix_to_sim3_components(mat)
    quat = Rotation.from_matrix(rot).as_quat()
    return np.concatenate([trans, quat, [scale]], axis=0).astype(np.float32, copy=False)


def _se3_to_matrix(se3: pp.LieTensor) -> np.ndarray:
    return se3.matrix().detach().cpu().numpy().astype(np.float64, copy=False)


def _sim3_to_matrix(sim3: pp.LieTensor, normalize_rotation: bool = False) -> np.ndarray:
    data = sim3.data.detach().cpu().numpy().astype(np.float64, copy=False)
    trans = data[:3]
    quat = data[3:7]
    scale = float(data[7])
    rot = Rotation.from_quat(quat).as_matrix().astype(np.float64, copy=False)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot if normalize_rotation else (scale * rot)
    out[:3, 3] = trans
    return out


def _solve_sparse_system(
    j_i: torch.Tensor,
    j_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    resid: torch.Tensor,
    lm: float,
    active_components: Optional[np.ndarray] = None,
) -> torch.Tensor:
    device = resid.device
    j_i = j_i.detach().cpu().numpy().astype(np.float64, copy=False)
    j_j = j_j.detach().cpu().numpy().astype(np.float64, copy=False)
    ii = ii.detach().cpu().numpy().astype(np.int64, copy=False)
    jj = jj.detach().cpu().numpy().astype(np.int64, copy=False)
    resid = resid.detach().cpu().numpy().astype(np.float64, copy=False)

    num_edges = resid.shape[0]
    dim = resid.shape[1]
    num_nodes = int(max(ii.max(), jj.max())) + 1 if num_edges > 0 else 0

    rows = []
    cols = []
    data = []
    for edge_idx in range(num_edges):
        src = ii[edge_idx]
        dst = jj[edge_idx]
        for r in range(dim):
            row = edge_idx * dim + r
            col_src = src * dim
            col_dst = dst * dim
            for c in range(dim):
                rows.append(row)
                cols.append(col_src + c)
                data.append(j_i[edge_idx, r, c])
                rows.append(row)
                cols.append(col_dst + c)
                data.append(j_j[edge_idx, r, c])

    J = coo_matrix((data, (rows, cols)), shape=(num_edges * dim, num_nodes * dim)).tocsc()
    resid_vec = resid.reshape(-1)
    b = -(J.T @ resid_vec)
    A = J.T @ J
    diag = A.diagonal()
    A.setdiag(diag * (1.0 + lm))

    # Fix the first node as gauge anchor.
    delta = np.zeros(num_nodes * dim, dtype=np.float64)
    if active_components is None:
        active_components = np.ones(dim, dtype=bool)
    active_components = np.asarray(active_components, dtype=bool).reshape(dim)
    active_cols = []
    for node_idx in range(1, num_nodes):
        base = node_idx * dim
        active_cols.extend((base + np.flatnonzero(active_components)).tolist())
    if active_cols:
        active_cols = np.asarray(active_cols, dtype=np.int64)
        A_free = A[active_cols][:, active_cols].tocsc()
        b_free = b[active_cols]
        delta_free = spsolve(A_free, b_free)
        delta[active_cols] = delta_free
    return torch.from_numpy(delta.astype(np.float32)).view(num_nodes, dim).to(device)


def _batch_jacobian(
    func: Callable[[pp.LieTensor, torch.Tensor, torch.Tensor], torch.Tensor],
    constants: pp.LieTensor,
    gi: torch.Tensor,
    gj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _func_sum(c, x, y):
        return func(c, x, y).sum(dim=0)

    jac = torch.autograd.functional.jacobian(
        _func_sum, (constants, gi, gj), vectorize=True
    )
    j_i = jac[1].permute(1, 0, 2).contiguous()
    j_j = jac[2].permute(1, 0, 2).contiguous()
    return j_i, j_j


def optimize_pose_graph_pypose(
    keyframe_c2w_init: np.ndarray,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    edge_measurements: Sequence[np.ndarray],
    edge_weights: Sequence[float],
    model: str = "se3",
    update_mode: str = "all",
    trans_weight: float = 1.0,
    rot_weight: float = 1.0,
    scale_weight: float = 1.0,
    max_iterations: int = 30,
    lambda_init: float = 1e-4,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    def log(msg: str) -> None:
        if verbose and log_fn is not None:
            log_fn(msg)

    if len(keyframe_c2w_init) <= 1:
        return keyframe_c2w_init.copy()
    if len(edge_measurements) == 0:
        return keyframe_c2w_init.copy()
    if model not in {"se3", "sim3"}:
        raise ValueError(f"Unsupported pose graph model: {model}")

    if model == "sim3":
        valid_update_modes = {"all", "translation_only", "translation_scale"}
        dim = 7
    else:
        valid_update_modes = {"all", "translation_only"}
        dim = 6
    if update_mode not in valid_update_modes:
        raise ValueError(
            f"Unsupported update_mode='{update_mode}' for model='{model}'. "
            f"Valid modes: {sorted(valid_update_modes)}"
        )
    active_components = np.zeros(dim, dtype=bool)
    if update_mode == "all":
        active_components[:] = True
    elif update_mode == "translation_only":
        active_components[:3] = True
    elif update_mode == "translation_scale":
        active_components[:3] = True
        active_components[6] = True

    if model == "sim3":
        pose_data = np.stack([_matrix_to_sim3_data(pose) for pose in keyframe_c2w_init], axis=0)
        meas_data = np.stack([_matrix_to_sim3_data(meas) for meas in edge_measurements], axis=0)
    else:
        pose_data = np.stack([_matrix_to_se3_data(pose) for pose in keyframe_c2w_init], axis=0)
        meas_data = np.stack([_matrix_to_se3_data(meas) for meas in edge_measurements], axis=0)
    device = torch.device("cpu")

    if model == "sim3":
        poses = pp.Sim3(torch.from_numpy(pose_data).to(device))
        constants = pp.Sim3(torch.from_numpy(meas_data).to(device))
    else:
        poses = pp.SE3(torch.from_numpy(pose_data).to(device))
        constants = pp.SE3(torch.from_numpy(meas_data).to(device))
    ii = torch.as_tensor(edge_src, dtype=torch.long, device=device)
    jj = torch.as_tensor(edge_dst, dtype=torch.long, device=device)
    weights = torch.as_tensor(edge_weights, dtype=torch.float32, device=device).view(-1, 1)

    Ginv = poses.Inv().Log().detach()
    lm = float(lambda_init)
    residual_history: list[float] = []

    def residual_fn(C: pp.LieTensor, Gi: torch.Tensor, Gj: torch.Tensor) -> torch.Tensor:
        out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
        resid = out.Log().tensor()
        resid = resid.clone()
        resid[:, :3] *= float(trans_weight)
        resid[:, 3:6] *= float(rot_weight)
        if model == "sim3":
            resid[:, 6:7] *= float(scale_weight)
        return resid

    def evaluate(current_ginv: torch.Tensor) -> torch.Tensor:
        resid = residual_fn(constants, current_ginv[ii], current_ginv[jj])
        return resid * weights

    log(
        f"PyPose pose graph ({model}, {update_mode}): {len(keyframe_c2w_init)} nodes, {len(edge_measurements)} edges, "
        f"weights=(t={trans_weight:g}, r={rot_weight:g}, s={scale_weight:g}), "
        f"max_iterations={max_iterations}, lambda_init={lambda_init:g}"
    )
    for itr in range(max_iterations):
        resid_unweighted = residual_fn(constants, Ginv[ii], Ginv[jj])
        j_i, j_j = _batch_jacobian(residual_fn, constants, Ginv[ii], Ginv[jj])
        j_i = j_i * weights.unsqueeze(-1)
        j_j = j_j * weights.unsqueeze(-1)
        resid = resid_unweighted * weights

        current_cost = float(resid.square().mean().item())
        residual_history.append(current_cost)
        delta = _solve_sparse_system(j_i, j_j, ii, jj, resid, lm, active_components=active_components)
        candidate = Ginv + delta

        new_resid = evaluate(candidate)
        new_cost = float(new_resid.square().mean().item())
        accepted = new_cost <= current_cost + 1e-12
        if accepted:
            Ginv = candidate
            lm = max(lm / 2.0, 1e-12)
        else:
            lm = min(lm * 2.0, 1e8)

        if verbose:
            status = "accepted" if accepted else "rejected"
            log(
                f"PyPose iter {itr + 1}/{max_iterations}: cost {current_cost:.8f} -> "
                f"{new_cost:.8f} ({status}, lambda={lm:.3e})"
            )

        if current_cost < 1e-6 and itr >= 4 and len(residual_history) >= 5:
            denom = max(residual_history[-1], 1e-12)
            improvement = residual_history[-5] / denom
            if improvement < 1.2:
                log(f"PyPose pose graph: converged at iter {itr + 1}")
                break

    optimized = pp.Exp(Ginv).Inv()
    if model == "sim3":
        out = np.stack(
            [_sim3_to_matrix(optimized[idx], normalize_rotation=False) for idx in range(len(keyframe_c2w_init))],
            axis=0,
        )
    else:
        out = np.stack([_se3_to_matrix(optimized[idx]) for idx in range(len(keyframe_c2w_init))], axis=0)
    return out
