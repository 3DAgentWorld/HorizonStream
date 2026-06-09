import torch
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial.transform import Rotation as R

try:
    from sksparse.cholmod import cholesky, analyze  # type: ignore
    HAS_CHOLMOD = True
except ImportError:
    cholesky = None
    analyze = None
    HAS_CHOLMOD = False


def _build_sparse_solver(A: sparse.csc_array):
    A = A.tocsc()
    if HAS_CHOLMOD:
        try:
            return analyze(A, ordering_method="amd").cholesky(A)
        except Exception:
            return cholesky(A, mode="simplicial")
    return factorized(A)

def _shrink_np(x: np.ndarray, kappa: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

def l1_lad_admm_cholmod(
    A: sparse.csc_array,   # (m,n) sparse, guaranteed CSC array
    b: np.ndarray,         # (m,)
    rho: float = 1.0,
    max_iter: int = 500,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-4,
):
    """
    Solve: min_x ||A x - b||_1 via ADMM
    x-step: (A^T A) x = A^T (b + z - u) using sparse Cholesky (CHOLMOD).

    Inputs:
      A: csc_array (m,n)
      b: ndarray (m,)

    Returns:
      x: (n,) float64
      info: dict
    """
    # --- validate b only (A is assumed correct) ---
    b = np.asarray(b)
    if b.ndim != 1:
        raise ValueError("b must be shape (m,).")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"b must have length m={m}, got {b.shape[0]}")

    # --- ensure float64 for CHOLMOD / stable norms ---
    if b.dtype != np.float64:
        b = b.astype(np.float64, copy=False)

    At = A.T  # reuse

    # --- factor once (AtA fixed) ---
    # Note: At @ A is (n,n) sparse
    AtA = (At @ A).tocsc()

    Fun = _build_sparse_solver(AtA)

    # --- ADMM variables (float64) ---
    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(m, dtype=np.float64)
    u = np.zeros(m, dtype=np.float64)

    # --- iterate ---
    for it in range(1, max_iter + 1):
        # x-update: rhs = A^T (b + z - u)
        y = b + z - u                 # (m,)
        rhs = At @ y                  # (n,)
        x = Fun(rhs)                    # solve (AtA)x = rhs

        # primal residual ingredients
        Ax_minus_b = (A @ x) - b      # (m,)

        # z-update
        z_old = z
        z = _shrink_np(Ax_minus_b + u, 1.0 / rho)

        # u-update
        r = Ax_minus_b - z
        u = u + r

        # dual residual: s = rho * A^T (z - z_old)
        dz = z - z_old
        s = rho * (At @ dz)

        # norms
        r_norm = np.linalg.norm(r)
        s_norm = np.linalg.norm(s)

        Ax_norm = np.linalg.norm(Ax_minus_b)
        z_norm  = np.linalg.norm(z)

        eps_pri = np.sqrt(m) * abs_tol + rel_tol * max(Ax_norm, z_norm)

        At_u = At @ u
        eps_dual = np.sqrt(n) * abs_tol + rel_tol * (rho * np.linalg.norm(At_u))

        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            break

    # objective
    obj = float(np.sum(np.abs((A @ x) - b)))

    info = {
        "iters": it,
        "objective": obj,
        "rho": float(rho),
        "primal_residual_norm": float(r_norm),
        "dual_residual_norm": float(s_norm),
    }

    return x, info

def rotation_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    device = matrix.device
    dtype = matrix.dtype
    orig_shape = matrix.shape[:-2]
    matrix_flat = matrix.view(-1, 3, 3).detach().cpu().numpy()
    rot_vecs_flat = R.from_matrix(matrix_flat).as_rotvec()
    return torch.from_numpy(rot_vecs_flat).to(device=device, dtype=dtype).view(*orig_shape, 3)

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    device = axis_angle.device
    dtype = axis_angle.dtype
    orig_shape = axis_angle.shape[:-1]
    axis_angle_flat = axis_angle.view(-1, 3).detach().cpu().numpy()
    matrices_flat = R.from_rotvec(axis_angle_flat).as_matrix()
    return torch.from_numpy(matrices_flat).to(device=device, dtype=dtype).view(*orig_shape, 3, 3)

def setup_linear_system(Sli: int, Win: int, F: int, fix_index: int) -> sparse.csc_array:
    """
    Build a sparse matrix of shape (rows_num, cols_num) in CSC.
    - Sli == F - (Win - 1)
    Returns:
    A: csc_array (3+3*Sli*(Win-1), 3*F) with int32 data (+1/-1)
    """
    assert Sli == F - (Win - 1)
    K = Sli * (Win - 1)          # number of (sli,win-1) pairs
    rows_num = 3 + 3 * K
    cols_num = 3 * F

    nnz = 3 + 6 * K              # exactly how many nonzeros we create
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)

    # ---- 1) First 3 rows: identity on the first 3 cols ----
    rows[:3] = np.arange(3, dtype=np.int32)
    cols[:3] = np.arange(3, dtype=np.int32) + fix_index * 3
    data[:3] = 1

    # Helper: generate base indices for all (s, w) pairs, flattened length K
    # order: s-major then w (or vice versa) doesn't matter as long as consistent
    s = np.arange(Sli, dtype=np.int32)              # (Sli,)
    w = np.arange(Win - 1, dtype=np.int32)          # (Win-1,)
    ss = np.repeat(s, Win - 1)                      # (K,) (000,111,222,...)
    ww = np.tile(w, Sli)                            # (K,) (012,012,012,...)

    # For each pair we create 3 columns (x,y,z) -> add [0,1,2]
    base3 = (3 * ss + 3 * ww)                       # (K,)   head base
    xyz = np.array([0, 1, 2], dtype=np.int32)

    # ---- 2) Head block (+1): rows 3 .. 3+3*K-1 ----
    head_start = 3
    head_end = 3 + 3 * K

    # rows: contiguous
    rows[head_start:head_end] = np.arange(head_start, head_end, dtype=np.int32)

    # cols: base3 repeated for xyz
    cols_head = (base3[:, None] + xyz[None, :]).reshape(-1)    # (3K,)
    cols[head_start:head_end] = cols_head

    data[head_start:head_end] = 1

    # ---- 3) Tail block (-1): rows head_end .. head_end+3*K-1 ----
    tail_start = head_end
    tail_end = tail_start + 3 * K

    rows[tail_start:tail_end] = np.arange(head_start, head_end, dtype=np.int32)

    # your original tail base was: + 3*(Win-1) plus 3*sli
    base3_tail = 3 * ss + 3 * (Win - 1)
    cols_tail = (base3_tail[:, None] + xyz[None, :]).reshape(-1)  # (3K,)
    cols[tail_start:tail_end] = cols_tail

    data[tail_start:tail_end] = -1

    # ---- sanity ----
    assert tail_end == nnz
    assert rows.max() < rows_num
    assert cols.max() < cols_num

    return sparse.coo_matrix((data, (rows, cols)), shape=(rows_num, cols_num), dtype=np.float64).tocsc()

def compute_residual_rotations(abs_rotations: torch.Tensor, rel_rotations: torch.Tensor, Win: int, fix_index: int) -> torch.Tensor:
    """
    Input:
    abs_rotations: (F,3,3)
    rel_rotations: (Sli,Win-1,3,3)
    Returns:
    res_axis_angles: (3+Sli*(Win-1)*3)
    """
    sli_abs_R = abs_rotations.unfold(dimension=0, size=Win, step=1) # (Sli,3,3,Win)
    sli_abs_R = sli_abs_R.permute(0, 3, 1, 2) # (Sli,Win,3,3)
    inv_abs_rotations_i = sli_abs_R[:, :Win - 1].mT # (Sli,Win-1,3,3)
    abs_rotations_j = sli_abs_R[:, -1:] # (Sli,1,3,3)
    res_rotations = inv_abs_rotations_i @ rel_rotations @ abs_rotations_j # (Sli,Win-1,3,3)
    res_axis_angles = rotation_matrix_to_axis_angle(res_rotations) # (Sli,Win-1,3)
    fix_res_axis_angles = rotation_matrix_to_axis_angle(abs_rotations[fix_index].mT) # (3)
    res_axis_angles = torch.cat([fix_res_axis_angles, res_axis_angles.reshape(-1)], dim=0) # (3+Sli*(Win-1)*3)
    # print(res_axis_angles.reshape(-1, 3).norm(dim=-1))
    return res_axis_angles

def solve_delta_axis_angles_l1(A: sparse.csc_array, res_axis_angles: torch.Tensor) -> torch.Tensor:
    """
    Input:
    A: (3+3*Sli*(Win-1), 3*F)
    res_axis_angles: (3+3*Sli*(Win-1))
    Returns:
    delta_axis_angles: (F,3)
    """
    device = res_axis_angles.device
    res_axis_angles_np = res_axis_angles.detach().cpu().numpy() 
    delta_axis_angles = l1_lad_admm_cholmod(A, res_axis_angles_np, max_iter=1000)[0].reshape(-1, 3)
    return torch.from_numpy(delta_axis_angles).to(device=device, dtype=torch.float32) # (F,3)

def solve_delta_axis_angles_irls(A: sparse.csc_array, AT: sparse.csc_array, res_axis_angles: torch.Tensor, sigma_degree) -> torch.Tensor:
    """
    Input:
    A: (3+3*Sli*(Win-1), 3*F)
    AT: (3*F,3+3*Sli*(Win-1))
    res_axis_angles: (3+3*Sli*(Win-1))
    Returns:
    delta_axis_angles: (F,3)
    """
    device = res_axis_angles.device
    res_axis_angles_np = res_axis_angles.detach().cpu().numpy()
    sigma_rad_square = math.radians(sigma_degree) ** 2
    
    err_sq = (res_axis_angles_np.reshape(-1, 3) ** 2).sum(axis=-1, keepdims=True)  # (1+Sli*(Win-1),1)
    # weight = np.minimum(err_sq ** (-0.75), 1e8)   # l_0.5: (1+Sli*(Win-1),1)
    weight = sigma_rad_square / (sigma_rad_square + err_sq) ** 2 # GM: (1+Sli*(Win-1),1)
    # weight = sigma_rad_square / (sigma_rad_square + err_sq) # Cauchy: (1+Sli*(Win-1),1)
    weight = np.repeat(weight, 3, axis=1).ravel() # (3+3*Sli*(Win-1),)
    AT_weight = AT.multiply(weight) # not use * for csc matrix
    H = (AT_weight @ A).tocsc()
    solve = _build_sparse_solver(H)
    delta_axis_angles = solve(AT_weight @ res_axis_angles_np).reshape(-1, 3)

    return torch.from_numpy(delta_axis_angles).to(device=device, dtype=torch.float32) # (F,3)

@torch.no_grad()
def rotation_averaging(
    valid_relative_rotations: torch.Tensor,                 # (F-(Win-1),Win,3,3)
    online_absolute_rotations: torch.Tensor,               # (F,3,3)
    max_iters: int = 100,
) -> torch.Tensor:
    Sli, Win = valid_relative_rotations.shape[:2]
    F = online_absolute_rotations.shape[0] # total frames number
    assert Sli == F - (Win -1)

    fix_index = Win - 1
    
    A = setup_linear_system(Sli, Win, F, fix_index)

    rel_rotations = valid_relative_rotations[:, :(Win-1)] # (Sli,Win-1,3,3)
    abs_rotations = online_absolute_rotations @ online_absolute_rotations[fix_index].mT # (F,3,3)

    # L1 Solve
    # for _ in range(5):
    #     res_axis_angles = compute_residual_rotations(abs_rotations, rel_rotations, Win) # (3+Sli*(Win-1)*3)
    #     delta_axis_angles = solve_delta_axis_angles_l1(A, res_axis_angles)
    #     average_delta_angles = delta_axis_angles.norm(dim=-1).mean(dim=0)
    #     delta_rotations = axis_angle_to_rotation_matrix(delta_axis_angles) # (F, 3, 3)
    #     abs_rotations = abs_rotations @ delta_rotations
    #     if average_delta_angles < 1e-4: break
    #     else: print(f"l1 iter: {_}, average delta angles: {average_delta_angles}")

    # # irls Solve
    AT = A.T.tocsc()
    for _ in range(max_iters):
        res_axis_angles = compute_residual_rotations(abs_rotations, rel_rotations, Win, fix_index) # (3+Sli*(Win-1)*3)
        delta_axis_angles = solve_delta_axis_angles_irls(A, AT, res_axis_angles, sigma_degree=5.0)
        average_delta_angles = delta_axis_angles.norm(dim=-1).mean(dim=0)
        delta_rotations = axis_angle_to_rotation_matrix(delta_axis_angles) # (F, 3, 3)
        abs_rotations = abs_rotations @ delta_rotations
        if average_delta_angles < 1e-6:
            break

    return abs_rotations # (F, 3, 3)
