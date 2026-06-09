import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
import torch

try:
    from sksparse.cholmod import analyze, cholesky  # type: ignore
    HAS_CHOLMOD = True
except ImportError:
    analyze = None
    cholesky = None
    HAS_CHOLMOD = False


def _build_sparse_solver(A: sparse.csc_array):
    A = A.tocsc()
    if HAS_CHOLMOD:
        try:
            return analyze(A).cholesky(A)
        except Exception:
            return cholesky(A)
    return factorized(A)

def constrained_l1_admm_cholmod(
    A: sparse.csc_array,
    b: np.ndarray,
    num_residuals: int,
    num_constraints: int,
    max_iter: int = 1000,
) -> np.ndarray:
    # -------- defaults --------
    absolute_tolerance = 1e-4
    relative_tolerance = 1e-2

    rho = 10.0
    rho_min = 1e-4
    rho_max = 1e4
    omega = 1.0
    omega_scale = float(2 ** (-0.01))
    # -------------------------

    # NOTE: issparse supports both sparse matrix and sparse array
    if not sparse.issparse(A):
        raise TypeError("A must be a scipy sparse matrix/array.")
    A = A.tocsc()

    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if m != b.shape[0]:
        raise ValueError(f"A has {m} rows but b has length {b.shape[0]}")
    if num_residuals + num_constraints != m:
        raise ValueError(
            f"num_residuals + num_constraints must equal A.rows, "
            f"got {num_residuals} + {num_constraints} != {m}"
        )

    # Pre-factorize AtA
    At = A.T  # let scipy decide CSR/CSC; At @ vec is fine
    AtA = (At @ A).tocsc()
    factor = _build_sparse_solver(AtA)

    # Allocate
    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(m, dtype=np.float64)
    z_old = np.zeros(m, dtype=np.float64)
    u = np.zeros(m, dtype=np.float64)
    z_plus_b = b.copy()  # z=0 initially

    rhs_norm = np.linalg.norm(b)
    primal_abs_eps = np.sqrt(m) * absolute_tolerance
    dual_abs_eps = np.sqrt(n) * absolute_tolerance

    def soft_threshold(xv: np.ndarray, k: float) -> np.ndarray:
        return np.maximum(xv - k, 0.0) - np.maximum(-xv - k, 0.0)

    for i in range(max_iter + 1):
        # x-update: x = (AtA)^-1 At (z_plus_b - u)
        rhs = At @ (z_plus_b - u)
        x = factor(rhs)

        a_times_x = A @ x
        residual = a_times_x - b + u

        # save z_old for dual residual
        z_old[:] = z

        # z-update
        z[:num_residuals] = soft_threshold(residual[:num_residuals], 1.0 / rho)
        z[num_residuals:] = np.maximum(residual[num_residuals:], 0.0)
        z_plus_b = z + b

        # primal residual r = Ax - (z + b)
        r = a_times_x - z_plus_b

        # u-update
        u += r

        # convergence check every 20 iters
        if (i + 1) % 20 == 0:
            # primal
            r_norm = np.linalg.norm(r)
            a_norm = np.linalg.norm(a_times_x)
            zb_norm = np.linalg.norm(z_plus_b)
            primal_eps = primal_abs_eps + relative_tolerance * max(a_norm, zb_norm, rhs_norm)

            # dual: s = rho * At (z - z_old)
            dz = z - z_old
            s = rho * (At @ dz)
            s_norm = np.linalg.norm(s)

            # dual tolerance: sqrt(n)*abs_tol + rel_tol * ||At u||
            Atu = At @ u
            dual_eps = dual_abs_eps + relative_tolerance * np.linalg.norm(Atu)

            if (r_norm < primal_eps) and (s_norm < dual_eps):
                break

        # adaptive rho (same as your code)
        z_norm = np.linalg.norm(z)
        u_norm = np.linalg.norm(u)
        denom = max(z_norm, 1e-12)
        new_rho = np.clip((u_norm / denom) * rho, rho_min, rho_max)
        rho = (1.0 - omega) * rho + omega * new_rho
        omega *= omega_scale

    return x

def setup_linear_system(F, Sli, Win, global_rel_T, fix_index: int):
    cols_num = 3 * F + Sli
    pairs_num = Sli * (Win - 1)
    num_residuals = 3 + 3 * pairs_num
    num_constraints = Sli
    rows_num = num_residuals + num_constraints
    
    rows = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.int32)
    cols = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.int32)
    data = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.float64)
    
    constrains_index = 3 + 9 * pairs_num
    
    rows[:3] = [0, 1, 2]
    cols[:3] = [3 * fix_index + 0, 3 * fix_index + 1, 3 * fix_index + 2]
    data[:3] = 1
    
    for i in range(Sli):
        rows[constrains_index + i] = num_residuals + i
        cols[constrains_index + i] = 3 * F + i
        data[constrains_index + i] = 1
        for j in range(Win - 1):
            index = 3 + 9 * (i * (Win - 1) + j)
            row_start = 3 + 3 * (i * (Win - 1) + j)
            col_start1 = 3 * (i + j)
            col_start2 = 3 * (i + Win - 1)
            translation = global_rel_T[i, j].detach().cpu().numpy() 
            for k in range(3):
                rows[index + 0 + k] = row_start + k
                cols[index + 0 + k] = col_start1 + k
                data[index + 0 + k] = 1
                rows[index + 3 + k] = row_start + k
                cols[index + 3 + k] = col_start2 + k
                data[index + 3 + k] = -1
                rows[index + 6 + k] = row_start + k
                cols[index + 6 + k] = 3 * F + i
                data[index + 6 + k] = translation[k]
        
    b = np.zeros(rows_num, np.float64)
    b[-Sli:] = 1
    return sparse.coo_matrix((data, (rows, cols)), shape=(rows_num, cols_num), dtype=np.float64).tocsc(), b, num_residuals, num_constraints

@torch.no_grad()
def position_averaging(rel_T: torch.Tensor, abs_R: torch.Tensor):
    """
    Input:
    rel_T: (Sli,Win,3,1)
    abs_R: (F,3,3)
    Returns:
    positions: (F,3)
    scales: (Sli,)
    """
    Sli, Win = rel_T.shape[:2]
    F = abs_R.shape[0] # total frames number
    assert Sli == F - (Win -1)
    fix_index = Win - 1
    
    sli_abs_R = abs_R.unfold(dimension=0, size=Win, step=1) # (Sli,3,3,Win)
    assert sli_abs_R.shape == (Sli,3,3,Win)
    sli_abs_R = sli_abs_R.permute(0, 3, 1, 2) # (Sli,Win,3,3)
    global_rel_T = (sli_abs_R.mT @ rel_T)[:, :Win - 1, :, 0] # (Sli,Win-1,3)
    device = global_rel_T.device
    
    positions_and_scales = constrained_l1_admm_cholmod(*setup_linear_system(F, Sli, Win, global_rel_T, fix_index))
    positions_and_scales = torch.from_numpy(positions_and_scales).to(device=device, dtype=torch.float32)
    positions = positions_and_scales[:3 * F].reshape(F, 3)
    scales = positions_and_scales[-Sli:]
    return positions, scales  
         
@torch.no_grad()
def l1_lad_admm_cholmod_no_reg(
    A,                  # scipy.sparse.csc_matrix, shape (m, n)
    b,                  # numpy.ndarray, shape (m,)
    rho: float = 1.0,
    max_iter: int = 1000,
):
    """
    Solve: min_x ||A x - b||_1
    ADMM, x-step solved by CHOLMOD.

    Returns:
      x:    (n,)
    """
    At = A.T
    AtA = (At @ A).tocsc()

    # old sksparse.cholmod style:
    F = _build_sparse_solver(AtA)

    def shrink(x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

    m, n = A.shape
    x = np.zeros(n, dtype=b.dtype)
    z = np.zeros(m, dtype=b.dtype)
    u = np.zeros(m, dtype=b.dtype)

    for it in range(max_iter):
        # x-update: solve (A^T A) x = A^T (b + z - u)
        y = b + z - u
        rhs = At @ y
        x = F(rhs)

        # z-update
        Ax_minus_b = A @ x - b
        z = shrink(Ax_minus_b + u, 1.0 / rho)

        # u-update
        u = u + (Ax_minus_b - z)

    return x

def setup_linear_system_legacy(F, Sli, Win, global_rel_T, fix_index: int):
    cols_num = 3 * F + Sli
    pairs_num = Sli * (Win - 1)
    num_residuals = 3 + 3 * pairs_num
    num_constraints = Sli
    rows_num = num_residuals + num_constraints
    
    rows = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.int32)
    cols = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.int32)
    data = np.zeros(3 + 9 * pairs_num + Sli, dtype=np.float64)
    
    constrains_index = 3 + 9 * pairs_num
    
    rows[:3] = [0, 1, 2]
    cols[:3] = [3 * fix_index + 0, 3 * fix_index + 1, 3 * fix_index + 2]
    data[:3] = 1
    
    for i in range(Sli):
        rows[constrains_index + i] = num_residuals + i
        cols[constrains_index + i] = 3 * F + i
        data[constrains_index + i] = 100
        for j in range(Win - 1):
            index = 3 + 9 * (i * (Win - 1) + j)
            row_start = 3 + 3 * (i * (Win - 1) + j)
            col_start1 = 3 * (i + j)
            col_start2 = 3 * (i + Win - 1)
            translation = global_rel_T[i, j].detach().cpu().numpy() 
            for k in range(3):
                rows[index + 0 + k] = row_start + k
                cols[index + 0 + k] = col_start1 + k
                data[index + 0 + k] = 1
                rows[index + 3 + k] = row_start + k
                cols[index + 3 + k] = col_start2 + k
                data[index + 3 + k] = -1
                rows[index + 6 + k] = row_start + k
                cols[index + 6 + k] = 3 * F + i
                data[index + 6 + k] = translation[k]
        
    b = np.zeros(rows_num, np.float64)
    b[-Sli:] = 100
    return sparse.coo_matrix((data, (rows, cols)), shape=(rows_num, cols_num), dtype=np.float64).tocsc(), b

@torch.no_grad()
def position_averaging_legacy(rel_T: torch.Tensor, abs_R: torch.Tensor):
    """
    Input:
    rel_T: (Sli,Win,3,1)
    abs_R: (F,3,3)
    Returns:
    positions: (F,3)
    scales: (Sli,)
    """
    Sli, Win = rel_T.shape[:2]
    F = abs_R.shape[0] # total frames number
    assert Sli == F - (Win -1)
    fix_index = Win - 1
    
    sli_abs_R = abs_R.unfold(dimension=0, size=Win, step=1) # (Sli,3,3,Win)
    assert sli_abs_R.shape == (Sli,3,3,Win)
    sli_abs_R = sli_abs_R.permute(0, 3, 1, 2) # (Sli,Win,3,3)
    global_rel_T = (sli_abs_R.mT @ rel_T)[:, :Win - 1, :, 0] # (Sli,Win-1,3)
    device = global_rel_T.device
    
    positions_and_scales = l1_lad_admm_cholmod_no_reg(*setup_linear_system_legacy(F, Sli, Win, global_rel_T, fix_index))
    positions_and_scales = torch.from_numpy(positions_and_scales).to(device=device, dtype=torch.float32)
    positions = positions_and_scales[:3 * F].reshape(F, 3)
    scales = positions_and_scales[-Sli:]
    return positions, scales  

         
