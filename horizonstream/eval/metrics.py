import numpy as np
from scipy.spatial import cKDTree


def similarity_align(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Expected Nx3 source and target point sets")
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var = np.mean(np.sum(src_centered ** 2, axis=1))
        scale = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    else:
        scale = 1.0
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def transform_points(points, scale, R, t):
    return (scale * (R @ points.T)).T + t[None]


def ate_rmse(pred_xyz, gt_xyz, align_scale=True):
    scale, R, t = similarity_align(pred_xyz, gt_xyz, with_scale=align_scale)
    pred_aligned = transform_points(pred_xyz, scale, R, t)
    err = np.linalg.norm(pred_aligned - gt_xyz, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean": float(np.mean(err)),
        "ate_median": float(np.median(err)),
        "num_pose_pairs": int(len(err)),
        "align_scale": bool(align_scale),
        "sim3_scale": float(scale),
        "sim3_rotation": R.tolist(),
        "sim3_translation": t.tolist(),
    }


def _rotation_angle_deg(R):
    tr = float(np.trace(R))
    cos = (tr - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _path_cumulative_lengths(xyz):
    if len(xyz) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xyz) == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])


def relative_pose_error_at_distances(pred_c2w, gt_c2w, distances=(1.0, 100.0)):
    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    if pred_c2w.shape != gt_c2w.shape or pred_c2w.ndim != 3 or pred_c2w.shape[1:] != (4, 4):
        raise ValueError("Expected pred_c2w and gt_c2w with shape (N,4,4)")
    n = pred_c2w.shape[0]
    if n < 2:
        return {}

    gt_xyz = gt_c2w[:, :3, 3]
    cum = _path_cumulative_lengths(gt_xyz)
    out = {}
    for d in distances:
        d = float(d)
        rre_vals = []
        rte_vals = []
        for i in range(n - 1):
            target = cum[i] + d
            j = int(np.searchsorted(cum, target, side="left"))
            if j >= n:
                continue
            gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[j]
            pred_rel = np.linalg.inv(pred_c2w[i]) @ pred_c2w[j]
            rel_err = np.linalg.inv(gt_rel) @ pred_rel
            rte_vals.append(float(np.linalg.norm(rel_err[:3, 3])))
            rre_vals.append(_rotation_angle_deg(rel_err[:3, :3]))
        if len(rte_vals) == 0:
            out[f"rte_{int(d)}m"] = None
            out[f"rre_{int(d)}m"] = None
            out[f"num_segments_{int(d)}m"] = 0
        else:
            out[f"rte_{int(d)}m"] = float(np.mean(rte_vals))
            out[f"rre_{int(d)}m"] = float(np.mean(rre_vals))
            out[f"num_segments_{int(d)}m"] = int(len(rte_vals))
    return out


def _voxel_downsample(points, voxel_size):
    if voxel_size is None:
        return points
    voxel_size = float(voxel_size)
    if voxel_size <= 0 or len(points) == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(coords, axis=0, return_index=True)
    keep.sort()
    return points[keep]


def _sample_points(points, max_points, seed):
    if max_points is None or len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=int(max_points), replace=False)
    return points[keep]


def prepare_pointcloud(points, max_points=None, voxel_size=None, seed=0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    points = _voxel_downsample(points, voxel_size)
    points = _sample_points(points, max_points, seed)
    return points


def chamfer_and_f1(
    pred_points, gt_points, threshold=0.25, max_points=None, voxel_size=None, seed=0
):
    pred = prepare_pointcloud(
        pred_points, max_points=max_points, voxel_size=voxel_size, seed=seed
    )
    gt = prepare_pointcloud(
        gt_points, max_points=max_points, voxel_size=voxel_size, seed=seed + 1
    )
    if len(pred) == 0 or len(gt) == 0:
        return None

    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    dist_pred_to_gt, _ = gt_tree.query(pred, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt, k=1)

    acc = float(np.mean(dist_pred_to_gt))
    comp = float(np.mean(dist_gt_to_pred))
    pred_norm = np.linalg.norm(pred, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    precision = float(np.mean(dist_pred_to_gt < threshold))
    recall = float(np.mean(dist_gt_to_pred < threshold))
    denom = precision + recall
    f1 = 0.0 if denom <= 0 else float(2.0 * precision * recall / denom)
    return {
        "cd": float((acc + comp) * 0.5),
        "acc": acc,
        "comp": comp,
        "f1": f1,
        "f1_threshold": float(threshold),
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
    }


def chamfer_and_f1_thresholds(
    pred_points,
    gt_points,
    thresholds=(0.25,),
    primary_threshold=0.25,
    max_points=None,
    voxel_size=None,
    seed=0,
):
    pred = prepare_pointcloud(
        pred_points, max_points=max_points, voxel_size=voxel_size, seed=seed
    )
    gt = prepare_pointcloud(
        gt_points, max_points=max_points, voxel_size=voxel_size, seed=seed + 1
    )
    if len(pred) == 0 or len(gt) == 0:
        return None

    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    dist_pred_to_gt, _ = gt_tree.query(pred, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt, k=1)

    acc = float(np.mean(dist_pred_to_gt))
    comp = float(np.mean(dist_gt_to_pred))
    pred_norm = np.linalg.norm(pred, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)

    f1_by_threshold = {}
    precision_by_threshold = {}
    recall_by_threshold = {}
    ordered_thresholds = []
    for threshold in thresholds:
        threshold = float(threshold)
        if threshold in ordered_thresholds:
            continue
        ordered_thresholds.append(threshold)
        precision = float(np.mean(dist_pred_to_gt < threshold))
        recall = float(np.mean(dist_gt_to_pred < threshold))
        denom = precision + recall
        f1 = 0.0 if denom <= 0 else float(2.0 * precision * recall / denom)
        key = f"{threshold:g}"
        f1_by_threshold[key] = f1
        precision_by_threshold[key] = precision
        recall_by_threshold[key] = recall

    primary_key = f"{float(primary_threshold):g}"
    if primary_key not in f1_by_threshold:
        precision = float(np.mean(dist_pred_to_gt < float(primary_threshold)))
        recall = float(np.mean(dist_gt_to_pred < float(primary_threshold)))
        denom = precision + recall
        f1_by_threshold[primary_key] = (
            0.0 if denom <= 0 else float(2.0 * precision * recall / denom)
        )
        precision_by_threshold[primary_key] = precision
        recall_by_threshold[primary_key] = recall

    return {
        "cd": float((acc + comp) * 0.5),
        "acc": acc,
        "comp": comp,
        "f1": f1_by_threshold[primary_key],
        "f1_threshold": float(primary_threshold),
        "f1_by_threshold": f1_by_threshold,
        "precision_by_threshold": precision_by_threshold,
        "recall_by_threshold": recall_by_threshold,
        "dist_pred_to_gt_percentiles": {
            str(q): float(np.percentile(dist_pred_to_gt, q))
            for q in (50, 90, 95, 99, 100)
        },
        "dist_gt_to_pred_percentiles": {
            str(q): float(np.percentile(dist_gt_to_pred, q))
            for q in (50, 90, 95, 99, 100)
        },
        "pred_bbox_min": pred.min(axis=0).tolist(),
        "pred_bbox_max": pred.max(axis=0).tolist(),
        "gt_bbox_min": gt.min(axis=0).tolist(),
        "gt_bbox_max": gt.max(axis=0).tolist(),
        "pred_norm_percentiles": {
            str(q): float(np.percentile(pred_norm, q)) for q in (50, 90, 99, 100)
        },
        "gt_norm_percentiles": {
            str(q): float(np.percentile(gt_norm, q)) for q in (50, 90, 99, 100)
        },
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
    }


def _compose_sim3(s1, R1, t1, s2, R2, t2):
    # Apply (s1,R1,t1) first, then (s2,R2,t2).
    s = float(s2) * float(s1)
    R = np.asarray(R2, dtype=np.float64) @ np.asarray(R1, dtype=np.float64)
    t = float(s2) * (np.asarray(R2, dtype=np.float64) @ np.asarray(t1, dtype=np.float64)) + np.asarray(t2, dtype=np.float64)
    return s, R, t


def align_pointcloud_umeyama_icp(
    pred_points,
    gt_points,
    *,
    coarse=True,
    icp=True,
    align_sample_points=50000,
    icp_max_iters=20,
    icp_max_corr_dist=0.5,
):
    pred = prepare_pointcloud(pred_points, max_points=align_sample_points, voxel_size=None, seed=0)
    gt = prepare_pointcloud(gt_points, max_points=align_sample_points, voxel_size=None, seed=1)
    if len(pred) == 0 or len(gt) == 0:
        return pred_points, {
            "coarse_enabled": bool(coarse),
            "icp_enabled": bool(icp),
            "coarse_applied": False,
            "icp_applied": False,
            "icp_iters": 0,
        }

    s_total = 1.0
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)

    pred_work = pred.copy()
    coarse_applied = False
    if coarse:
        gt_tree = cKDTree(gt)
        _, nn_idx = gt_tree.query(pred_work, k=1)
        matched_gt = gt[nn_idx]
        s_c, R_c, t_c = similarity_align(pred_work, matched_gt, with_scale=True)
        pred_work = transform_points(pred_work, s_c, R_c, t_c)
        s_total, R_total, t_total = _compose_sim3(s_total, R_total, t_total, s_c, R_c, t_c)
        coarse_applied = True

    icp_applied = False
    iters_done = 0
    if icp:
        gt_tree = cKDTree(gt)
        for _ in range(max(int(icp_max_iters), 0)):
            dists, nn_idx = gt_tree.query(pred_work, k=1)
            inlier = dists < float(icp_max_corr_dist)
            if np.count_nonzero(inlier) < 10:
                break
            src = pred_work[inlier]
            dst = gt[nn_idx[inlier]]
            s_i, R_i, t_i = similarity_align(src, dst, with_scale=False)
            pred_work = transform_points(pred_work, s_i, R_i, t_i)
            s_total, R_total, t_total = _compose_sim3(s_total, R_total, t_total, s_i, R_i, t_i)
            iters_done += 1
        icp_applied = iters_done > 0

    pred_aligned = transform_points(
        np.asarray(pred_points, dtype=np.float64).reshape(-1, 3),
        s_total,
        R_total,
        t_total,
    )
    return pred_aligned.astype(np.float32, copy=False), {
        "coarse_enabled": bool(coarse),
        "icp_enabled": bool(icp),
        "coarse_applied": bool(coarse_applied),
        "icp_applied": bool(icp_applied),
        "icp_iters": int(iters_done),
        "sim3_scale": float(s_total),
        "sim3_rotation": np.asarray(R_total, dtype=np.float64).tolist(),
        "sim3_translation": np.asarray(t_total, dtype=np.float64).tolist(),
    }
