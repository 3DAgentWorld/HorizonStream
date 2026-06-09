from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from horizonstream.eval.io import frame_stems, read_opencv_camera_yml


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] not in [(3, 4), (4, 4)]:
        raise ValueError(f"Expected (N,3,4) or (N,4,4), got {w2c.shape}")
    if w2c.shape[-2:] == (4, 4):
        w2c = w2c[:, :3, :4]

    c2w = np.empty((w2c.shape[0], 4, 4), dtype=np.float64)
    c2w[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for i, mat in enumerate(w2c):
        full = np.eye(4, dtype=np.float64)
        full[:3, :4] = mat
        c2w[i] = np.linalg.inv(full)
    return c2w


def _align_sim3(gt_poses: np.ndarray, pred_poses: np.ndarray) -> np.ndarray:
    aligned = pred_poses.copy()
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses[:, :3, 3]

    gt_center = gt_xyz.mean(axis=0)
    pred_center = pred_xyz.mean(axis=0)
    gt_demean = gt_xyz - gt_center
    pred_demean = pred_xyz - pred_center

    pred_var = np.sum(pred_demean**2) / max(1, len(pred_xyz))
    if pred_var < 1e-12:
        return aligned

    scale = np.sqrt((np.sum(gt_demean**2) / max(1, len(gt_xyz))) / pred_var)
    H = (pred_demean * scale).T @ gt_demean
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_center - scale * (R @ pred_center)

    aligned[:, :3, :3] = R[None, ...] @ pred_poses[:, :3, :3]
    aligned[:, :3, 3] = (scale * (R @ pred_xyz.T)).T + t[None, :]
    return aligned


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.trace(R)
    cosv = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def _compute_per_frame_errors(gt_poses: np.ndarray, pred_poses: np.ndarray):
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses[:, :3, 3]
    ate = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)

    rte = np.zeros_like(ate)
    rre = np.zeros_like(ate)
    for i in range(1, len(gt_poses)):
        gt_rel = np.linalg.inv(gt_poses[i - 1]) @ gt_poses[i]
        pred_rel = np.linalg.inv(pred_poses[i - 1]) @ pred_poses[i]
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        rte[i] = np.linalg.norm(rel_err[:3, 3])
        rre[i] = _rotation_angle_deg(rel_err[:3, :3])
    return ate, rte, rre


def _set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-6))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _set_2d_bounds(ax, values: np.ndarray, axis: str) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-6:
        pad = 1e-3 if abs(lo) < 1e-3 else abs(lo) * 1e-3
    else:
        pad = 0.05 * (hi - lo)
    if axis == "x":
        ax.set_xlim(lo - pad, hi + pad)
    else:
        ax.set_ylim(lo - pad, hi + pad)


def _load_gt_w2c_from_sequence(seq) -> Optional[np.ndarray]:
    scene_root = getattr(seq, "scene_root", None)
    if not scene_root:
        return None

    camera = getattr(seq, "camera", None)
    if isinstance(camera, str) and camera:
        extri_path = os.path.join(scene_root, "cameras", camera, "extri.yml")
        intri_path = os.path.join(scene_root, "cameras", camera, "intri.yml")
    else:
        extri_path = os.path.join(scene_root, "extri.yml")
        intri_path = os.path.join(scene_root, "intri.yml")

    if not os.path.exists(extri_path):
        return None

    gt_extri, _, _ = read_opencv_camera_yml(extri_path, intri_path)
    if not gt_extri:
        return None

    stems = frame_stems(getattr(seq, "image_paths", []))
    if not stems:
        return None

    mats = []
    for stem in stems:
        mat = gt_extri.get(stem, None)
        if mat is None:
            continue
        mats.append(np.asarray(mat, dtype=np.float64))

    if len(mats) < 2:
        return None
    return np.stack(mats, axis=0)


def _resolve_output_file(save_path: str, title: str, suffix: str = ".png") -> str:
    save_path = os.path.abspath(save_path)
    lower = save_path.lower()
    if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
        out_file = save_path
    else:
        _ensure_dir(save_path)
        out_file = os.path.join(save_path, f"{title}{suffix}")
    _ensure_dir(os.path.dirname(out_file))
    return out_file


def _normalize_pred_maps(pred_w2c_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = {}
    for name, mat in pred_w2c_maps.items():
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] not in [(3, 4), (4, 4)]:
            continue
        normalized[name] = arr
    return normalized


def save_trajectory_comparison_plot(
    save_path: str,
    pred_w2c_maps: Dict[str, np.ndarray],
    *,
    seq=None,
    gt_w2c: Optional[np.ndarray] = None,
    title: str = "Camera Trajectory",
    dpi: int = 140,
    figure_size: Tuple[int, int] = (20, 12),
    offline_error: str = "",
) -> Optional[str]:
    pred_w2c_maps = _normalize_pred_maps(pred_w2c_maps)
    if not pred_w2c_maps:
        return None

    if gt_w2c is None and seq is not None:
        gt_w2c = _load_gt_w2c_from_sequence(seq)

    if gt_w2c is None:
        gt_w2c = None

    pred_c2w = {name: _w2c_to_c2w(mat) for name, mat in pred_w2c_maps.items()}
    gt_c2w = _w2c_to_c2w(gt_w2c) if gt_w2c is not None else None

    if gt_c2w is not None:
        min_len = min([gt_c2w.shape[0]] + [pose.shape[0] for pose in pred_c2w.values()])
    else:
        min_len = min(pose.shape[0] for pose in pred_c2w.values())
    if min_len < 2:
        return None

    if gt_c2w is None:
        return _save_pred_only_plot(save_path, pred_c2w, title=title, dpi=dpi)

    gt_c2w = gt_c2w[:min_len]
    pred_c2w = {name: pose[:min_len] for name, pose in pred_c2w.items()}

    aligned = {}
    curves = {}
    for name, pose in pred_c2w.items():
        pose_aligned = _align_sim3(gt_c2w, pose)
        ate, rte, rre = _compute_per_frame_errors(gt_c2w, pose_aligned)
        aligned[name] = pose_aligned
        curves[name] = dict(ate=ate, rte=rte, rre=rre)

    out_file = _resolve_output_file(save_path, title)
    _plot_pose_report(
        out_file=out_file,
        gt_poses=gt_c2w,
        pred_poses=aligned,
        curves=curves,
        alignment_tag="SIM3",
        title=title,
        dpi=dpi,
        figure_size=figure_size,
        offline_error=offline_error,
    )
    return out_file


def _save_pred_only_plot(
    save_path: str,
    pred_c2w: Dict[str, np.ndarray],
    *,
    title: str,
    dpi: int,
) -> str:
    out_file = _resolve_output_file(save_path, title)
    fig = plt.figure(figsize=(16, 5), dpi=dpi)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)

    color_map = {
        "online": "#1f77b4",
        "offline": "#2ca02c",
        "lag": "#ff7f0e",
        "last": "#8c564b",
    }

    all_xyz = []
    for name, pose in pred_c2w.items():
        xyz = pose[:, :3, 3]
        all_xyz.append(xyz)
        color = color_map.get(name, "#9467bd")
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2.0, label=name)
        ax_xy.plot(xyz[:, 0], xyz[:, 1], color=color, linewidth=2.0, label=name)
        ax_xz.plot(xyz[:, 0], xyz[:, 2], color=color, linewidth=2.0, label=name)

    all_xyz = np.concatenate(all_xyz, axis=0)
    _set_equal_3d_axes(ax3d, all_xyz)
    ax3d.set_title(title)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="best")
    ax3d.grid(True)

    ax_xy.set_title("XY Projection")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(loc="best")
    ax_xy.grid(True, alpha=0.4)

    ax_xz.set_title("XZ Projection")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.legend(loc="best")
    ax_xz.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def _plot_pose_report(
    *,
    out_file: str,
    gt_poses: np.ndarray,
    pred_poses: Dict[str, np.ndarray],
    curves: Dict[str, Dict[str, np.ndarray]],
    alignment_tag: str,
    title: str,
    dpi: int,
    figure_size: Tuple[int, int],
    offline_error: str = "",
) -> None:
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xz = fig.add_subplot(2, 3, 3)
    ax_ate = fig.add_subplot(2, 3, 4)
    ax_rte = fig.add_subplot(2, 3, 5)
    ax_rre = fig.add_subplot(2, 3, 6)

    color_map = {
        "online": "#1f77b4",
        "offline": "#2ca02c",
        "lag": "#ff7f0e",
        "last": "#8c564b",
    }
    line_style_map = {
        "online": "-",
        "offline": "-",
        "lag": "--",
        "last": "-.",
    }
    zorder_map = {
        "online": 2,
        "offline": 3,
        "last": 4,
        "lag": 5,
    }

    gt_xyz = gt_poses[:, :3, 3]
    pred_xyzs = {name: pose[:, :3, 3] for name, pose in pred_poses.items()}
    all_xyz = np.concatenate([gt_xyz] + list(pred_xyzs.values()), axis=0)
    _set_equal_3d_axes(ax3d, all_xyz)
    _set_2d_bounds(ax_xy, all_xyz[:, 0], "x")
    _set_2d_bounds(ax_xy, all_xyz[:, 1], "y")
    _set_2d_bounds(ax_xz, all_xyz[:, 0], "x")
    _set_2d_bounds(ax_xz, all_xyz[:, 2], "y")

    ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "r-", linewidth=2.2, alpha=0.9, label="GT")
    ax3d.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], c="red", s=6, alpha=0.6)
    ax_xy.plot(gt_xyz[:, 0], gt_xyz[:, 1], "r-", linewidth=2.2, alpha=0.9, label="GT")
    ax_xy.scatter(gt_xyz[:, 0], gt_xyz[:, 1], c="red", s=6, alpha=0.6)
    ax_xz.plot(gt_xyz[:, 0], gt_xyz[:, 2], "r-", linewidth=2.2, alpha=0.9, label="GT")
    ax_xz.scatter(gt_xyz[:, 0], gt_xyz[:, 2], c="red", s=6, alpha=0.6)

    for name, xyz in pred_xyzs.items():
        color = color_map.get(name, "#9467bd")
        line_style = line_style_map.get(name, "-")
        zorder = zorder_map.get(name, 2)
        curve = curves[name]
        stats = (
            f"{name}: "
            f"ATE={curve['ate'].mean():.3f}m "
            f"RTE={curve['rte'].mean():.3f}m "
            f"RRE={curve['rre'].mean():.3f}deg"
        )
        ax3d.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            alpha=0.9,
            label=stats,
            zorder=zorder,
        )
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=8, alpha=0.55, zorder=zorder)
        ax_xy.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            alpha=0.9,
            label=name,
            zorder=zorder,
        )
        ax_xy.scatter(xyz[:, 0], xyz[:, 1], c=color, s=7, alpha=0.55, zorder=zorder)
        ax_xz.plot(
            xyz[:, 0],
            xyz[:, 2],
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            alpha=0.9,
            label=name,
            zorder=zorder,
        )
        ax_xz.scatter(xyz[:, 0], xyz[:, 2], c=color, s=7, alpha=0.55, zorder=zorder)

    ax3d.set_title(f"3D Trajectory ({alignment_tag})" + (f"\n[ERROR] {offline_error}" if offline_error else ""))
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="best", fontsize=8)
    ax3d.grid(True)

    ax_xy.set_title("XY Projection")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(loc="best")
    ax_xy.grid(True)

    ax_xz.set_title("XZ Projection")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.legend(loc="best")
    ax_xz.grid(True)

    frame_idx = np.arange(len(gt_xyz))
    for name, curve in curves.items():
        color = color_map.get(name, "#9467bd")
        line_style = line_style_map.get(name, "-")
        zorder = zorder_map.get(name, 2)
        ax_ate.plot(
            frame_idx,
            curve["ate"],
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=f"{name} mean={curve['ate'].mean():.3f}",
            zorder=zorder,
        )
        ax_ate.fill_between(frame_idx, 0.0, curve["ate"], color=color, alpha=0.15)
        ax_rte.plot(
            frame_idx,
            curve["rte"],
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=f"{name} mean={curve['rte'].mean():.3f}",
            zorder=zorder,
        )
        ax_rte.fill_between(frame_idx, 0.0, curve["rte"], color=color, alpha=0.15)
        ax_rre.plot(
            frame_idx,
            curve["rre"],
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=f"{name} mean={curve['rre'].mean():.3f}",
            zorder=zorder,
        )
        ax_rre.fill_between(frame_idx, 0.0, curve["rre"], color=color, alpha=0.15)

    ax_ate.set_title("ATE Curve")
    ax_ate.set_xlabel("Frame")
    ax_ate.set_ylabel("ATE (m)")
    ax_ate.legend(loc="best", fontsize=8)
    ax_ate.grid(True, alpha=0.4)

    ax_rte.set_title("RTE Curve")
    ax_rte.set_xlabel("Frame")
    ax_rte.set_ylabel("RTE (m)")
    ax_rte.legend(loc="best", fontsize=8)
    ax_rte.grid(True, alpha=0.4)

    ax_rre.set_title("RRE Curve")
    ax_rre.set_xlabel("Frame")
    ax_rre.set_ylabel("RRE (deg)")
    ax_rre.legend(loc="best", fontsize=8)
    ax_rre.grid(True, alpha=0.4)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
