from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from horizonstream.core.infer import (
    _apply_depth_percentile_bounds,
    _build_point_valid_mask,
    _camera_points_to_world,
    _chunk_schedule,
    _mask_points_and_colors,
    _parse_depth_bounds,
    _pointcloud_filter_kwargs,
    _save_full_pointcloud,
    _sync_device_for_timing,
    _to_uint8_rgb,
)
from horizonstream.io.save_points import save_pointcloud
from horizonstream.utils.depth import unproject_depth_to_points
from horizonstream.core.model import HorizonStreamModel
from horizonstream.data.dataloader import HorizonStreamDataLoader
from horizonstream.eval.io import read_pred_w2c_txt
from horizonstream.io.save_poses_txt import save_intri_txt, save_w2c_txt
from horizonstream.loop.runtime import (
    LoopCandidate,
    LoopConfig,
    LoopEdge,
    SaladVPRModel,
    _batch_to_imagenet_tensor,
    _load_orb_vocabulary,
    _load_salad_state_dict,
    _normalize_rows,
    _resize_rgb_batch,
    build_keyframe_odometry_edges,
    compact_gt_metrics,
    evaluate_trajectory_against_gt,
    load_ground_truth_trajectory,
    non_max_suppress_loop_candidates,
    optimize_keyframe_pose_graph,
    plot_trajectory_panels,
    relative_measurement_from_c2w,
    save_loop_edges_json,
)
from horizonstream.utils.vendor.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from horizonstream.runtime.motion_averaging import affine_padding, compute_motion_averaged_camera_maps


def _log(msg: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[online-loop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c4 = affine_padding(torch.from_numpy(np.asarray(w2c, dtype=np.float64))).numpy()
    return np.linalg.inv(w2c4)


def _candidate_to_json(cand: LoopCandidate) -> Dict[str, Any]:
    return asdict(cand)


def _filter_loop_edges_by_score(
    loop_edges: List[LoopEdge],
    min_score: Optional[float],
) -> List[LoopEdge]:
    if min_score is None:
        return loop_edges
    threshold = float(min_score)
    return [edge for edge in loop_edges if float(edge.score) >= threshold]


def _centered_range(center: int, size: int, total: int) -> Tuple[int, int]:
    size = max(1, min(int(size), int(total)))
    half = size // 2
    start = int(center) - half
    end = start + size
    if start < 0:
        start = 0
        end = size
    if end > total:
        end = total
        start = total - size
    return int(start), int(end)


def _hist_score(vocab, hist_a, hist_b) -> float:
    if isinstance(hist_a, dict):
        small, large = (hist_a, hist_b) if len(hist_a) <= len(hist_b) else (hist_b, hist_a)
        score = 0.0
        for word_id, value in small.items():
            other = large.get(word_id)
            if other is not None:
                score += min(float(value), float(other))
        return float(score)
    a = np.asarray(hist_a, dtype=np.float32)
    b = np.asarray(hist_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _positions_from_metric(
    c2w: np.ndarray,
    frame_ids: Sequence[int],
    metric: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, np.ndarray]]:
    if metric is not None:
        xyz = np.asarray(metric["pred_xyz_aligned"], dtype=np.float64)
        gt = np.asarray(metric["gt_xyz"], dtype=np.float64)
        ids = [int(frame_id) for frame_id in metric["frame_ids"]]
    else:
        xyz = np.asarray(c2w, dtype=np.float64)[:, :3, 3]
        gt = None
        ids = [int(frame_id) for frame_id in frame_ids]
    return xyz, gt, {frame_id: xyz[idx] for idx, frame_id in enumerate(ids)}


def _plot_plane(
    ax,
    *,
    plane: str,
    xyz: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    gt_xyz: Optional[np.ndarray] = None,
    base_xyz: Optional[np.ndarray] = None,
    loop_edges: Sequence[LoopEdge] = (),
    frame_id_to_xyz: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    if plane == "xy":
        dims = (0, 1)
        xlabel, ylabel = "x", "y"
    elif plane == "xz":
        dims = (0, 2)
        xlabel, ylabel = "x", "z"
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    if gt_xyz is not None:
        ax.plot(gt_xyz[:, dims[0]], gt_xyz[:, dims[1]], color="#2ca02c", linewidth=2.0, label="gt")
    if base_xyz is not None:
        ax.plot(
            base_xyz[:, dims[0]],
            base_xyz[:, dims[1]],
            color="#808080",
            linewidth=1.5,
            linestyle="--",
            label="no_loop",
        )
    ax.plot(xyz[:, dims[0]], xyz[:, dims[1]], color=color, linewidth=2.0, linestyle=linestyle, label=label)
    if frame_id_to_xyz is not None:
        for edge in loop_edges:
            a = frame_id_to_xyz.get(int(edge.src_frame))
            b = frame_id_to_xyz.get(int(edge.dst_frame))
            if a is None or b is None:
                continue
            ax.plot([a[dims[0]], b[dims[0]]], [a[dims[1]], b[dims[1]]], color="#d62728", alpha=0.35, linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_online_loop_planes(
    output_path: str,
    *,
    baseline_c2w: np.ndarray,
    loop_c2w: np.ndarray,
    frame_ids: Sequence[int],
    loop_edges: Sequence[LoopEdge],
    gt_metrics: Dict[str, Optional[Dict[str, Any]]],
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    no_loop_metric = gt_metrics.get("no_loop")
    loop_metric = gt_metrics.get("online_loop_reinfer")
    base_xyz, base_gt, _ = _positions_from_metric(baseline_c2w, frame_ids, no_loop_metric)
    loop_xyz, loop_gt, loop_frame_xyz = _positions_from_metric(loop_c2w, frame_ids, loop_metric)
    gt_xyz = loop_gt if loop_gt is not None else base_gt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    no_loop_title = "No Loop"
    if no_loop_metric is not None:
        no_loop_title += f"\nATE {float(no_loop_metric['ate_rmse']):.2f} m"
    loop_title = "online_loop_reinfer"
    if loop_metric is not None:
        loop_title += f"\nATE {float(loop_metric['ate_rmse']):.2f} m"

    for row, plane in enumerate(("xz", "xy")):
        _plot_plane(
            axes[row, 0],
            plane=plane,
            xyz=base_xyz,
            label="no_loop",
            color="#1f77b4",
            gt_xyz=gt_xyz,
        )
        axes[row, 0].set_title(f"{no_loop_title} ({plane})")
        _plot_plane(
            axes[row, 1],
            plane=plane,
            xyz=loop_xyz,
            label="online_loop_reinfer",
            color="#1f77b4",
            gt_xyz=gt_xyz,
            base_xyz=base_xyz,
            loop_edges=loop_edges,
            frame_id_to_xyz=loop_frame_xyz,
        )
        axes[row, 1].set_title(f"{loop_title} ({plane})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@dataclass
class OnlineLoopReinferConfig:
    enabled_methods: Tuple[str, ...] = ("salad",)
    salad_ckpt_path: str = "weights/dino_salad.ckpt"
    salad_dino_weights_path: str = "weights/dinov2_vitb14_pretrain.pth"
    salad_backbone: str = "dinov2_vitb14"
    salad_image_size: Tuple[int, int] = (336, 336)
    salad_batch_size: int = 32
    salad_score_thresh: float = 0.85
    retrieval_top_k: int = 5
    orb_features: int = 4096
    dbow_thresh: float = 0.034
    dbow_num_repeat: int = 3
    temporal_exclusion: int = 10
    nms_radius: int = 25
    min_frame_separation: int = 30
    loop_chunk_size: int = 20
    max_candidates: int = 1000
    max_reinfer_candidates: int = 50
    dbow_direct_loop_edges: bool = False
    dbow_direct_edge_mode: str = "identity"
    dbow_direct_max_edges: int = 50
    dbow_direct_inliers: int = 64
    loop_edge_score_threshold: Optional[float] = None
    edge_mode: str = "relative_pose"
    pose_graph_backend: str = "pypose"
    pose_graph_model: str = "se3"
    pose_graph_update_mode: str = "all"
    pose_graph_loop_weight: float = 0.1
    pose_graph_trans_weight: float = 2.0
    pose_graph_rot_weight: float = 1.0
    pose_graph_scale_weight: float = 1.0
    pose_graph_inlier_cap: int = 512
    pose_graph_max_iterations: int = 30
    pose_graph_lambda_init: float = 1e-6
    pose_graph_solver_verbose: bool = False
    retrieval_log_every: int = 200
    verbose: bool = True
    auto_output_suffix: bool = False
    reuse_online_poses: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "OnlineLoopReinferConfig":
        cfg = dict(cfg or {})
        methods = cfg.get("methods", cfg.get("enabled_methods", ["salad"]))
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.split(",") if m.strip()]
        return cls(
            enabled_methods=tuple(methods or ["salad"]),
            salad_ckpt_path=str(cfg.get("salad_ckpt_path", cls.salad_ckpt_path)),
            salad_dino_weights_path=str(
                cfg.get("salad_dino_weights_path", cls.salad_dino_weights_path)
            ),
            salad_backbone=str(cfg.get("salad_backbone", cls.salad_backbone)),
            salad_image_size=tuple(
                int(x) for x in cfg.get("salad_image_size", cls.salad_image_size)
            ),
            salad_batch_size=int(cfg.get("salad_batch_size", cls.salad_batch_size)),
            salad_score_thresh=float(cfg.get("salad_score_thresh", cls.salad_score_thresh)),
            retrieval_top_k=int(cfg.get("retrieval_top_k", cls.retrieval_top_k)),
            orb_features=int(cfg.get("orb_features", cls.orb_features)),
            dbow_thresh=float(cfg.get("dbow_thresh", cls.dbow_thresh)),
            dbow_num_repeat=int(cfg.get("dbow_num_repeat", cls.dbow_num_repeat)),
            temporal_exclusion=int(cfg.get("temporal_exclusion", cls.temporal_exclusion)),
            nms_radius=int(cfg.get("nms_radius", cls.nms_radius)),
            min_frame_separation=int(cfg.get("min_frame_separation", cls.min_frame_separation)),
            loop_chunk_size=int(cfg.get("loop_chunk_size", cls.loop_chunk_size)),
            max_candidates=int(cfg.get("max_candidates", cls.max_candidates)),
            max_reinfer_candidates=int(cfg.get("max_reinfer_candidates", cls.max_reinfer_candidates)),
            dbow_direct_loop_edges=bool(
                cfg.get("dbow_direct_loop_edges", cls.dbow_direct_loop_edges)
            ),
            dbow_direct_edge_mode=str(
                cfg.get("dbow_direct_edge_mode", cls.dbow_direct_edge_mode)
            ),
            dbow_direct_max_edges=int(
                cfg.get("dbow_direct_max_edges", cls.dbow_direct_max_edges)
            ),
            dbow_direct_inliers=int(cfg.get("dbow_direct_inliers", cls.dbow_direct_inliers)),
            loop_edge_score_threshold=(
                None
                if cfg.get("loop_edge_score_threshold", cls.loop_edge_score_threshold) is None
                else float(cfg.get("loop_edge_score_threshold"))
            ),
            edge_mode=str(cfg.get("edge_mode", cls.edge_mode)),
            pose_graph_backend=str(cfg.get("pose_graph_backend", cls.pose_graph_backend)),
            pose_graph_model=str(cfg.get("pose_graph_model", cls.pose_graph_model)),
            pose_graph_update_mode=str(
                cfg.get("pose_graph_update_mode", cls.pose_graph_update_mode)
            ),
            pose_graph_loop_weight=float(
                cfg.get("pose_graph_loop_weight", cls.pose_graph_loop_weight)
            ),
            pose_graph_trans_weight=float(
                cfg.get("pose_graph_trans_weight", cls.pose_graph_trans_weight)
            ),
            pose_graph_rot_weight=float(
                cfg.get("pose_graph_rot_weight", cls.pose_graph_rot_weight)
            ),
            pose_graph_scale_weight=float(
                cfg.get("pose_graph_scale_weight", cls.pose_graph_scale_weight)
            ),
            pose_graph_inlier_cap=int(cfg.get("pose_graph_inlier_cap", cls.pose_graph_inlier_cap)),
            pose_graph_max_iterations=int(
                cfg.get("pose_graph_max_iterations", cls.pose_graph_max_iterations)
            ),
            pose_graph_lambda_init=float(cfg.get("pose_graph_lambda_init", cls.pose_graph_lambda_init)),
            pose_graph_solver_verbose=bool(
                cfg.get("pose_graph_solver_verbose", cls.pose_graph_solver_verbose)
            ),
            retrieval_log_every=int(cfg.get("retrieval_log_every", cls.retrieval_log_every)),
            verbose=bool(cfg.get("verbose", cls.verbose)),
            auto_output_suffix=bool(cfg.get("auto_output_suffix", cls.auto_output_suffix)),
            reuse_online_poses=bool(cfg.get("reuse_online_poses", cls.reuse_online_poses)),
        )


def _read_intri_txt(path: str) -> Tuple[List[int], np.ndarray]:
    frames: List[int] = []
    intri = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 5:
                continue
            frame = int(vals[0])
            fx, fy, cx, cy = vals[1:]
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            frames.append(frame)
            intri.append(K)
    return frames, np.asarray(intri, dtype=np.float64)


def _load_online_pose_from_scene_root(scene_root: str, frames: int) -> Tuple[np.ndarray, np.ndarray]:
    pose_dir = os.path.join(scene_root, "poses")
    w2c_candidates = [
        os.path.join(pose_dir, "abs_pose.txt"),
        os.path.join(pose_dir, "online_abs_pose.txt"),
    ]
    w2c_path = next((p for p in w2c_candidates if os.path.exists(p)), None)
    if w2c_path is None:
        raise FileNotFoundError(f"Missing online pose file for reuse mode under {pose_dir}")

    frame_ids, w2c = read_pred_w2c_txt(w2c_path)
    if len(w2c) < frames:
        raise ValueError(
            f"Reuse online poses frame count mismatch: poses={len(w2c)} images={frames} at {w2c_path}"
        )
    if len(w2c) > frames:
        frame_ids = frame_ids[:frames]
        w2c = w2c[:frames]
    if list(frame_ids) != list(range(frames)):
        raise ValueError(
            "Reuse online poses requires contiguous frame ids 0..N-1 to match image order "
            f"(got first={frame_ids[:3]} last={frame_ids[-3:]})"
        )
    online_w2c = np.asarray(w2c, dtype=np.float64)

    intri_candidates = [
        os.path.join(pose_dir, "online_intri.txt"),
        os.path.join(pose_dir, "intri.txt"),
    ]
    intri_path = next((p for p in intri_candidates if os.path.exists(p)), None)
    if intri_path is None:
        raise FileNotFoundError(f"Missing intrinsics for reuse mode under {pose_dir}")
    intri_frame_ids, intri = _read_intri_txt(intri_path)
    if len(intri) < frames:
        raise ValueError(
            f"Reuse intrinsics mismatch at {intri_path}: len={len(intri)} expected={frames}"
        )
    if len(intri) > frames:
        intri_frame_ids = intri_frame_ids[:frames]
        intri = intri[:frames]
    if intri_frame_ids != list(range(frames)):
        raise ValueError(
            f"Reuse intrinsics frame ids mismatch at {intri_path}: expected contiguous 0..{frames - 1}"
        )
    return online_w2c, intri


class OnlineDbowLoopDetector:
    def __init__(self, cfg: OnlineLoopReinferConfig):
        self.cfg = cfg
        self.backend = str(cfg.dbow_backend).lower()
        if self.backend not in {"auto", "python", "dpretrieval"}:
            raise ValueError(f"Unsupported dbow_backend: {cfg.dbow_backend}")
        self.dbow = None
        self.vocab = None
        self.orb = None
        if self.backend in {"auto", "dpretrieval"}:
            try:
                import dpretrieval  # type: ignore

                radius = max(1, int(cfg.temporal_exclusion), int(cfg.min_frame_separation))
                self.dbow = dpretrieval.DPRetrieval(str(cfg.dbow_vocab_path), radius)
                self.backend = "dpretrieval"
            except Exception as exc:
                if self.backend == "dpretrieval":
                    raise RuntimeError(
                        "dbow_backend='dpretrieval' requires the DPRetrieval "
                        "pybind module to be installed"
                    ) from exc
                self.backend = "python"
                _log(
                    f"dpretrieval unavailable ({exc}); falling back to Python DBoW",
                    cfg.verbose,
                )
        if self.backend == "python":
            self.vocab = _load_orb_vocabulary(cfg.dbow_vocab_path)
            self.orb = cv2.ORB_create(
                nfeatures=int(cfg.orb_features),
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                patchSize=31,
                fastThreshold=20,
            )
        self.histograms: List[Any] = []
        self.inverted: Dict[int, List[Tuple[int, float]]] = {}
        self.found: List[Tuple[int, int, float]] = []
        self.accepted_pairs: List[Tuple[int, int]] = []
        self.candidates: List[LoopCandidate] = []
        vocab_words = self.vocab.num_words if self.vocab is not None else "DBoW2"
        _log(
            f"DBoW online detector backend={self.backend} loaded {cfg.dbow_vocab_path} with {vocab_words} words",
            cfg.verbose,
        )

    def _extract_histogram(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)
        if hasattr(self.vocab, "sparse_histogram"):
            return self.vocab.sparse_histogram(desc)
        return self.vocab.histogram(desc)

    def _best_inverted_match(self, hist: Dict[int, float], max_dst: int) -> Tuple[int, float]:
        scores: Dict[int, float] = {}
        for word_id, value in hist.items():
            for dst, other in self.inverted.get(int(word_id), ()):
                if dst <= max_dst:
                    scores[dst] = scores.get(dst, 0.0) + min(float(value), float(other))
        if not scores:
            return -1, -1.0
        best_dst, best_score = max(scores.items(), key=lambda item: item[1])
        return int(best_dst), float(best_score)

    def _append_histogram(self, frame_idx: int, hist: Any) -> None:
        self.histograms.append(hist)
        if isinstance(hist, dict):
            for word_id, value in hist.items():
                self.inverted.setdefault(int(word_id), []).append((int(frame_idx), float(value)))

    def update(self, frame_idx: int, rgb: np.ndarray) -> Optional[LoopCandidate]:
        if self.backend == "dpretrieval":
            if self.dbow is None:
                raise RuntimeError("Internal error: dpretrieval backend is not initialized")
            self.dbow.insert_image(np.ascontiguousarray(rgb))
            best_score, best_dst, _ = self.dbow.query(int(frame_idx))
            best_dst = int(best_dst)
            best_score = float(best_score)
        else:
            hist = self._extract_histogram(rgb)

            max_dst = int(frame_idx) - max(1, int(self.cfg.temporal_exclusion))
            if max_dst < 0:
                self._append_histogram(frame_idx, hist)
                return None

            if isinstance(hist, dict):
                best_dst, best_score = self._best_inverted_match(hist, max_dst)
            else:
                best_dst = -1
                best_score = -1.0
                for dst in range(max_dst + 1):
                    score = _hist_score(self.vocab, hist, self.histograms[dst])
                    if score > best_score:
                        best_score = score
                        best_dst = dst
            self._append_histogram(frame_idx, hist)
        if best_dst < 0 or best_score < float(self.cfg.dbow_thresh):
            return None

        if abs(int(frame_idx) - int(best_dst)) < int(self.cfg.min_frame_separation):
            return None

        nms = max(0, int(self.cfg.nms_radius))
        if nms > 0:
            for prev_src, prev_dst in self.accepted_pairs:
                if (frame_idx - prev_src) ** 2 + (best_dst - prev_dst) ** 2 < nms * nms:
                    return None

        self.found.append((int(frame_idx), int(best_dst), float(best_score)))
        repeat = max(1, int(self.cfg.dbow_num_repeat))
        if len(self.found) < repeat:
            return None
        latest = self.found[-repeat:]
        first_src = latest[0][0]
        if (1 + int(frame_idx) - first_src) != repeat:
            return None

        src, dst, score = latest[repeat // 2]
        cand = LoopCandidate(
            src_pos=int(src),
            dst_pos=int(dst),
            src_frame=int(src),
            dst_frame=int(dst),
            score=float(score),
            method="dbow_online",
        )
        self.accepted_pairs.append((int(src), int(dst)))
        self.candidates.append(cand)
        self.found.clear()
        _log(
            f"DBoW candidate frame {cand.src_frame} -> {cand.dst_frame} score={cand.score:.3f}",
            self.cfg.verbose,
        )
        return cand


def retrieve_online_salad_candidates(
    rgb: np.ndarray,
    cfg: OnlineLoopReinferConfig,
    device: str,
) -> List[LoopCandidate]:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "SALAD loop retrieval requires faiss. Install faiss-cpu or faiss-gpu first."
        ) from exc

    if not os.path.exists(cfg.salad_ckpt_path):
        raise FileNotFoundError(
            f"SALAD checkpoint not found: {cfg.salad_ckpt_path}. "
            "Expected SALAD dino_salad.ckpt."
        )
    torch_device = torch.device(device)
    model = SaladVPRModel(
        backbone_name=str(cfg.salad_backbone),
        dino_weights_path=str(cfg.salad_dino_weights_path),
    )
    missing = _load_salad_state_dict(model, str(cfg.salad_ckpt_path))
    if any(key.startswith("backbone.") for key in missing) and not os.path.exists(
        cfg.salad_dino_weights_path
    ):
        raise FileNotFoundError(
            f"DINOv2 backbone weights not found: {cfg.salad_dino_weights_path}. "
            "The SALAD checkpoint did not include all backbone weights."
        )
    model = model.to(torch_device).eval()

    frames = int(rgb.shape[0])
    batch_size = max(1, int(cfg.salad_batch_size))
    image_size = tuple(int(x) for x in cfg.salad_image_size)
    desc_chunks: List[np.ndarray] = []
    use_amp = torch_device.type == "cuda"
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, frames, batch_size), start=1):
            batch = rgb[start : start + batch_size]
            resized = _resize_rgb_batch(batch, image_size)
            images = _batch_to_imagenet_tensor(resized, torch_device)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                desc = model(images)
            desc_chunks.append(desc.float().cpu().numpy().astype(np.float32))
    descriptors = _normalize_rows(np.concatenate(desc_chunks, axis=0).astype(np.float32))

    index = faiss.IndexFlatIP(int(descriptors.shape[1]))
    index.add(descriptors)
    top_k = max(1, int(cfg.retrieval_top_k))
    similarities, indices = index.search(descriptors, top_k + 1)

    min_gap = max(1, int(cfg.min_frame_separation))
    score_thresh = float(cfg.salad_score_thresh)
    candidates_by_key: Dict[Tuple[int, int], LoopCandidate] = {}
    for src_pos in range(frames):
        for rank in range(1, top_k + 1):
            dst_pos = int(indices[src_pos, rank])
            if dst_pos < 0 or dst_pos == src_pos:
                continue
            score = float(similarities[src_pos, rank])
            if score <= score_thresh:
                continue
            if abs(src_pos - dst_pos) <= min_gap:
                continue
            high = max(src_pos, dst_pos)
            low = min(src_pos, dst_pos)
            key = (high, low)
            prev = candidates_by_key.get(key)
            if prev is not None and prev.score >= score:
                continue
            candidates_by_key[key] = LoopCandidate(
                src_pos=int(high),
                dst_pos=int(low),
                src_frame=int(high),
                dst_frame=int(low),
                score=score,
                method="salad_online",
            )

    candidates = non_max_suppress_loop_candidates(
        list(candidates_by_key.values()),
        nms_threshold=max(0, int(cfg.nms_radius)),
        limit=max(1, int(cfg.max_candidates)),
    )
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


class LoopReinferRefiner:
    def __init__(
        self,
        model: HorizonStreamModel,
        images: torch.Tensor,
        device: str,
        image_hw: Tuple[int, int],
        cfg: OnlineLoopReinferConfig,
    ):
        self.model = model
        self.images = images
        self.device = device
        self.image_hw = image_hw
        self.cfg = cfg

    def refine(self, cand: LoopCandidate) -> Optional[LoopEdge]:
        if self.cfg.edge_mode != "relative_pose":
            raise ValueError(f"Unsupported online loop edge_mode: {self.cfg.edge_mode}")
        total = int(self.images.shape[1])
        if abs(int(cand.src_frame) - int(cand.dst_frame)) < int(self.cfg.min_frame_separation):
            return None

        src_start, src_end = _centered_range(cand.src_frame, self.cfg.loop_chunk_size, total)
        dst_start, dst_end = _centered_range(cand.dst_frame, self.cfg.loop_chunk_size, total)
        src_ids = list(range(src_start, src_end))
        dst_ids = list(range(dst_start, dst_end))
        if cand.src_frame not in src_ids or cand.dst_frame not in dst_ids:
            return None
        loop_frame_ids = src_ids + dst_ids
        if len(loop_frame_ids) < 4:
            return None

        loop_images = self.images[:, loop_frame_ids].to(self.device, non_blocking=True)
        use_amp = isinstance(self.device, str) and self.device.startswith("cuda")
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp and torch.cuda.is_available(),
            ):
                outputs = self.model.forward_window(loop_images)
        pose_enc = outputs.get("pose_enc", None)
        if pose_enc is None:
            return None
        extri, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=self.image_hw)
        w2c = extri[0].detach().cpu().numpy().astype(np.float64, copy=False)
        c2w = _w2c_to_c2w(w2c)

        src_local = loop_frame_ids.index(int(cand.src_frame))
        dst_local = loop_frame_ids.index(int(cand.dst_frame))
        transform_ji = relative_measurement_from_c2w(c2w[src_local], c2w[dst_local])

        depth_conf = outputs.get("depth_conf", None)
        inliers = 1
        if depth_conf is not None:
            conf_np = depth_conf[0].detach().float().cpu().numpy()
            src_conf = conf_np[src_local]
            dst_conf = conf_np[dst_local]
            thresh = min(float(np.median(src_conf)), float(np.median(dst_conf))) * 0.1
            inliers = int(np.count_nonzero((src_conf > thresh) & (dst_conf > thresh)))
            inliers = min(inliers, int(self.cfg.pose_graph_inlier_cap))

        del outputs, loop_images
        if torch.cuda.is_available() and isinstance(self.device, str) and self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return LoopEdge(
            src_pos=int(cand.src_pos),
            dst_pos=int(cand.dst_pos),
            src_frame=int(cand.src_frame),
            dst_frame=int(cand.dst_frame),
            score=float(cand.score),
            inliers=int(max(1, inliers)),
            method=f"{cand.method}_reinfer",
            transform_ji=np.asarray(transform_ji, dtype=np.float64),
        )


def _build_loop_cfg(cfg: OnlineLoopReinferConfig) -> LoopConfig:
    loop_cfg = LoopConfig(
        verbose=bool(cfg.verbose),
        pose_graph_backend=str(cfg.pose_graph_backend),
        pose_graph_model=str(cfg.pose_graph_model),
        pose_graph_update_mode=str(cfg.pose_graph_update_mode),
        pose_graph_solver_verbose=bool(cfg.pose_graph_solver_verbose),
        keyframe_stride=1,
        min_keyframe_gap=int(cfg.min_frame_separation),
        dbow_temporal_exclusion=int(cfg.temporal_exclusion),
        loop_chunk_size=int(cfg.loop_chunk_size),
        dbow_num_repeat=int(cfg.dbow_num_repeat),
        retrieval_score_thresh_dbow=float(cfg.dbow_thresh),
        retrieval_top_k=int(cfg.retrieval_top_k),
        retrieval_score_thresh_salad=float(cfg.salad_score_thresh),
        nms_radius=int(cfg.nms_radius),
        max_candidates_per_method=int(cfg.max_candidates),
        max_verified_loops_per_method=int(cfg.max_reinfer_candidates),
        salad_ckpt_path=str(cfg.salad_ckpt_path),
        salad_dino_weights_path=str(cfg.salad_dino_weights_path),
        salad_backbone=str(cfg.salad_backbone),
        salad_image_size=tuple(int(x) for x in cfg.salad_image_size),
        salad_batch_size=int(cfg.salad_batch_size),
        pose_graph_loop_weight=float(cfg.pose_graph_loop_weight),
        pose_graph_trans_weight=float(cfg.pose_graph_trans_weight),
        pose_graph_rot_weight=float(cfg.pose_graph_rot_weight),
        pose_graph_scale_weight=float(cfg.pose_graph_scale_weight),
        pose_graph_max_iterations=int(cfg.pose_graph_max_iterations),
        pose_graph_lambda_init=float(cfg.pose_graph_lambda_init),
    )
    return loop_cfg


def _build_direct_loop_edges(
    candidates: Sequence[LoopCandidate],
    cfg: OnlineLoopReinferConfig,
) -> List[LoopEdge]:
    mode = str(cfg.dbow_direct_edge_mode).strip().lower()
    if mode not in {"identity"}:
        raise ValueError(f"Unsupported dbow_direct_edge_mode: {cfg.dbow_direct_edge_mode}")

    max_edges = max(0, int(cfg.dbow_direct_max_edges))
    if max_edges == 0:
        return []
    selected = list(candidates[:max_edges])
    inliers = max(1, int(cfg.dbow_direct_inliers))
    transform_ji = np.eye(4, dtype=np.float64)

    edges: List[LoopEdge] = []
    for cand in selected:
        edges.append(
            LoopEdge(
                src_pos=int(cand.src_pos),
                dst_pos=int(cand.dst_pos),
                src_frame=int(cand.src_frame),
                dst_frame=int(cand.dst_frame),
                score=float(cand.score),
                inliers=inliers,
                method=f"{cand.method}_direct",
                transform_ji=transform_ji.copy(),
            )
        )
    return edges


def _rebuild_pointcloud_from_online_depth(
    online_output_seq_dir: str,
    lc_seq_dir: str,
    loop_w2c: np.ndarray,
    loop_intri: np.ndarray,
    frame_ids: List[int],
    output_cfg: Dict[str, Any],
    verbose: bool = True,
) -> None:
    """Read depth npy + rgb from online output, unproject with LC poses, save to LC output."""
    depth_dir = os.path.join(online_output_seq_dir, "depth", "dpt")
    rgb_dir = os.path.join(online_output_seq_dir, "images", "rgb")
    if not os.path.isdir(depth_dir):
        _log(f"skip pointcloud: depth dir not found: {depth_dir}", verbose)
        return

    point_depth_min, point_depth_max = _parse_depth_bounds(output_cfg)
    pmin = output_cfg.get("point_depth_percentile_min", None)
    pmax = output_cfg.get("point_depth_percentile_max", None)
    point_mask_sky = bool(output_cfg.get("point_mask_sky", False))
    max_full = output_cfg.get("max_full_pointcloud_points", None)
    max_full = None if max_full in (None, "", "null") else int(max_full)
    max_frame = output_cfg.get("max_frame_pointcloud_points", None)
    max_frame = None if max_frame in (None, "", "null") else int(max_frame)
    voxel_size = output_cfg.get("point_voxel_size", None)
    voxel_size = None if voxel_size in (None, "", "null") else float(voxel_size)
    random_sample_ratio = output_cfg.get("point_random_sample_ratio", None)
    random_sample_ratio = (
        None
        if random_sample_ratio in (None, "", "null")
        else float(random_sample_ratio)
    )
    pointcloud_filter_kwargs = _pointcloud_filter_kwargs(output_cfg)
    save_frame_points = bool(output_cfg.get("save_frame_points", False))
    sky_mask_paths = []
    if point_mask_sky:
        sky_dir = os.path.join(online_output_seq_dir, "sky_masks")
        if os.path.isdir(sky_dir):
            sky_mask_paths = sorted(
                os.path.join(sky_dir, name)
                for name in os.listdir(sky_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            )

    point_chunks: List[np.ndarray] = []
    color_chunks: List[np.ndarray] = []
    for i in frame_ids:
        depth_path = os.path.join(depth_dir, f"frame_{i:06d}.npy")
        if not os.path.exists(depth_path):
            continue
        d = np.load(depth_path)
        rgb_path = os.path.join(rgb_dir, f"frame_{i:06d}.png")
        rgb_frame = None
        if os.path.exists(rgb_path):
            rgb_frame = np.array(Image.open(rgb_path))

        sky_mask = None
        if point_mask_sky and i < len(sky_mask_paths):
            sky_mask = cv2.imread(sky_mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if sky_mask is not None and sky_mask.shape[:2] != d.shape[:2]:
                sky_mask = cv2.resize(sky_mask, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid_mask = _build_point_valid_mask(d, sky_mask=sky_mask, depth_min=point_depth_min, depth_max=point_depth_max)
        valid_mask = _apply_depth_percentile_bounds(d, valid_mask, pmin, pmax)
        if not np.any(valid_mask):
            continue
        depth_for_unproj = np.where(valid_mask, d, 0.0).astype(np.float32)
        depth_b = torch.from_numpy(depth_for_unproj[None]).float()
        intri_b = torch.from_numpy(loop_intri[i : i + 1]).float()
        pts_cam = unproject_depth_to_points(depth_b, intri_b)[0].numpy()
        cols = rgb_frame[:pts_cam.shape[0], :pts_cam.shape[1]] if rgb_frame is not None else None
        pts, cols = _mask_points_and_colors(pts_cam, cols, valid_mask)
        pts_world = _camera_points_to_world(pts, loop_w2c[i][:3])
        if pts_world.shape[0] == 0:
            continue
        point_chunks.append(pts_world)
        if cols is not None:
            color_chunks.append(cols)
        if save_frame_points:
            frame_pc_dir = os.path.join(lc_seq_dir, "points", "frames_lc")
            _ensure_dir(frame_pc_dir)
            save_pointcloud(
                os.path.join(frame_pc_dir, f"frame_{i:06d}.ply"),
                pts_world,
                colors=cols,
                max_points=max_frame,
                voxel_size=voxel_size,
                random_sample_ratio=random_sample_ratio,
                **pointcloud_filter_kwargs,
                seed=i,
            )

    if point_chunks:
        pc_dir = os.path.join(lc_seq_dir, "points")
        _ensure_dir(pc_dir)
        _save_full_pointcloud(
            os.path.join(pc_dir, "full_lc.ply"),
            point_chunks,
            color_chunks,
            max_points=max_full,
            voxel_size=voxel_size,
            random_sample_ratio=random_sample_ratio,
            filter_kwargs=pointcloud_filter_kwargs,
        )
    else:
        _log("no valid points for pointcloud", verbose)


def run_online_loop_reinfer(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {}) or {}
    gt_data_cfg = cfg.get("gt_data", data_cfg) or data_cfg
    infer_cfg = cfg.get("inference", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    loop_cfg_dict = cfg.get("online_loop", cfg.get("loop", {})) or {}
    online_cfg = OnlineLoopReinferConfig.from_dict(loop_cfg_dict)
    online_pose_root = loop_cfg_dict.get("online_output_root", None)
    online_output_root = online_pose_root
    rebuild_pointcloud = bool(loop_cfg_dict.get("rebuild_pointcloud", True))
    if not rebuild_pointcloud:
        online_output_root = None

    enabled_methods = {m.lower() for m in online_cfg.enabled_methods}
    unsupported = enabled_methods.difference({"salad"})
    if unsupported:
        raise ValueError(
            "online_loop_reinfer currently supports methods=[salad] only "
            f"(got unsupported={sorted(unsupported)})"
        )

    out_root = output_cfg.get("root", "outputs_horizonstream")
    _ensure_dir(out_root)

    no_reinfer_mode = bool(online_cfg.reuse_online_poses) and int(online_cfg.max_reinfer_candidates) <= 0
    _log(f"device={device}", online_cfg.verbose)
    if not rebuild_pointcloud:
        _log("pointcloud rebuild disabled", online_cfg.verbose)
    if no_reinfer_mode:
        model = None
        _log(
            "reuse_online_poses=1 and max_reinfer_candidates<=0: skip model loading and loop-window re-inference",
            online_cfg.verbose,
        )
    else:
        model = None
    loader = HorizonStreamDataLoader(data_cfg)

    def ensure_model() -> HorizonStreamModel:
        nonlocal model
        if model is None:
            model = HorizonStreamModel(cfg.get("model", {})).to(device)
            model.eval()
        return model

    window_size = int(infer_cfg.get("window_size", 10))
    sliding_size = int(infer_cfg.get("sliding_size", window_size))
    if window_size <= 0 or sliding_size <= 0:
        raise ValueError("window_size and sliding_size must be positive")

    summaries: Dict[str, Any] = {
        "output_root": out_root,
        "device": device,
        "window_size": window_size,
        "sliding_size": sliding_size,
        "loop_config": asdict(online_cfg),
        "rebuild_pointcloud": bool(rebuild_pointcloud),
        "sequences": {},
    }
    root_summary_path = os.path.join(out_root, "summary.json")

    with torch.no_grad():
        prev_seq_done_t = time.perf_counter()
        for seq in loader:
            seq_ready_t = time.perf_counter()
            loader_wait_sec = seq_ready_t - prev_seq_done_t
            seq_t0 = time.perf_counter()
            _log(
                f"sequence {seq.name}: dataloader wait {loader_wait_sec:.2f}s",
                online_cfg.verbose,
            )
            images = seq.images.cpu()
            cpu_copy_sec = time.perf_counter() - seq_t0
            batch, frames, _, height, width = images.shape
            if batch != 1:
                raise ValueError("online_loop_reinfer expects batch size 1")
            _log(
                f"sequence {seq.name}: start ({frames} frames), image_cpu_copy={cpu_copy_sec:.2f}s",
                online_cfg.verbose,
            )

            rgb = _to_uint8_rgb(images[0].permute(0, 2, 3, 1))
            detector = OnlineDbowLoopDetector(online_cfg) if "dbow" in enabled_methods else None
            candidates: List[LoopCandidate] = []
            forward_time = 0.0
            retrieval_t0 = time.perf_counter()
            retrieval_last_log_t = retrieval_t0

            if "salad" in enabled_methods:
                candidates.extend(retrieve_online_salad_candidates(rgb, online_cfg, device))

            if online_cfg.reuse_online_poses:
                if detector is not None:
                    for frame_idx in range(frames):
                        cand = detector.update(frame_idx, rgb[frame_idx])
                        if cand is not None:
                            if len(candidates) < int(online_cfg.max_candidates):
                                candidates.append(cand)
                            elif len(candidates) == int(online_cfg.max_candidates):
                                _log(
                                    "candidate limit reached; retrieval remains cached but no more candidates are kept",
                                    online_cfg.verbose,
                                )
                        if (
                            (frame_idx + 1) % max(1, int(online_cfg.retrieval_log_every)) == 0
                            or frame_idx + 1 == frames
                        ):
                            now = time.perf_counter()
                            dt = max(now - retrieval_last_log_t, 1e-9)
                            retrieval_last_log_t = now
                            done = frame_idx + 1
                            _log(
                                f"sequence {seq.name}: retrieval {done}/{frames} candidates={len(candidates)} "
                                f"chunk_time={dt:.2f}s fps={int(done / max(now - retrieval_t0, 1e-9))}",
                                online_cfg.verbose,
                            )
                if online_pose_root is not None:
                    pose_scene_root = os.path.join(online_pose_root, seq.name)
                else:
                    pose_scene_root = seq.scene_root
                online_w2c, online_intri_np = _load_online_pose_from_scene_root(pose_scene_root, frames)
            else:
                model = ensure_model()
                chunks = _chunk_schedule(frames, window_size, sliding_size)
                state = model.build_sequence_state()
                chunk_cam_maps: List[torch.Tensor] = []
                for chunk_idx, (start, end) in enumerate(chunks):
                    if detector is not None:
                        for frame_idx in range(start, end):
                            cand = detector.update(frame_idx, rgb[frame_idx])
                            if cand is not None:
                                if len(candidates) < int(online_cfg.max_candidates):
                                    candidates.append(cand)
                                elif len(candidates) == int(online_cfg.max_candidates):
                                    _log(
                                        "candidate limit reached; retrieval remains cached but no more candidates are kept",
                                        online_cfg.verbose,
                                    )

                    _sync_device_for_timing(device)
                    t0 = time.perf_counter()
                    chunk_images = images[:, start:end].to(device, non_blocking=True)
                    current_window_size = end - start if chunk_idx == 0 else window_size
                    outputs = model.forward_chunk(
                        chunk_images,
                        window_size=current_window_size,
                        chunk_idx=chunk_idx,
                        state=state,
                    )
                    _sync_device_for_timing(device)
                    forward_time += time.perf_counter() - t0
                    chunk_cam_maps.append(outputs["chunk_cam_map"].detach().float().cpu())
                    model.advance_sequence_state(state, is_last_chunk=chunk_idx == len(chunks) - 1)
                    del outputs, chunk_images
                    _log(
                        f"sequence {seq.name}: chunk {chunk_idx + 1}/{len(chunks)} done, candidates={len(candidates)}",
                        online_cfg.verbose,
                    )

                motion_maps = compute_motion_averaged_camera_maps(
                    chunk_cam_maps,
                    frames_num=frames,
                    window_size=window_size,
                    dtype=torch.float32,
                )
                online_cam_map = motion_maps["online_cam_map"]
                online_extri, online_intri = pose_encoding_to_extri_intri(
                    online_cam_map,
                    image_size_hw=(height, width),
                )
                online_w2c = online_extri[0].detach().cpu().numpy().astype(np.float64, copy=False)
                online_intri_np = online_intri[0].detach().cpu().numpy().astype(np.float64, copy=False)
                if online_pose_root is not None:
                    pose_scene_root = os.path.join(online_pose_root, seq.name)
                    online_w2c, online_intri_np = _load_online_pose_from_scene_root(
                        pose_scene_root,
                        frames,
                    )
                    _log(
                        f"sequence {seq.name}: using existing online poses from {pose_scene_root}",
                        online_cfg.verbose,
                    )
            retrieval_sec = time.perf_counter() - retrieval_t0
            online_c2w = _w2c_to_c2w(online_w2c)

            candidates = candidates[: int(online_cfg.max_candidates)]
            candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
            if candidates:
                preview = ", ".join(
                    f"{c.src_frame}->{c.dst_frame}:{c.score:.3f}/{c.method}"
                    for c in candidates[:10]
                )
                _log(f"sequence {seq.name}: candidates top={preview}", online_cfg.verbose)
            refine_candidates = candidates[: int(online_cfg.max_reinfer_candidates)]
            _log(
                f"sequence {seq.name}: refining {len(refine_candidates)}/{len(candidates)} candidates",
                online_cfg.verbose,
            )
            loop_edges: List[LoopEdge] = []
            refine_t0 = time.perf_counter()
            if int(online_cfg.max_reinfer_candidates) <= 0 and bool(online_cfg.dbow_direct_loop_edges):
                loop_edges = _build_direct_loop_edges(candidates, online_cfg)
                _log(
                    f"sequence {seq.name}: direct dbow loop edges {len(loop_edges)}/{len(candidates)}",
                    online_cfg.verbose,
                )
            elif len(refine_candidates) > 0:
                model = ensure_model()
                refiner = LoopReinferRefiner(
                    model=model,
                    images=images,
                    device=device,
                    image_hw=(height, width),
                    cfg=online_cfg,
                )
                for idx, cand in enumerate(refine_candidates, start=1):
                    edge = refiner.refine(cand)
                    if edge is not None:
                        loop_edges.append(edge)
                        _log(
                            f"accepted loop {edge.src_frame}->{edge.dst_frame} "
                            f"score={edge.score:.3f} valid={edge.inliers}",
                            online_cfg.verbose,
                        )
                    if idx == 1 or idx == len(refine_candidates) or idx % 5 == 0:
                        _log(
                            f"sequence {seq.name}: refine {idx}/{len(refine_candidates)} accepted={len(loop_edges)}",
                            online_cfg.verbose,
                        )
            refine_sec = time.perf_counter() - refine_t0
            raw_loop_edges = loop_edges
            loop_edges = _filter_loop_edges_by_score(
                loop_edges,
                online_cfg.loop_edge_score_threshold,
            )
            if len(loop_edges) != len(raw_loop_edges):
                _log(
                    f"sequence {seq.name}: filtered loop edges "
                    f"{len(raw_loop_edges)} -> {len(loop_edges)} "
                    f"with score >= {online_cfg.loop_edge_score_threshold}",
                    online_cfg.verbose,
                )
            pgo_cfg = _build_loop_cfg(online_cfg)
            odom_edges = build_keyframe_odometry_edges(online_c2w)
            pgo_t0 = time.perf_counter()
            loop_c2w = optimize_keyframe_pose_graph(online_c2w, odom_edges, loop_edges, pgo_cfg)
            pgo_sec = time.perf_counter() - pgo_t0
            loop_w2c = _c2w_to_w2c(loop_c2w)

            seq_dir = os.path.join(out_root, seq.name)
            pose_dir = os.path.join(seq_dir, "poses")
            plot_dir = os.path.join(seq_dir, "plots")
            loop_dir = os.path.join(seq_dir, "loop_online_reinfer")
            _ensure_dir(pose_dir)
            _ensure_dir(plot_dir)
            _ensure_dir(loop_dir)

            frame_ids = list(range(frames))
            save_w2c_txt(os.path.join(pose_dir, "loop_abs_pose.txt"), loop_w2c, frame_ids)
            if not os.path.exists(os.path.join(pose_dir, "abs_pose.txt")):
                save_w2c_txt(os.path.join(pose_dir, "abs_pose.txt"), online_w2c, frame_ids)
            if not os.path.exists(os.path.join(pose_dir, "intri.txt")):
                save_intri_txt(os.path.join(pose_dir, "intri.txt"), online_intri_np, frame_ids)

            with open(os.path.join(loop_dir, "candidates.json"), "w") as f:
                json.dump([_candidate_to_json(c) for c in candidates], f, indent=2)
            save_loop_edges_json(os.path.join(loop_dir, "loop_edges.json"), loop_edges)

            eval_t0 = time.perf_counter()
            run_eval = bool(output_cfg.get("run_eval", True))
            save_plots = bool(output_cfg.get("save_plots", True))
            gt = None
            gt_metrics = {"no_loop": None, "online_loop_reinfer": None}
            if run_eval:
                gt = load_ground_truth_trajectory(
                    seq_dir,
                    gt_data_cfg,
                    verbose=online_cfg.verbose,
                    seq_name_hint=seq.name,
                )
                gt_metrics = {
                    "no_loop": evaluate_trajectory_against_gt(online_c2w, frame_ids, gt),
                    "online_loop_reinfer": evaluate_trajectory_against_gt(loop_c2w, frame_ids, gt),
                }
            plot_path = None
            plane_plot_path = None
            if save_plots:
                plot_path = os.path.join(plot_dir, "trajectory_online_loop_reinfer.png")
                plot_trajectory_panels(
                    plot_path,
                    baseline_c2w=online_c2w,
                    method_results={"online_loop_reinfer": loop_c2w},
                    loop_edges={"online_loop_reinfer": loop_edges},
                    frame_ids=frame_ids,
                    gt_metrics=gt_metrics,
                )
                plane_plot_path = os.path.join(plot_dir, "trajectory_online_loop_reinfer_xy_xz.png")
                plot_online_loop_planes(
                    plane_plot_path,
                    baseline_c2w=online_c2w,
                    loop_c2w=loop_c2w,
                    frame_ids=frame_ids,
                    loop_edges=loop_edges,
                    gt_metrics=gt_metrics,
                )
            eval_sec = time.perf_counter() - eval_t0
            pc_sec = None

            seq_summary = {
                "sequence": seq.name,
                "sequence_dir": seq_dir,
                "num_frames": int(frames),
                "num_candidates": int(len(candidates)),
                "num_refined_candidates": int(len(refine_candidates)),
                "num_loop_edges": int(len(loop_edges)),
                "forward_time_sec": float(forward_time),
                "elapsed_sec": float(time.perf_counter() - seq_t0),
                "timing_sec": {
                    "dataloader_wait": float(loader_wait_sec),
                    "image_cpu_copy": float(cpu_copy_sec),
                    "retrieval": float(retrieval_sec),
                    "refine": float(refine_sec),
                    "pgo": float(pgo_sec),
                    "pointcloud_rebuild": pc_sec,
                    "eval_and_plot": float(eval_sec),
                },
                "pointcloud_rebuild_pending": online_output_root is not None,
                "pose_paths": {
                    "online": os.path.join(pose_dir, "abs_pose.txt"),
                    "loop": os.path.join(pose_dir, "loop_abs_pose.txt"),
                },
                "plot_path": plot_path,
                "plane_plot_path": plane_plot_path,
                "gt_pose": {
                    "available": gt is not None,
                    "no_loop": compact_gt_metrics(gt_metrics["no_loop"]),
                    "online_loop_reinfer": compact_gt_metrics(gt_metrics["online_loop_reinfer"]),
                },
            }
            with open(os.path.join(seq_dir, "summary.json"), "w") as f:
                json.dump(seq_summary, f, indent=2)
            summaries["sequences"][seq.name] = seq_summary
            with open(root_summary_path, "w") as f:
                json.dump(summaries, f, indent=2)
            _log(
                f"sequence {seq.name}: wrote pose summary before pointcloud "
                f"eval={eval_sec:.2f}s summary={os.path.join(seq_dir, 'summary.json')}",
                online_cfg.verbose,
            )

            if online_output_root is not None:
                online_seq_dir = os.path.join(online_output_root, seq.name)
                pc_t0 = time.perf_counter()
                _rebuild_pointcloud_from_online_depth(
                    online_output_seq_dir=online_seq_dir,
                    lc_seq_dir=seq_dir,
                    loop_w2c=loop_w2c,
                    loop_intri=online_intri_np,
                    frame_ids=frame_ids,
                    output_cfg=output_cfg,
                    verbose=online_cfg.verbose,
                )
                pc_sec = time.perf_counter() - pc_t0
                seq_summary["timing_sec"]["pointcloud_rebuild"] = float(pc_sec)
                seq_summary["pointcloud_rebuild_pending"] = False
                seq_summary["elapsed_sec"] = float(time.perf_counter() - seq_t0)
                with open(os.path.join(seq_dir, "summary.json"), "w") as f:
                    json.dump(seq_summary, f, indent=2)
                summaries["sequences"][seq.name] = seq_summary
                with open(root_summary_path, "w") as f:
                    json.dump(summaries, f, indent=2)
            else:
                pc_sec = 0.0
                seq_summary["timing_sec"]["pointcloud_rebuild"] = float(pc_sec)
                seq_summary["pointcloud_rebuild_pending"] = False
                with open(os.path.join(seq_dir, "summary.json"), "w") as f:
                    json.dump(seq_summary, f, indent=2)
                summaries["sequences"][seq.name] = seq_summary
                with open(root_summary_path, "w") as f:
                    json.dump(summaries, f, indent=2)
            prev_seq_done_t = time.perf_counter()

    with open(root_summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    return summaries
