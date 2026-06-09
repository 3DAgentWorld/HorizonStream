import json
import math
import os
import time
import contextlib
import io
import warnings
from dataclasses import asdict, dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from horizonstream.data import HorizonStreamDataLoader, HorizonStreamSequenceInfo
from horizonstream.eval.io import frame_stems, read_opencv_camera_yml, read_pred_w2c_txt
from horizonstream.eval.metrics import ate_rmse, transform_points
from horizonstream.loop.pypose_optimizer import optimize_pose_graph_pypose
from horizonstream.streaming.keyframe_selector import KeyframeSelector


@dataclass
class LoopConfig:
    verbose: bool = True
    pose_graph_backend: str = "pypose"
    pose_graph_model: str = "se3"
    pose_graph_update_mode: str = "all"
    pose_graph_solver_verbose: bool = True
    keyframe_stride: int = 1
    min_keyframe_gap: int = 10
    dbow_temporal_exclusion: int = 10
    loop_chunk_size: int = 20
    retrieval_top_k: int = 5
    dbow_num_repeat: int = 3
    retrieval_score_thresh_dbow: float = 0.034
    retrieval_score_thresh_salad: float = 0.85
    nms_radius: int = 25
    max_candidates_per_method: int = 1000
    max_verified_loops_per_method: int = 1000
    dbow_vocab_path: Optional[str] = None
    salad_ckpt_path: str = "weights/dino_salad.ckpt"
    salad_dino_weights_path: str = "weights/dinov2_vitb14_pretrain.pth"
    salad_backbone: str = "dinov2_vitb14"
    salad_image_size: Tuple[int, int] = (336, 336)
    salad_batch_size: int = 32
    orb_features: int = 4096
    patch_match_thresh: float = 0.40
    max_pair_matches: int = 256
    orb_ratio_test: float = 0.80
    rigid_ransac_iters: int = 1
    rigid_inlier_thresh: float = 0.1
    rigid_min_inliers: int = 24
    loop_edge_min_separation: int = 30
    sim3_irls_delta: float = 0.1
    sim3_irls_max_iters: int = 5
    sim3_irls_tol: float = 1e-9
    pose_graph_rot_weight: float = 1.0
    pose_graph_trans_weight: float = 2.0
    pose_graph_scale_weight: float = 1.0
    pose_graph_loop_weight: float = 0.1
    pose_graph_max_nfev: int = 200
    pose_graph_max_iterations: int = 30
    pose_graph_lambda_init: float = 1e-6


def _log(msg: str, enabled: bool = True) -> None:
    if not enabled:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[loop_runtime {stamp}] {msg}", flush=True)


def _log_every(
    label: str,
    index: int,
    total: int,
    enabled: bool = True,
    step: int = 10,
) -> None:
    if not enabled:
        return
    if total <= 0:
        return
    if index == 1 or index == total or index % max(1, step) == 0:
        _log(f"{label}: {index}/{total}", enabled=enabled)


@dataclass
class SequenceArtifacts:
    seq_dir: str
    pose_variant: str
    pose_path: str
    intri_path: str
    image_paths: List[str]
    depth_paths: List[str]
    frame_ids: List[int]
    w2c: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    image_hw: Tuple[int, int]


@dataclass
class LoopCandidate:
    src_pos: int
    dst_pos: int
    src_frame: int
    dst_frame: int
    score: float
    method: str


@dataclass
class LoopEdge:
    src_pos: int
    dst_pos: int
    src_frame: int
    dst_frame: int
    score: float
    inliers: int
    method: str
    transform_ji: np.ndarray


@dataclass
class PoseGraphSummary:
    method: str
    num_candidates: int
    num_verified_loops: int
    mean_loop_score: float
    mean_loop_inliers: float
    trajectory_path: str
    loop_plot_path: str


@dataclass
class GroundTruthTrajectory:
    seq_name: str
    scene_root: str
    camera: Optional[str]
    c2w: np.ndarray
    valid_mask: np.ndarray


def _read_intri_txt(path: str) -> Tuple[List[int], np.ndarray]:
    frames = []
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
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = vals[1]
            K[1, 1] = vals[2]
            K[0, 2] = vals[3]
            K[1, 2] = vals[4]
            frames.append(frame)
            intri.append(K)
    return frames, np.asarray(intri, dtype=np.float64)


def _sorted_frame_files(path_pattern: str) -> List[str]:
    def frame_key(path: str) -> Tuple[int, str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        return (int(digits) if digits else -1, path)

    return sorted(glob(path_pattern), key=frame_key)


def _resolve_pose_and_intri_paths(
    pose_dir: str,
    pose_variant: str = "main",
) -> Tuple[str, str, str]:
    variant = str(pose_variant or "main").strip().lower()
    if variant in {"", "main", "default"}:
        variant = "main"
        pose_path = os.path.join(pose_dir, "abs_pose.txt")
        intri_path = os.path.join(pose_dir, "intri.txt")
    else:
        pose_path = os.path.join(pose_dir, f"{variant}_abs_pose.txt")
        intri_path = os.path.join(pose_dir, f"{variant}_intri.txt")

    if not os.path.exists(pose_path):
        raise FileNotFoundError(
            f"Pose file not found for pose_variant='{variant}': {pose_path}"
        )
    if not os.path.exists(intri_path):
        fallback_intri = os.path.join(pose_dir, "intri.txt")
        if os.path.exists(fallback_intri):
            intri_path = fallback_intri
        else:
            raise FileNotFoundError(
                f"Intrinsic file not found for pose_variant='{variant}': {intri_path}"
            )
    return variant, pose_path, intri_path


def load_sequence_artifacts(
    seq_dir: str,
    image_dir: Optional[str] = None,
    depth_dir: Optional[str] = None,
    pose_variant: str = "main",
) -> SequenceArtifacts:
    pose_dir = os.path.join(seq_dir, "poses")
    image_dir = image_dir or os.path.join(seq_dir, "images", "rgb")
    depth_dir = depth_dir or os.path.join(seq_dir, "depth", "dpt")
    pose_variant, pose_path, intri_path = _resolve_pose_and_intri_paths(
        pose_dir,
        pose_variant=pose_variant,
    )

    frame_ids, w2c_list = read_pred_w2c_txt(pose_path)
    intri_frames, intri = _read_intri_txt(intri_path)
    if frame_ids != intri_frames:
        raise ValueError("Frame ids in abs_pose.txt and intri.txt do not match")

    image_paths = _sorted_frame_files(os.path.join(image_dir, "frame_*.*"))
    depth_paths = _sorted_frame_files(os.path.join(depth_dir, "frame_*.npy"))
    if len(image_paths) != len(frame_ids):
        raise ValueError(
            f"Expected {len(frame_ids)} RGB frames in {image_dir}, found {len(image_paths)}"
        )
    if len(depth_paths) != len(frame_ids):
        raise ValueError(
            f"Expected {len(frame_ids)} depth frames in {depth_dir}, found {len(depth_paths)}"
        )

    sample = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(image_paths[0])
    h, w = sample.shape[:2]
    w2c = np.asarray(w2c_list, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    return SequenceArtifacts(
        seq_dir=seq_dir,
        pose_variant=pose_variant,
        pose_path=pose_path,
        intri_path=intri_path,
        image_paths=image_paths,
        depth_paths=depth_paths,
        frame_ids=frame_ids,
        w2c=w2c,
        c2w=c2w,
        intrinsics=intri,
        image_hw=(h, w),
    )


class SequenceCache:
    def __init__(self, artifacts: SequenceArtifacts):
        self.artifacts = artifacts
        self._rgb: Dict[int, np.ndarray] = {}
        self._gray: Dict[int, np.ndarray] = {}
        self._depth: Dict[int, np.ndarray] = {}

    def rgb(self, frame_idx: int) -> np.ndarray:
        if frame_idx not in self._rgb:
            image = cv2.imread(self.artifacts.image_paths[frame_idx], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.artifacts.image_paths[frame_idx])
            self._rgb[frame_idx] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._rgb[frame_idx]

    def gray(self, frame_idx: int) -> np.ndarray:
        if frame_idx not in self._gray:
            self._gray[frame_idx] = cv2.cvtColor(self.rgb(frame_idx), cv2.COLOR_RGB2GRAY)
        return self._gray[frame_idx]

    def depth(self, frame_idx: int) -> np.ndarray:
        if frame_idx not in self._depth:
            self._depth[frame_idx] = np.load(self.artifacts.depth_paths[frame_idx]).astype(
                np.float32
            )
        return self._depth[frame_idx]


def select_keyframes(num_frames: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    selector = KeyframeSelector(
        min_interval=stride,
        max_interval=stride,
        force_first=True,
        mode="fixed",
    )
    is_keyframe, keyframe_indices = selector.select_keyframes(
        num_frames, batch_size=1, device=torch.device("cpu")
    )
    return is_keyframe[0].numpy(), keyframe_indices[0].numpy()


def pose_matrix_to_vec(pose: np.ndarray) -> np.ndarray:
    rotvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate([rotvec, pose[:3, 3]], axis=0)


def pose_vec_to_matrix(vec: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(vec[:3]).as_matrix()
    pose[:3, 3] = vec[3:6]
    return pose


def project_to_rotation(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = u @ vt
    return rot.astype(np.float64, copy=False)


def similarity_components(mat: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(mat[:3, :3], dtype=np.float64)
    trans = np.asarray(mat[:3, 3], dtype=np.float64)
    col_norms = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(col_norms))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    rot = project_to_rotation(linear / scale)
    return scale, rot, trans


def similarity_matrix(scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = float(scale) * np.asarray(rot, dtype=np.float64)
    out[:3, 3] = np.asarray(trans, dtype=np.float64)
    return out


def normalize_pose_matrix(mat: np.ndarray) -> np.ndarray:
    _, rot, trans = similarity_components(mat)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def relative_measurement_from_c2w(src_c2w: np.ndarray, dst_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(dst_c2w) @ src_c2w


def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    rot = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -(np.transpose(rot, (0, 2, 1)) @ t[..., None])[..., 0]


def _top_down_xy(c2w: np.ndarray) -> np.ndarray:
    xyz = c2w[:, :3, 3]
    return xyz[:, [0, 2]]


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _copy_loop_cfg(cfg: LoopConfig) -> LoopConfig:
    return LoopConfig(**asdict(cfg))


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _resize_rgb_batch(batch: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    return np.stack(
        [
            cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            for image in batch
        ],
        axis=0,
    )


def _batch_to_imagenet_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    images = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device) / 255.0
    return _imagenet_normalize(images)


def salad_log_otp_solver(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    scores: torch.Tensor,
    num_iters: int = 20,
    reg: float = 1.0,
) -> torch.Tensor:
    scores = scores / reg
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def salad_matching_probs(
    scores: torch.Tensor,
    dustbin_score: torch.Tensor,
    num_iters: int = 3,
    reg: float = 1.0,
) -> torch.Tensor:
    batch_size, m, n = scores.size()
    scores_aug = torch.empty(batch_size, m + 1, n, dtype=scores.dtype, device=scores.device)
    scores_aug[:, :m, :n] = scores
    scores_aug[:, m, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m), device=scores.device)
    log_a = norm.expand(m + 1).contiguous()
    log_b = norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)
    return salad_log_otp_solver(log_a, log_b, scores_aug, num_iters=num_iters, reg=reg) - norm


class SaladAggregator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 768,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dropout_layer: torch.nn.Module
        dropout_layer = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.num_channels = int(num_channels)
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.token_features = torch.nn.Sequential(
            torch.nn.Linear(self.num_channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.token_dim),
        )
        self.cluster_features = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features_and_token: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, token = features_and_token
        cluster_features = self.cluster_features(features).flatten(2)
        scores = self.score(features).flatten(2)
        token = self.token_features(token)

        probs = torch.exp(salad_matching_probs(scores, self.dust_bin, 3))
        probs = probs[:, :-1, :]
        probs = probs.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        cluster_features = cluster_features.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        descriptor = torch.cat(
            [
                torch.nn.functional.normalize(token, p=2, dim=-1),
                torch.nn.functional.normalize(
                    (cluster_features * probs).sum(dim=-1), p=2, dim=1
                ).flatten(1),
            ],
            dim=-1,
        )
        return torch.nn.functional.normalize(descriptor, p=2, dim=-1)


class DinoV2BackboneForSalad(torch.nn.Module):
    CHANNELS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str, weights_path: Optional[str], num_trainable_blocks: int = 4):
        super().__init__()
        if model_name not in self.CHANNELS:
            raise ValueError(f"Unsupported SALAD backbone: {model_name}")
        self.model_name = model_name
        self.num_channels = int(self.CHANNELS[model_name])
        self.num_trainable_blocks = int(num_trainable_blocks)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        feature_map = patch_tokens.reshape(
            batch, height // 14, width // 14, self.num_channels
        ).permute(0, 3, 1, 2)
        return feature_map, cls_token, patch_tokens


class SaladVPRModel(torch.nn.Module):
    def __init__(self, backbone_name: str, dino_weights_path: Optional[str]):
        super().__init__()
        self.backbone = DinoV2BackboneForSalad(backbone_name, dino_weights_path)
        self.aggregator = SaladAggregator(
            num_channels=self.backbone.num_channels,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )

    def forward_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map, cls_token, patch_tokens = self.backbone(images)
        descriptor = self.aggregator((feature_map, cls_token))
        return descriptor, torch.nn.functional.normalize(patch_tokens, dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        descriptor, _ = self.forward_features(images)
        return descriptor


def _load_salad_state_dict(model: torch.nn.Module, ckpt_path: str) -> List[str]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"SALAD checkpoint not found: {ckpt_path}. "
            "Download it from https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
        )
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state.items():
        out_key = str(key)
        for prefix in ("model.", "module."):
            if out_key.startswith(prefix):
                out_key = out_key[len(prefix) :]
        cleaned[out_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = list(missing)
    missing_aggregator = [key for key in missing if key.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            "SALAD checkpoint did not provide the expected aggregator weights: "
            + ", ".join(missing_aggregator[:5])
        )
    if unexpected:
        _log(
            "SALAD: ignored unexpected checkpoint keys: "
            + ", ".join(str(key) for key in unexpected[:5]),
            enabled=True,
        )
    return missing


_POPCOUNT_TABLE = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)


class FlatORBVocabulary:
    def __init__(self, path: str):
        self.path = path
        self.centers, self.idf = self._load(path)
        self.num_words = int(self.centers.shape[0])

    @staticmethod
    def _load(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            data = np.load(path)
            if "centers" not in data:
                raise ValueError(f"Expected 'centers' in vocabulary file: {path}")
            centers = np.asarray(data["centers"], dtype=np.float32)
            idf = np.asarray(data["idf"], dtype=np.float32) if "idf" in data else None
        elif ext == ".npy":
            centers = np.asarray(np.load(path), dtype=np.float32)
            idf = None
        else:
            try:
                centers = np.asarray(np.loadtxt(path), dtype=np.float32)
            except Exception as exc:
                raise ValueError(
                    f"Unsupported ORB vocabulary text format in {path}. "
                    "This loader expects a flat centers matrix (.npz/.npy/.txt), not a hierarchical DBoW2 ORBvoc.txt."
                ) from exc
            idf = None

        if centers.ndim != 2 or centers.shape[1] != 32:
            raise ValueError(
                f"Expected ORB vocabulary centers with shape [K, 32], got {tuple(centers.shape)} from {path}"
            )
        if idf is not None:
            if idf.ndim != 1 or idf.shape[0] != centers.shape[0]:
                raise ValueError(
                    f"Expected idf shape [{centers.shape[0]}], got {tuple(idf.shape)} from {path}"
                )
        return centers.astype(np.float32, copy=False), idf

    def quantize(self, desc: np.ndarray) -> np.ndarray:
        diff = desc.astype(np.float32, copy=False)[:, None, :] - self.centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        return dist2.argmin(axis=1).astype(np.int64, copy=False)

    def histogram(self, desc: Optional[np.ndarray]) -> np.ndarray:
        hist = np.zeros(self.num_words, dtype=np.float32)
        if desc is None or len(desc) == 0:
            return hist
        words = self.quantize(desc)
        np.add.at(hist, words, 1.0)
        if self.idf is not None:
            hist *= self.idf
        return _normalize_rows(hist[None] + 1e-12)[0]

    def score_matrix(self, hists: np.ndarray) -> np.ndarray:
        hists = _normalize_rows(hists.astype(np.float32))
        return hists @ hists.T


class DBoWTextVocabulary:
    def __init__(self, path: str):
        self.path = path
        self.nodes: List[Dict[str, object]] = [
            {
                "id": 0,
                "parent": -1,
                "children": [],
                "descriptor": None,
                "weight": 0.0,
                "word_id": -1,
                "is_leaf": False,
            }
        ]
        self.word_node_ids: List[int] = []
        self.k = 0
        self.L = 0
        self.scoring = 0
        self.weighting = 0
        self._load(path)
        self.num_words = len(self.word_node_ids)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            header = f.readline().strip().split()
            if len(header) != 4:
                raise ValueError(f"Invalid ORBvoc header in {path}: {' '.join(header)}")
            self.k, self.L, self.scoring, self.weighting = [int(x) for x in header]
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 35:
                    raise ValueError(
                        f"Expected 35 fields per ORBvoc node, got {len(parts)} in {path}"
                    )
                parent = int(parts[0])
                is_leaf = int(parts[1]) > 0
                descriptor = np.asarray([int(x) for x in parts[2:34]], dtype=np.uint8)
                weight = float(parts[34])
                node_id = len(self.nodes)
                node = {
                    "id": node_id,
                    "parent": parent,
                    "children": [],
                    "descriptor": descriptor,
                    "weight": weight,
                    "word_id": -1,
                    "is_leaf": is_leaf,
                }
                self.nodes.append(node)
                self.nodes[parent]["children"].append(node_id)
                if is_leaf:
                    node["word_id"] = len(self.word_node_ids)
                    self.word_node_ids.append(node_id)

        if self.weighting != 0 or self.scoring != 0:
            raise ValueError(
                f"Unsupported ORBvoc scoring/weighting ({self.scoring}, {self.weighting}) in {path}. "
                "This loader currently supports the standard TF_IDF + L1_NORM ORBvoc only."
            )

    def _hamming_distance(self, desc: np.ndarray, child_descs: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(desc[:, None, :], child_descs[None, :, :])
        return _POPCOUNT_TABLE[xor].sum(axis=2)

    def quantize(self, desc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        current = np.zeros(len(desc), dtype=np.int64)
        while True:
            next_nodes = current.copy()
            active = False
            for parent_id in np.unique(current):
                idx = np.flatnonzero(current == parent_id)
                if idx.size == 0:
                    continue
                children = self.nodes[parent_id]["children"]
                if not children:
                    continue
                active = True
                child_descs = np.stack(
                    [self.nodes[child_id]["descriptor"] for child_id in children], axis=0
                )
                dist = self._hamming_distance(desc[idx], child_descs)
                best = dist.argmin(axis=1)
                next_nodes[idx] = np.asarray(children, dtype=np.int64)[best]
            current = next_nodes
            if not active:
                break
            if all(bool(self.nodes[node_id]["is_leaf"]) for node_id in np.unique(current)):
                break

        word_ids = np.asarray(
            [int(self.nodes[node_id]["word_id"]) for node_id in current], dtype=np.int64
        )
        weights = np.asarray(
            [float(self.nodes[node_id]["weight"]) for node_id in current], dtype=np.float32
        )
        return word_ids, weights

    def sparse_histogram(self, desc: Optional[np.ndarray]) -> Dict[int, float]:
        hist: Dict[int, float] = {}
        if desc is None or len(desc) == 0:
            return hist
        word_ids, weights = self.quantize(desc)
        for word_id, weight in zip(word_ids.tolist(), weights.tolist()):
            hist[int(word_id)] = hist.get(int(word_id), 0.0) + float(weight)
        total = float(sum(hist.values()))
        if total > 0:
            hist = {word_id: value / total for word_id, value in hist.items()}
        return hist

    def score_matrix(self, hists: Sequence[Dict[int, float]]) -> np.ndarray:
        n = len(hists)
        sim = np.eye(n, dtype=np.float32)
        for i in range(n):
            hi = hists[i]
            for j in range(i):
                hj = hists[j]
                if len(hi) > len(hj):
                    hi, hj = hj, hi
                score = 0.0
                for word_id, value in hi.items():
                    other = hj.get(word_id)
                    if other is not None:
                        score += min(value, other)
                sim[i, j] = score
                sim[j, i] = score
        return sim


def _load_orb_vocabulary(path: str):
    if os.path.basename(path) == "ORBvoc.txt":
        return DBoWTextVocabulary(path)
    return FlatORBVocabulary(path)


def _resolve_gt_sequence_info(
    seq_dir: str,
    data_cfg: Optional[dict],
    seq_name_hint: Optional[str] = None,
) -> Optional[HorizonStreamSequenceInfo]:
    if not data_cfg:
        return None
    seq_names = []
    if seq_name_hint:
        seq_names.append(str(seq_name_hint).replace("\\", "/").strip("/"))
    seq_names.append(str(os.path.basename(os.path.normpath(seq_dir))).replace("\\", "/").strip("/"))
    preferred_camera = data_cfg.get("camera", None)
    if preferred_camera is not None:
        preferred_camera = str(preferred_camera).strip()
    loader = HorizonStreamDataLoader(dict(data_cfg))
    prefixed_matches: List[HorizonStreamSequenceInfo] = []
    suffixed_matches: List[HorizonStreamSequenceInfo] = []
    for seq_info in loader.iter_sequence_infos():
        name_norm = str(seq_info.name).replace("\\", "/").strip("/")
        for seq_name_norm in seq_names:
            if name_norm == seq_name_norm:
                return seq_info
        parts = [part for part in name_norm.split("/") if part]
        for seq_name_norm in seq_names:
            if parts and parts[0] == seq_name_norm:
                prefixed_matches.append(seq_info)
            if parts and parts[-1] == seq_name_norm:
                suffixed_matches.append(seq_info)
    candidates = prefixed_matches or suffixed_matches
    if not candidates:
        return None

    if preferred_camera:
        for seq_info in candidates:
            if str(seq_info.camera or "") == preferred_camera:
                return seq_info

    for seq_info in candidates:
        if str(seq_info.camera or "") == "02":
            return seq_info

    candidates = sorted(candidates, key=lambda item: str(item.name))
    if candidates:
        return candidates[0]
    return None


def load_ground_truth_trajectory(
    seq_dir: str,
    data_cfg: Optional[dict],
    verbose: bool = True,
    seq_name_hint: Optional[str] = None,
) -> Optional[GroundTruthTrajectory]:
    seq_info = _resolve_gt_sequence_info(seq_dir, data_cfg, seq_name_hint=seq_name_hint)
    if seq_info is None:
        _log("GT: no matching sequence info found from data config", verbose)
        return None

    if seq_info.camera is not None:
        cam_dir = os.path.join(seq_info.scene_root, "cameras", seq_info.camera)
        extri_path = os.path.join(cam_dir, "extri.yml")
        intri_path = os.path.join(cam_dir, "intri.yml")
    else:
        extri_path = os.path.join(seq_info.scene_root, "extri.yml")
        intri_path = os.path.join(seq_info.scene_root, "intri.yml")
    if not os.path.exists(extri_path):
        _log(f"GT: missing pose file {extri_path}", verbose)
        return None

    extri, _, _ = read_opencv_camera_yml(extri_path, intri_path)
    stems = frame_stems(seq_info.image_paths)
    gt_c2w = []
    valid_mask = []
    for stem in stems:
        if stem in extri:
            gt_c2w.append(np.linalg.inv(extri[stem]))
            valid_mask.append(True)
        else:
            gt_c2w.append(np.eye(4, dtype=np.float64))
            valid_mask.append(False)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        _log(
            f"GT: pose file {extri_path} does not contain any matching frame ids",
            verbose,
        )
        return None
    _log(
        f"GT: loaded {int(valid_mask.sum())}/{len(valid_mask)} poses from {seq_info.scene_root}"
        + (f" camera={seq_info.camera}" if seq_info.camera else ""),
        verbose,
    )
    return GroundTruthTrajectory(
        seq_name=seq_info.name,
        scene_root=seq_info.scene_root,
        camera=seq_info.camera,
        c2w=np.asarray(gt_c2w, dtype=np.float64),
        valid_mask=valid_mask,
    )


def evaluate_trajectory_against_gt(
    pred_c2w: np.ndarray,
    frame_ids: Sequence[int],
    gt: Optional[GroundTruthTrajectory],
) -> Optional[Dict[str, object]]:
    if gt is None:
        return None

    pred_xyz = []
    gt_xyz = []
    used_frame_ids = []
    for pred_idx, frame_id in enumerate(frame_ids):
        if frame_id < 0 or frame_id >= len(gt.c2w):
            continue
        if not gt.valid_mask[frame_id]:
            continue
        pred_xyz.append(pred_c2w[pred_idx, :3, 3])
        gt_xyz.append(gt.c2w[frame_id, :3, 3])
        used_frame_ids.append(int(frame_id))
    if len(pred_xyz) < 3:
        return None

    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    metrics = ate_rmse(pred_xyz, gt_xyz, align_scale=True)
    scale = float(metrics["sim3_scale"])
    rot = np.asarray(metrics["sim3_rotation"], dtype=np.float64)
    trans = np.asarray(metrics["sim3_translation"], dtype=np.float64)
    pred_xyz_aligned = transform_points(pred_xyz, scale, rot, trans)
    metrics["frame_ids"] = used_frame_ids
    metrics["pred_xyz_aligned"] = pred_xyz_aligned.tolist()
    metrics["gt_xyz"] = gt_xyz.tolist()
    return metrics


def compact_gt_metrics(metrics: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if metrics is None:
        return None
    out = dict(metrics)
    out.pop("pred_xyz_aligned", None)
    out.pop("gt_xyz", None)
    out.pop("frame_ids", None)
    return out


class ORBBoWRetriever:
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        if not cfg.dbow_vocab_path:
            raise ValueError("LoopConfig.dbow_vocab_path must be set for ORB-BoW retrieval")
        self.orb = cv2.ORB_create(
            nfeatures=cfg.orb_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=20,
        )
        self.keypoints: Dict[int, List[cv2.KeyPoint]] = {}
        self.descriptors: Dict[int, Optional[np.ndarray]] = {}
        self.histograms = None
        self.vocab = _load_orb_vocabulary(cfg.dbow_vocab_path)
        _log(
            f"ORB-BoW: loaded fixed vocabulary {cfg.dbow_vocab_path} "
            f"with {self.vocab.num_words} words",
            self.cfg.verbose,
        )

    def fit(self, cache: SequenceCache, keyframe_ids: Sequence[int]) -> np.ndarray:
        _log(
            f"ORB-BoW: extracting ORB descriptors for {len(keyframe_ids)} keyframes",
            self.cfg.verbose,
        )
        total = len(keyframe_ids)
        for idx, frame_idx in enumerate(keyframe_ids, start=1):
            gray = cache.gray(frame_idx)
            kps, desc = self.orb.detectAndCompute(gray, None)
            self.keypoints[frame_idx] = kps or []
            self.descriptors[frame_idx] = desc
            _log_every("ORB keyframes", idx, total, self.cfg.verbose, step=20)

        if not any(desc is not None and len(desc) for desc in self.descriptors.values()):
            if isinstance(self.vocab, DBoWTextVocabulary):
                self.histograms = [{} for _ in keyframe_ids]
            else:
                self.histograms = np.zeros(
                    (len(keyframe_ids), int(self.vocab.num_words)), dtype=np.float32
                )
            _log("ORB-BoW: no descriptors found, returning empty histograms", self.cfg.verbose)
            return self.histograms

        if isinstance(self.vocab, DBoWTextVocabulary):
            hists = []
            for frame_idx in keyframe_ids:
                desc = self.descriptors[frame_idx]
                hists.append(self.vocab.sparse_histogram(desc))
            self.histograms = hists
            mean_nnz = float(np.mean([len(hist) for hist in hists])) if hists else 0.0
            _log(
                f"ORB-BoW: built sparse descriptor bank for {len(hists)} keyframes "
                f"(mean nnz={mean_nnz:.1f})",
                self.cfg.verbose,
            )
        else:
            hists = []
            for frame_idx in keyframe_ids:
                desc = self.descriptors[frame_idx]
                hists.append(self.vocab.histogram(desc))
            self.histograms = np.asarray(hists, dtype=np.float32)
            _log(
                f"ORB-BoW: built descriptor bank with shape {tuple(self.histograms.shape)}",
                self.cfg.verbose,
            )
        return self.histograms

    def score_matrix(self) -> np.ndarray:
        if self.histograms is None:
            raise RuntimeError("ORB-BoW histograms are not initialized")
        return self.vocab.score_matrix(self.histograms)

    def collect_3d_correspondences(
        self,
        cache: SequenceCache,
        artifacts: SequenceArtifacts,
        src_frame: int,
        dst_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        desc_src = self.descriptors.get(src_frame)
        desc_dst = self.descriptors.get(dst_frame)
        if desc_src is None or desc_dst is None or len(desc_src) < 8 or len(desc_dst) < 8:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_a = matcher.knnMatch(desc_src, desc_dst, k=2)
        knn_b = matcher.knnMatch(desc_dst, desc_src, k=2)

        def ratio_filter(knn_matches):
            passed = {}
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.cfg.orb_ratio_test * n.distance:
                    passed[m.queryIdx] = (m.trainIdx, float(m.distance))
            return passed

        forward = ratio_filter(knn_a)
        backward = ratio_filter(knn_b)
        matches = []
        for q_idx, (t_idx, dist) in forward.items():
            rev = backward.get(t_idx)
            if rev is not None and rev[0] == q_idx:
                matches.append((q_idx, t_idx, dist))
        if len(matches) < self.cfg.rigid_min_inliers:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        kps_src = self.keypoints[src_frame]
        kps_dst = self.keypoints[dst_frame]
        uv_src = np.asarray([kps_src[q].pt for q, _, _ in matches], dtype=np.float32)
        uv_dst = np.asarray([kps_dst[t].pt for _, t, _ in matches], dtype=np.float32)
        return depth_correspondences_to_3d(
            uv_src,
            uv_dst,
            cache.depth(src_frame),
            cache.depth(dst_frame),
            artifacts.intrinsics[src_frame],
            artifacts.intrinsics[dst_frame],
        )


class SaladRetriever:
    def __init__(self, cfg: LoopConfig, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        if not os.path.exists(cfg.salad_ckpt_path):
            raise FileNotFoundError(
                f"SALAD checkpoint not found: {cfg.salad_ckpt_path}. "
                "Download it from https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
            )
        self.model = SaladVPRModel(
            backbone_name=cfg.salad_backbone,
            dino_weights_path=cfg.salad_dino_weights_path,
        )
        missing = _load_salad_state_dict(self.model, cfg.salad_ckpt_path)
        if any(key.startswith("backbone.") for key in missing) and not os.path.exists(
            cfg.salad_dino_weights_path
        ):
            raise FileNotFoundError(
                f"DINOv2 backbone weights not found: {cfg.salad_dino_weights_path}. "
                "Download it from https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
            )
        self.model = self.model.to(self.device).eval()
        self.global_desc: Optional[np.ndarray] = None
        self.local_desc: Dict[int, np.ndarray] = {}
        self.patch_hw: Optional[Tuple[int, int]] = None
        self.patch_size = 14

    def fit(self, cache: SequenceCache, keyframe_ids: Sequence[int]) -> np.ndarray:
        batch_size = max(1, int(self.cfg.salad_batch_size))
        image_size = tuple(int(x) for x in self.cfg.salad_image_size)
        globals_out = []
        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(keyframe_ids), batch_size), start=1):
                batch_ids = list(keyframe_ids[start : start + batch_size])
                batch = np.stack([cache.rgb(i) for i in batch_ids], axis=0)
                resized = _resize_rgb_batch(batch, image_size)
                salad_images = _batch_to_imagenet_tensor(resized, self.device)
                local_images = _batch_to_imagenet_tensor(batch, self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    desc = self.model(salad_images)
                    _, patch_tokens = self.model.forward_features(local_images)
                globals_out.append(desc.float().cpu().numpy().astype(np.float32))
                patch_tokens_np = patch_tokens.float().cpu().numpy().astype(np.float32)
                for frame_idx, patch_desc in zip(batch_ids, patch_tokens_np):
                    self.local_desc[frame_idx] = patch_desc
                if self.patch_hw is None:
                    h, w = batch.shape[1:3]
                    self.patch_hw = (h // self.patch_size, w // self.patch_size)

        self.global_desc = np.concatenate(globals_out, axis=0)
        self.global_desc = _normalize_rows(self.global_desc.astype(np.float32))
        _log(
            f"SALAD: built descriptor bank with shape {tuple(self.global_desc.shape)}",
            self.cfg.verbose,
        )
        return self.global_desc

    def collect_3d_correspondences(
        self,
        cache: SequenceCache,
        artifacts: SequenceArtifacts,
        src_frame: int,
        dst_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        desc_src = self.local_desc.get(src_frame)
        desc_dst = self.local_desc.get(dst_frame)
        if desc_src is None or desc_dst is None:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        sim = desc_src @ desc_dst.T
        dst_best = sim.argmax(axis=1)
        dst_score = sim[np.arange(sim.shape[0]), dst_best]
        src_best = sim.argmax(axis=0)
        matches = []
        for src_idx, dst_idx in enumerate(dst_best.tolist()):
            if src_best[dst_idx] != src_idx:
                continue
            score = float(dst_score[src_idx])
            if score < self.cfg.patch_match_thresh:
                continue
            matches.append((src_idx, dst_idx, score))
        if len(matches) < self.cfg.rigid_min_inliers:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        matches.sort(key=lambda item: item[2], reverse=True)
        matches = matches[: self.cfg.max_pair_matches]
        uv_src = np.asarray(
            [patch_index_to_uv(m[0], self.patch_hw, self.patch_size) for m in matches],
            dtype=np.float32,
        )
        uv_dst = np.asarray(
            [patch_index_to_uv(m[1], self.patch_hw, self.patch_size) for m in matches],
            dtype=np.float32,
        )
        return depth_correspondences_to_3d(
            uv_src,
            uv_dst,
            cache.depth(src_frame),
            cache.depth(dst_frame),
            artifacts.intrinsics[src_frame],
            artifacts.intrinsics[dst_frame],
        )


def patch_index_to_uv(index: int, patch_hw: Tuple[int, int], patch_size: int) -> Tuple[float, float]:
    h_patch, w_patch = patch_hw
    y = index // w_patch
    x = index % w_patch
    cx = (x + 0.5) * patch_size
    cy = (y + 0.5) * patch_size
    return float(cx), float(cy)


def sample_depth_at_uv(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    x = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
    return depth[y, x]


def unproject_points(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = depth.astype(np.float64)
    x = (uv[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (uv[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1] * z
    return np.stack([x, y, z], axis=1)


def depth_correspondences_to_3d(
    uv_src: np.ndarray,
    uv_dst: np.ndarray,
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intr_src: np.ndarray,
    intr_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    z_src = sample_depth_at_uv(depth_src, uv_src)
    z_dst = sample_depth_at_uv(depth_dst, uv_dst)
    valid = np.isfinite(z_src) & np.isfinite(z_dst) & (z_src > 1e-4) & (z_dst > 1e-4)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    pts_src = unproject_points(uv_src[valid], z_src[valid], intr_src)
    pts_dst = unproject_points(uv_dst[valid], z_dst[valid], intr_dst)
    return pts_src, pts_dst


def estimate_similarity_umeyama(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    allow_scale: bool = True,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Expected src_pts and dst_pts with shape [N, 3]")
    if len(src) < 3:
        raise ValueError("At least 3 points are required")

    if weights is None:
        weights = np.ones(len(src), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != src.shape[0]:
            raise ValueError("weights must have shape [N]")
    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()

    src_mean = np.sum(src * weights[:, None], axis=0)
    dst_mean = np.sum(dst * weights[:, None], axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (weights[:, None] * dst_centered).T @ src_centered
    u, singular_vals, vt = np.linalg.svd(cov)
    s_mat = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    rot = u @ s_mat @ vt

    if allow_scale:
        src_var = float(np.sum(weights * np.sum(src_centered * src_centered, axis=1)))
        if src_var < 1e-12:
            raise ValueError("Degenerate source variance for Sim3 estimation")
        scale = float(np.sum(singular_vals * np.diag(s_mat)) / src_var)
    else:
        scale = 1.0

    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot.astype(np.float64, copy=False), trans.astype(np.float64, copy=False)


def apply_similarity_to_points(
    pts: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    return scale * (np.asarray(rot, dtype=np.float64) @ np.asarray(pts, dtype=np.float64).T).T + np.asarray(
        trans, dtype=np.float64
    )


def robust_refine_similarity_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cfg: LoopConfig,
) -> np.ndarray:
    allow_scale = cfg.pose_graph_model == "sim3"
    weights = np.ones(len(src_pts), dtype=np.float64)
    scale, rot, trans = estimate_similarity_umeyama(
        src_pts, dst_pts, allow_scale=allow_scale, weights=weights
    )
    prev_cost = float("inf")
    delta = float(cfg.sim3_irls_delta)

    for _ in range(max(1, int(cfg.sim3_irls_max_iters))):
        pred = apply_similarity_to_points(src_pts, scale, rot, trans)
        resid = np.linalg.norm(pred - dst_pts, axis=1)
        huber = np.ones_like(resid)
        large = resid > delta
        huber[large] = delta / np.clip(resid[large], 1e-12, None)
        huber = np.clip(huber, 1e-6, None)
        scale_new, rot_new, trans_new = estimate_similarity_umeyama(
            src_pts, dst_pts, allow_scale=allow_scale, weights=huber
        )
        current_cost = float(
            np.sum(
                np.where(
                    resid <= delta,
                    0.5 * resid * resid,
                    delta * (resid - 0.5 * delta),
                )
            )
        )
        param_change = abs(scale_new - scale) + np.linalg.norm(trans_new - trans)
        rot_angle = Rotation.from_matrix(rot_new @ rot.T).magnitude()
        scale, rot, trans = scale_new, rot_new, trans_new
        if (
            param_change < float(cfg.sim3_irls_tol)
            and rot_angle < np.deg2rad(0.1)
        ) or abs(prev_cost - current_cost) < float(cfg.sim3_irls_tol) * max(prev_cost, 1.0):
            break
        prev_cost = current_cost

    return similarity_matrix(scale, rot, trans)


def estimate_similarity_transform_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cfg: LoopConfig,
) -> Optional[Tuple[np.ndarray, int]]:
    if len(src_pts) < cfg.rigid_min_inliers:
        return None

    allow_scale = cfg.pose_graph_model == "sim3"
    rng = np.random.default_rng(0)
    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    sample_size = 3

    for _ in range(cfg.rigid_ransac_iters):
        sample = rng.choice(len(src_pts), size=sample_size, replace=False)
        try:
            scale, rot, trans = estimate_similarity_umeyama(
                src_pts[sample], dst_pts[sample], allow_scale=allow_scale
            )
        except ValueError:
            continue
        pred = apply_similarity_to_points(src_pts, scale, rot, trans)
        err = np.linalg.norm(pred - dst_pts, axis=1)
        inliers = err < cfg.rigid_inlier_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_inliers = inliers
            best_count = count

    if best_inliers is None or best_count < cfg.rigid_min_inliers:
        return None

    refined = robust_refine_similarity_transform(src_pts[best_inliers], dst_pts[best_inliers], cfg)
    return refined, best_count


def retrieve_dbow_candidates(
    sim: np.ndarray,
    keyframe_ids: Sequence[int],
    cfg: LoopConfig,
) -> List[LoopCandidate]:
    candidates: List[LoopCandidate] = []
    found: List[Tuple[int, int, float]] = []
    accepted_pairs: List[Tuple[int, int]] = []
    repeat = max(1, int(cfg.dbow_num_repeat))
    temporal_exclusion = max(1, int(cfg.dbow_temporal_exclusion))
    nms = max(0, int(cfg.nms_radius))

    for src_pos in range(len(keyframe_ids)):
        max_dst = src_pos - temporal_exclusion
        if max_dst < 0:
            continue
        prev_scores = sim[src_pos, : max_dst + 1]
        dst_pos = int(np.argmax(prev_scores))
        score = float(prev_scores[dst_pos])
        if score < cfg.retrieval_score_thresh_dbow:
            continue

        if accepted_pairs and nms > 0:
            dists_sq = [
                (src_pos - prev_src) ** 2 + (dst_pos - prev_dst) ** 2
                for prev_src, prev_dst in accepted_pairs
            ]
            if min(dists_sq, default=float("inf")) < nms * nms:
                continue

        found.append((src_pos, dst_pos, score))
        if len(found) < repeat:
            continue

        latest = found[-repeat:]
        first_src = latest[0][0]
        if (1 + src_pos - first_src) != repeat:
            continue
        accepted_src, accepted_dst, accepted_score = latest[repeat // 2]
        accepted_pairs.append((accepted_src, accepted_dst))
        candidates.append(
            LoopCandidate(
                src_pos=int(accepted_src),
                dst_pos=int(max(accepted_dst, 1)),
                src_frame=int(keyframe_ids[accepted_src]),
                dst_frame=int(keyframe_ids[max(accepted_dst, 1)]),
                score=float(accepted_score),
                method="dbow",
            )
        )
        found.clear()

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: cfg.max_candidates_per_method]


def non_max_suppress_loop_candidates(
    candidates: Sequence[LoopCandidate],
    nms_threshold: int,
    limit: int,
) -> List[LoopCandidate]:
    if not candidates or nms_threshold <= 0:
        return list(candidates)[:limit]
    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    max_pos = max(max(item.src_pos, item.dst_pos) for item in sorted_candidates)
    filtered: List[LoopCandidate] = []
    suppressed = set()
    for cand in sorted_candidates:
        low = min(cand.src_pos, cand.dst_pos)
        high = max(cand.src_pos, cand.dst_pos)
        if low in suppressed or high in suppressed:
            continue
        filtered.append(cand)

        start1 = max(0, low - nms_threshold)
        end1 = min(low + nms_threshold + 1, high)
        suppressed.update(range(start1, end1))

        start2 = max(low + 1, high - nms_threshold)
        end2 = min(high + nms_threshold + 1, max_pos + 1)
        suppressed.update(range(start2, end2))
        if len(filtered) >= limit:
            break
    return filtered


def retrieve_vpr_candidates(
    descriptors: np.ndarray,
    keyframe_ids: Sequence[int],
    method: str,
    score_thresh: float,
    cfg: LoopConfig,
) -> List[LoopCandidate]:
    desc = _normalize_rows(descriptors.astype(np.float32))
    sim = desc @ desc.T
    top_k = max(1, int(cfg.retrieval_top_k))
    min_gap = max(1, int(cfg.min_keyframe_gap))
    candidates_by_key: Dict[Tuple[int, int], LoopCandidate] = {}
    for src_pos in range(len(keyframe_ids)):
        order = np.argsort(sim[src_pos])[::-1]
        keep = 0
        for dst_pos in order.tolist():
            if dst_pos == src_pos:
                continue
            if keep >= top_k:
                break
            score = float(sim[src_pos, dst_pos])
            if score <= score_thresh:
                break
            keep += 1
            if abs(int(keyframe_ids[src_pos]) - int(keyframe_ids[dst_pos])) <= min_gap:
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
                src_frame=int(keyframe_ids[high]),
                dst_frame=int(keyframe_ids[low]),
                score=score,
                method=method,
            )

    return non_max_suppress_loop_candidates(
        list(candidates_by_key.values()),
        nms_threshold=max(0, int(cfg.nms_radius)),
        limit=max(1, int(cfg.max_candidates_per_method)),
    )


def centered_window_positions(center: int, size: int, total: int) -> np.ndarray:
    size = max(1, int(size))
    total = max(0, int(total))
    if total <= 0:
        return np.empty((0,), dtype=np.int64)
    if size >= total:
        return np.arange(total, dtype=np.int64)
    half = size // 2
    start = center - half
    end = start + size
    if start < 0:
        start = 0
        end = size
    if end > total:
        end = total
        start = total - size
    return np.arange(start, end, dtype=np.int64)


def candidate_chunk_frame_pairs(
    candidate: LoopCandidate,
    keyframe_ids: Sequence[int],
    chunk_size: int,
) -> List[Tuple[int, int]]:
    src_pos = int(candidate.src_pos)
    dst_pos = int(candidate.dst_pos)
    src_window = centered_window_positions(src_pos, chunk_size, len(keyframe_ids))
    dst_window = centered_window_positions(dst_pos, chunk_size, len(keyframe_ids))
    span = min(len(src_window), len(dst_window))
    if span <= 0:
        return []
    return [
        (int(keyframe_ids[src_window[idx]]), int(keyframe_ids[dst_window[idx]]))
        for idx in range(span)
    ]


def verify_candidates(
    candidates: Sequence[LoopCandidate],
    verifier,
    cache: SequenceCache,
    artifacts: SequenceArtifacts,
    keyframe_ids: Sequence[int],
    cfg: LoopConfig,
) -> List[LoopEdge]:
    method = candidates[0].method if candidates else "unknown"
    _log(
        f"{method}: verifying {len(candidates)} loop candidates with geometry",
        cfg.verbose,
    )
    verified: List[LoopEdge] = []
    total = len(candidates)
    for idx, cand in enumerate(candidates, start=1):
        if abs(int(cand.src_pos) - int(cand.dst_pos)) < int(cfg.loop_edge_min_separation):
            _log_every(
                f"{method} verify",
                idx,
                total,
                cfg.verbose,
                step=5,
            )
            continue
        frame_pairs = candidate_chunk_frame_pairs(cand, keyframe_ids, cfg.loop_chunk_size)
        pts_src_all = []
        pts_dst_all = []
        for src_frame, dst_frame in frame_pairs:
            pts_src, pts_dst = verifier.collect_3d_correspondences(
                cache, artifacts, src_frame, dst_frame
            )
            if len(pts_src) == 0:
                continue
            pts_src_all.append(pts_src)
            pts_dst_all.append(pts_dst)
        if not pts_src_all:
            _log_every(
                f"{method} verify",
                idx,
                total,
                cfg.verbose,
                step=5,
            )
            continue
        pts_src = np.concatenate(pts_src_all, axis=0)
        pts_dst = np.concatenate(pts_dst_all, axis=0)
        result = estimate_similarity_transform_ransac(pts_src, pts_dst, cfg)
        if result is None:
            _log_every(
                f"{method} verify",
                idx,
                total,
                cfg.verbose,
                step=5,
            )
            continue
        transform_ji, inliers = result
        verified.append(
            LoopEdge(
                src_pos=cand.src_pos,
                dst_pos=cand.dst_pos,
                src_frame=cand.src_frame,
                dst_frame=cand.dst_frame,
                score=cand.score,
                inliers=inliers,
                method=cand.method,
                transform_ji=transform_ji,
            )
        )
        _log(
            f"{method}: accepted loop frame {cand.src_frame} -> {cand.dst_frame} "
            f"(score={cand.score:.3f}, inliers={inliers}, chunk={max(1, cfg.loop_chunk_size)})",
            cfg.verbose,
        )
        _log_every(
            f"{method} verify",
            idx,
            total,
            cfg.verbose,
            step=5,
        )
    verified.sort(key=lambda edge: (edge.inliers, edge.score), reverse=True)
    verified = verified[: cfg.max_verified_loops_per_method]
    _log(
        f"{method}: kept {len(verified)} verified loops after ranking",
        cfg.verbose,
    )
    return verified


def build_keyframe_odometry_edges(keyframe_c2w: np.ndarray) -> List[LoopEdge]:
    edges = []
    for idx in range(1, len(keyframe_c2w)):
        meas = relative_measurement_from_c2w(
            keyframe_c2w[idx - 1], keyframe_c2w[idx]
        )
        edges.append(
            LoopEdge(
                src_pos=idx - 1,
                dst_pos=idx,
                src_frame=idx - 1,
                dst_frame=idx,
                score=1.0,
                inliers=0,
                method="odometry",
                transform_ji=meas,
            )
        )
    return edges


def optimize_keyframe_pose_graph(
    keyframe_c2w_init: np.ndarray,
    odom_edges: Sequence[LoopEdge],
    loop_edges: Sequence[LoopEdge],
    cfg: LoopConfig,
) -> np.ndarray:
    if cfg.pose_graph_backend == "pypose":
        all_edges = list(odom_edges) + list(loop_edges)
        edge_weights = []
        for edge in all_edges:
            if edge.method == "odometry":
                edge_weights.append(1.0)
            else:
                edge_weights.append(
                    cfg.pose_graph_loop_weight
                    * math.sqrt(max(edge.inliers, 1) / max(cfg.rigid_min_inliers, 1))
                )
        return optimize_pose_graph_pypose(
            keyframe_c2w_init=keyframe_c2w_init,
            edge_src=[edge.src_pos for edge in all_edges],
            edge_dst=[edge.dst_pos for edge in all_edges],
            edge_measurements=[edge.transform_ji for edge in all_edges],
            edge_weights=edge_weights,
            model=cfg.pose_graph_model,
            update_mode=cfg.pose_graph_update_mode,
            trans_weight=cfg.pose_graph_trans_weight,
            rot_weight=cfg.pose_graph_rot_weight,
            scale_weight=cfg.pose_graph_scale_weight,
            max_iterations=cfg.pose_graph_max_iterations,
            lambda_init=cfg.pose_graph_lambda_init,
            verbose=cfg.pose_graph_solver_verbose,
            log_fn=lambda msg: _log(msg, cfg.verbose),
        )
    if cfg.pose_graph_backend != "scipy":
        raise ValueError(f"Unsupported pose_graph_backend: {cfg.pose_graph_backend}")
    if cfg.pose_graph_model != "se3":
        raise ValueError("SciPy pose graph backend only supports pose_graph_model='se3'")

    if len(keyframe_c2w_init) <= 1:
        return keyframe_c2w_init.copy()
    num_nodes = len(keyframe_c2w_init)
    num_vars = (num_nodes - 1) * 6
    num_edges = len(odom_edges) + len(loop_edges)
    num_residuals = num_edges * 6
    _log(
        f"Pose graph: optimizing {num_nodes} keyframes with "
        f"{len(odom_edges)} odom edges and {len(loop_edges)} loop edges "
        f"({num_vars} vars, {num_residuals} residuals)",
        cfg.verbose,
    )
    x0 = np.concatenate(
        [pose_matrix_to_vec(pose) for pose in keyframe_c2w_init[1:]], axis=0
    )

    def unpack_poses(x: np.ndarray) -> List[np.ndarray]:
        poses = [keyframe_c2w_init[0]]
        for idx in range(0, len(x), 6):
            poses.append(pose_vec_to_matrix(x[idx : idx + 6]))
        return poses

    def edge_residual(edge: LoopEdge, poses: Sequence[np.ndarray], loop_scale: float) -> np.ndarray:
        pred = np.linalg.inv(poses[edge.dst_pos]) @ poses[edge.src_pos]
        rot_err = Rotation.from_matrix(
            edge.transform_ji[:3, :3].T @ pred[:3, :3]
        ).as_rotvec()
        trans_err = pred[:3, 3] - edge.transform_ji[:3, 3]
        if edge.method == "odometry":
            w = 1.0
        else:
            w = loop_scale * math.sqrt(max(edge.inliers, 1) / max(cfg.rigid_min_inliers, 1))
        return np.concatenate(
            [
                rot_err * cfg.pose_graph_rot_weight * w,
                trans_err * cfg.pose_graph_trans_weight * w,
            ],
            axis=0,
        )

    def residual_fn(x: np.ndarray) -> np.ndarray:
        poses = unpack_poses(x)
        residuals = []
        for edge in odom_edges:
            residuals.append(edge_residual(edge, poses, loop_scale=1.0))
        for edge in loop_edges:
            residuals.append(edge_residual(edge, poses, cfg.pose_graph_loop_weight))
        return np.concatenate(residuals, axis=0)

    def build_jac_sparsity() -> lil_matrix:
        sparsity = lil_matrix((num_residuals, num_vars), dtype=np.int8)
        row = 0
        for edge in list(odom_edges) + list(loop_edges):
            for node_idx in (edge.src_pos, edge.dst_pos):
                if node_idx <= 0:
                    continue
                col0 = (node_idx - 1) * 6
                sparsity[row : row + 6, col0 : col0 + 6] = 1
            row += 6
        return sparsity

    jac_sparsity = build_jac_sparsity()
    _log(
        "Pose graph: built sparse Jacobian pattern for finite differences",
        cfg.verbose,
    )

    result = least_squares(
        residual_fn,
        x0,
        loss="huber",
        f_scale=1.0,
        max_nfev=cfg.pose_graph_max_nfev,
        jac_sparsity=jac_sparsity,
        x_scale="jac",
        verbose=2 if cfg.pose_graph_solver_verbose else 0,
    )
    poses = unpack_poses(result.x)
    _log(
        f"Pose graph: done status={result.status} cost={float(result.cost):.4f} "
        f"nfev={int(result.nfev)}",
        cfg.verbose,
    )
    return np.asarray(poses, dtype=np.float64)


def propagate_keyframe_corrections(
    c2w_base: np.ndarray,
    keyframe_ids: Sequence[int],
    frame_ref_keyframe_pos: np.ndarray,
    keyframe_c2w_opt: np.ndarray,
) -> np.ndarray:
    corrected = c2w_base.copy()
    keyframe_base = c2w_base[np.asarray(keyframe_ids)]
    correction = np.asarray(
        [
            keyframe_c2w_opt[idx] @ np.linalg.inv(keyframe_base[idx])
            for idx in range(len(keyframe_ids))
        ],
        dtype=np.float64,
    )
    for frame_idx in range(len(c2w_base)):
        ref_pos = int(frame_ref_keyframe_pos[frame_idx])
        corrected[frame_idx] = normalize_pose_matrix(correction[ref_pos] @ c2w_base[frame_idx])
    return corrected


def frame_to_keyframe_position(
    keyframe_mask: np.ndarray, keyframe_indices: np.ndarray
) -> np.ndarray:
    key_positions = np.cumsum(keyframe_mask.astype(np.int64)) - 1
    out = np.zeros_like(keyframe_indices, dtype=np.int64)
    for idx in range(len(keyframe_indices)):
        if keyframe_mask[idx]:
            out[idx] = key_positions[idx]
        else:
            ref_frame = int(keyframe_indices[idx])
            out[idx] = key_positions[ref_frame]
    return out


def save_w2c_txt(path: str, w2c: np.ndarray, frame_ids: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# w2c\n")
        for idx, frame_id in enumerate(frame_ids):
            row = [frame_id] + w2c[idx, :3, :3].reshape(-1).tolist() + w2c[idx, :3, 3].tolist()
            f.write(" ".join(map(str, row)) + "\n")


def save_loop_edges_json(path: str, edges: Sequence[LoopEdge]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = []
    for edge in edges:
        item = asdict(edge)
        item["transform_ji"] = edge.transform_ji.tolist()
        payload.append(item)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _xy_from_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[:, [0, 2]]


def plot_trajectory_panels(
    output_path: str,
    baseline_c2w: np.ndarray,
    method_results: Dict[str, np.ndarray],
    loop_edges: Dict[str, Sequence[LoopEdge]],
    frame_ids: Sequence[int],
    gt_metrics: Optional[Dict[str, Dict[str, object]]] = None,
    selected_loop_weights: Optional[Dict[str, float]] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extra_names = [name for name in method_results.keys() if name != "no_loop"]
    names = ["no_loop", *extra_names]
    titles = {
        "no_loop": "No Loop",
        "dbow": "ORB-BoW",
        "salad": "SALAD",
    }
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    base_metrics = (gt_metrics or {}).get("no_loop")
    if base_metrics is not None:
        base_xy = _xy_from_xyz(np.asarray(base_metrics["pred_xyz_aligned"], dtype=np.float64))
    else:
        base_xy = _top_down_xy(baseline_c2w)

    for ax, name in zip(axes, names):
        method_metric = (gt_metrics or {}).get(name)
        if method_metric is not None:
            xy = _xy_from_xyz(np.asarray(method_metric["pred_xyz_aligned"], dtype=np.float64))
            gt_xy = _xy_from_xyz(np.asarray(method_metric["gt_xyz"], dtype=np.float64))
            frame_id_to_xy = {
                int(frame_id): xy[idx]
                for idx, frame_id in enumerate(method_metric["frame_ids"])
            }
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#2ca02c", linewidth=2.0, label="gt")
            if name != "no_loop":
                ax.plot(
                    base_xy[:, 0],
                    base_xy[:, 1],
                    color="#808080",
                    linewidth=1.6,
                    linestyle="--",
                    label="no_loop",
                )
        else:
            traj = method_results.get(name, baseline_c2w)
            xy = _top_down_xy(traj)
            gt_xy = None
            frame_id_to_xy = {
                int(frame_id): xy[idx] for idx, frame_id in enumerate(frame_ids)
            }
            if name != "no_loop":
                ax.plot(
                    base_xy[:, 0],
                    base_xy[:, 1],
                    color="#808080",
                    linewidth=1.6,
                    linestyle="--",
                    label="no_loop",
                )
        ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=2.0, label=name)
        if name != "no_loop":
            for edge in loop_edges.get(name, []):
                a = frame_id_to_xy.get(int(edge.src_frame))
                b = frame_id_to_xy.get(int(edge.dst_frame))
                if a is None or b is None:
                    continue
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color="#d62728",
                    alpha=0.35,
                    linewidth=1.0,
                )
        ax.set_aspect("equal")
        title = titles.get(name, name)
        if method_metric is not None:
            title += f"\nATE {float(method_metric['ate_rmse']):.2f} m"
        if name != "no_loop" and selected_loop_weights is not None and name in selected_loop_weights:
            title += f", w={float(selected_loop_weights[name]):.2f}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    output_path: str,
    baseline_c2w: np.ndarray,
    method_results: Dict[str, np.ndarray],
    gt_metrics: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 7))
    if gt_metrics and gt_metrics.get("no_loop") is not None:
        gt_xy = _xy_from_xyz(
            np.asarray(gt_metrics["no_loop"]["gt_xyz"], dtype=np.float64)
        )
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="gt", linewidth=2.0, color="#2ca02c")
        color_map = {
            "no_loop": "#808080",
            "dbow": "#1f77b4",
            "salad": "#9467bd",
        }
        for name in ["no_loop", *[key for key in method_results.keys() if key != "no_loop"]]:
            metric = gt_metrics.get(name)
            if metric is None:
                continue
            xy = _xy_from_xyz(np.asarray(metric["pred_xyz_aligned"], dtype=np.float64))
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                label=name,
                linewidth=2.0,
                color=color_map.get(name, None),
            )
    else:
        entries = [(baseline_c2w, "no_loop", "#808080")]
        if "dbow" in method_results:
            entries.append((method_results["dbow"], "orb_bow", "#1f77b4"))
        if "salad" in method_results:
            entries.append((method_results["salad"], "salad", "#9467bd"))
        for c2w, label, color in entries:
            xy = _top_down_xy(c2w)
            plt.plot(xy[:, 0], xy[:, 1], label=label, linewidth=2.0, color=color)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Offline Loop Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def summarize_loop_edges(
    method: str,
    num_candidates: int,
    edges: Sequence[LoopEdge],
    trajectory_path: str,
    loop_plot_path: str,
) -> PoseGraphSummary:
    return PoseGraphSummary(
        method=method,
        num_candidates=num_candidates,
        num_verified_loops=len(edges),
        mean_loop_score=float(np.mean([edge.score for edge in edges])) if edges else 0.0,
        mean_loop_inliers=float(np.mean([edge.inliers for edge in edges])) if edges else 0.0,
        trajectory_path=trajectory_path,
        loop_plot_path=loop_plot_path,
    )
