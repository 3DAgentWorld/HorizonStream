"""Load scene reconstruction sources."""

from __future__ import annotations

import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from ..core.scene_graph import FrameSequence


_NPZ_KEYS = {
    "images", "depth", "depth_conf", "confidence", "intrinsic", "extrinsic",
}


def _window_items(items, skip_first: int = 0, max_items: int = 0, step: int = 1):
    skip_first = max(0, int(skip_first))
    max_items = max(0, int(max_items))
    step = max(1, int(step))
    selected = items[skip_first:]
    if max_items > 0:
        selected = selected[:max_items]
    return selected[::step]


def _read_indexed_rows(path: Path, values_per_row: int) -> Dict[int, np.ndarray]:
    rows: Dict[int, np.ndarray] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != values_per_row + 1:
            raise ValueError(f"Unexpected row in {path}: {line}")
        rows[int(parts[0])] = np.asarray(parts[1:], dtype=np.float32)
    return rows


def _pose_dir(root: Path) -> Optional[Path]:
    for name in ("poses", "poses_online"):
        path = root / name
        if path.is_dir():
            return path
    return None


def _pose_file(root: Path) -> Optional[Path]:
    pose_dir = _pose_dir(root)
    if pose_dir is None:
        return None
    for name in ("loop_abs_pose.txt", "abs_pose.txt"):
        path = pose_dir / name
        if path.is_file():
            return path
    return None


def is_sequence_dir(path: str | Path) -> bool:
    root = Path(path)
    pose_dir = _pose_dir(root)
    return (
        pose_dir is not None
        and (root / "images" / "rgb").is_dir()
        and (root / "depth" / "dpt").is_dir()
        and _pose_file(root) is not None
        and (pose_dir / "intri.txt").is_file()
    )


def _load_frame(job):
    idx, image_path, depth_path, conf_path = job
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    conf = None
    if conf_path is not None:
        conf = np.load(conf_path, allow_pickle=False).astype(np.float32)
        if conf.ndim == 3:
            conf = conf[..., 0]
    return idx, image, depth, conf


def _load_sequence_dir(
    path: str | Path,
    workers: int = 16,
    skip_first: int = 0,
    max_items: int = 0,
    step: int = 1,
) -> FrameSequence:
    root = Path(path)
    image_dir = root / "images" / "rgb"
    depth_dir = root / "depth" / "dpt"
    conf_dir = root / "depth" / "conf"
    if not conf_dir.is_dir():
        conf_dir = None

    pose_path = _pose_file(root)
    pose_dir = _pose_dir(root)
    if pose_path is None or pose_dir is None:
        raise ValueError(f"{root} does not contain pose files")

    pose_rows = _read_indexed_rows(pose_path, 12)
    intri_rows = _read_indexed_rows(pose_dir / "intri.txt", 4)

    jobs = []
    for image_path in sorted(image_dir.glob("frame_*.png")):
        idx = int(image_path.stem.split("_")[-1])
        depth_path = depth_dir / f"frame_{idx:06d}.npy"
        if not depth_path.exists() or idx not in pose_rows or idx not in intri_rows:
            continue
        conf_path = None
        if conf_dir is not None:
            candidate = conf_dir / f"frame_{idx:06d}.npy"
            conf_path = candidate if candidate.exists() else None
        jobs.append((idx, image_path, depth_path, conf_path))
    jobs = _window_items(jobs, skip_first, max_items, step)
    if not jobs:
        raise ValueError(f"No matching frames found in {root}")

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        frames = list(pool.map(_load_frame, jobs))
    frames.sort(key=lambda item: item[0])

    frame_ids = np.asarray([item[0] for item in frames], dtype=np.int32)
    images = np.stack([item[1] for item in frames], axis=0)
    depths = np.stack([item[2] for item in frames], axis=0)
    conf_items = [item[3] for item in frames]
    confidence = None
    if conf_dir is not None and all(c is not None for c in conf_items):
        confidence = np.stack(conf_items, axis=0)

    K = np.zeros((len(frame_ids), 3, 3), dtype=np.float32)
    w2c = np.zeros((len(frame_ids), 4, 4), dtype=np.float32)
    w2c[:, 3, 3] = 1.0
    for out_i, frame_id in enumerate(frame_ids):
        fx, fy, cx, cy = intri_rows[int(frame_id)]
        K[out_i] = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        pose = pose_rows[int(frame_id)]
        w2c[out_i, :3, :3] = pose[:9].reshape(3, 3)
        w2c[out_i, :3, 3] = pose[9:12]

    return FrameSequence(images, depths, _w2c_to_c2w(w2c), K, confidence, frame_ids)


def _load_npz_file(
    path: str | Path,
    skip_first: int = 0,
    max_items: int = 0,
    step: int = 1,
) -> FrameSequence:
    loaded = np.load(path, allow_pickle=False)
    raw = {key: loaded[key] for key in loaded.files if key in _NPZ_KEYS}
    sequence = _parse_npz(raw)
    indices = np.arange(sequence.num_frames)
    return sequence.subset(_window_items(indices, skip_first, max_items, step))


def _load_npz_dir(
    path: str | Path,
    workers: int = 16,
    skip_first: int = 0,
    max_items: int = 0,
    step: int = 1,
) -> FrameSequence:
    files = sorted(glob.glob(str(Path(path) / "frame_*.npz")))
    files = _window_items(files, skip_first, max_items, step)
    if not files:
        raise ValueError(f"No frame_*.npz files found in {path}")

    def _load_one(p):
        data = np.load(p, allow_pickle=False)
        return {key: data[key] for key in data.files if key in _NPZ_KEYS}

    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        parts = list(pool.map(_load_one, files))
    raw = {key: np.stack([part[key] for part in parts], axis=0) for key in parts[0].keys()}
    return _parse_npz(raw)


def _parse_npz(raw: dict[str, np.ndarray]) -> FrameSequence:
    images = raw["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.dtype != np.uint8:
        images = images.astype(np.float32)
        if images.max() <= 1.0:
            images *= 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)

    depths = raw["depth"].astype(np.float32)
    if depths.ndim == 4:
        depths = depths[..., 0]

    K = raw["intrinsic"].astype(np.float32)
    if K.ndim == 2:
        K = np.tile(K[None], (len(images), 1, 1))

    ext = raw["extrinsic"].astype(np.float32)
    w2c = np.zeros((ext.shape[0], 4, 4), dtype=np.float32)
    w2c[:, :3, :] = ext[:, :3, :]
    w2c[:, 3, 3] = 1.0

    confidence = None
    for key in ("depth_conf", "confidence"):
        if key in raw:
            confidence = raw[key].astype(np.float32)
            if confidence.ndim == 4:
                confidence = confidence[..., 0]
            break
    frame_ids = np.arange(len(images), dtype=np.int32)
    return FrameSequence(images, depths, _w2c_to_c2w(w2c), K, confidence, frame_ids)


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3:4]
    Rt = R.transpose(0, 2, 1)
    c2w = np.zeros_like(w2c, dtype=np.float32)
    c2w[:, :3, :3] = Rt
    c2w[:, :3, 3:4] = -Rt @ t
    c2w[:, 3, 3] = 1.0
    return c2w


def load_scene_source(
    path: str | Path,
    workers: int = 16,
    skip_first: int = 0,
    max_items: int = 0,
    step: int = 1,
) -> FrameSequence:
    path = Path(path)
    if path.is_dir() and is_sequence_dir(path):
        return _load_sequence_dir(path, workers, skip_first, max_items, step)
    if path.is_dir():
        return _load_npz_dir(path, workers, skip_first, max_items, step)
    return _load_npz_file(path, skip_first, max_items, step)
