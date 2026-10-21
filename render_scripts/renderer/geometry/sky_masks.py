"""Optional ONNX sky segmentation for depth masking."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from tqdm.auto import tqdm


_INPUT_SIZE = (320, 320)
_SOFT_THRESHOLD = 0.1


def _prepare_rgb(image_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_rgb, _INPUT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (resized / 255.0 - mean) / std
    return x.transpose(2, 0, 1)


def _normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    return ((raw - lo) / max(hi - lo, 1e-8)).astype(np.float32)


def _save_visualization(image: np.ndarray, sky_mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (sky_mask.astype(np.uint8) * 255)
    mask_rgb = np.repeat(mask_u8[..., None], 3, axis=2)
    overlay = image.astype(np.float32).copy()
    overlay[sky_mask] = overlay[sky_mask] * 0.35 + np.array([255, 64, 64], dtype=np.float32) * 0.65
    panel = np.concatenate([image, mask_rgb, overlay.clip(0, 255).astype(np.uint8)], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def make_sky_masks(
    images: np.ndarray,
    model_path: str,
    batch_size: int,
    visualization_dir: str = "",
    log: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """Return boolean sky masks, or None if ONNX/model is unavailable."""
    log = log or (lambda msg: None)
    model = Path(model_path)
    if not model.is_file():
        log(f"[sky] model not found: {model}; sky mask skipped")
        return None
    try:
        import onnxruntime
    except ImportError:
        log("[sky] onnxruntime not installed; sky mask skipped")
        return None

    available = set(onnxruntime.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    session = onnxruntime.InferenceSession(str(model), providers=providers)
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    model_batch = model_input.shape[0]
    batch_size = max(1, int(batch_size))
    if isinstance(model_batch, int) and model_batch > 0:
        batch_size = model_batch
    output_name = session.get_outputs()[0].name
    H, W = images.shape[1:3]
    masks = []
    viz_root = Path(visualization_dir) if visualization_dir else None

    for start in tqdm(range(0, len(images), batch_size), desc="Sky masks", unit="batch"):
        batch = images[start:start + batch_size]
        inp = np.stack([_prepare_rgb(img) for img in batch]).astype(np.float32)
        if isinstance(model_batch, int) and inp.shape[0] < model_batch:
            inp = np.concatenate([inp, np.repeat(inp[-1:], model_batch - len(inp), axis=0)])
        raw = np.asarray(session.run([output_name], {input_name: inp}))
        while raw.ndim > 3 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        if raw.ndim == 4:
            raw = raw[:, -1]
        if raw.ndim == 2:
            raw = raw[None]
        for j in range(len(batch)):
            score = _normalize(raw[j])
            score = cv2.resize(score, (W, H), interpolation=cv2.INTER_LINEAR)
            sky = (1.0 - score) <= _SOFT_THRESHOLD
            masks.append(sky)
            if viz_root is not None:
                _save_visualization(batch[j], sky, viz_root / f"frame_{start + j:06d}.png")

    return np.stack(masks, axis=0).astype(bool)
