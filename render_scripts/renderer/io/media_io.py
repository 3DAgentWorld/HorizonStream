"""Frame and video output helpers."""

from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def write_rgb_png(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def encode_png_sequence(frame_dir: str | Path, output_path: str | Path, fps: int = 30) -> None:
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    pattern = str(frame_dir / "frame_%06d.png")
    if shutil.which("ffmpeg"):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            return

    files = sorted(glob.glob(str(frame_dir / "frame_*.png")))
    if not files:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")
    first = cv2.imread(files[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for file in files:
        frame = cv2.imread(file)
        if frame is not None:
            writer.write(frame)
    writer.release()


def encode_bgr_frames(frames: list[np.ndarray], output_path: str | Path, fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
