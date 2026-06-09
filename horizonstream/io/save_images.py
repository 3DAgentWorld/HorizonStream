import os
import subprocess
import tempfile
from glob import glob
from shutil import copy2, which
from typing import List

import numpy as np
from PIL import Image


def save_image_sequence(
    path, images: List[np.ndarray], prefix: str = "frame", ext: str = "png"
):
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(path, f"{prefix}_{i:06d}.{ext}")
        Image.fromarray(img).save(out_path)


def _ffmpeg_bin() -> str:
    explicit = os.environ.get("LONGSTREAM_FFMPEG_PATH", "")
    if explicit:
        return explicit
    system_ffmpeg = which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return ""


def save_video(output_path, pattern, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timeout_sec = int(os.environ.get("LONGSTREAM_FFMPEG_TIMEOUT_SEC", "900"))
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        print(
            f"[horizonstream] warning: ffmpeg not found; skip video export {output_path}",
            flush=True,
        )
        return
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec)
        return
    except FileNotFoundError:
        print(
            f"[horizonstream] warning: ffmpeg executable not found; skip video export {output_path}",
            flush=True,
        )
        return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Fallback: filter out broken PNG frames, rebuild a clean contiguous sequence, and retry.
    frame_paths = sorted(glob(pattern))
    valid_paths = []
    bad_count = 0
    for p in frame_paths:
        try:
            if os.path.getsize(p) <= 0:
                bad_count += 1
                continue
            with Image.open(p) as im:
                im.verify()
            valid_paths.append(p)
        except Exception:
            bad_count += 1

    if not valid_paths:
        raise RuntimeError(f"No valid frames found for video export: pattern={pattern}")

    with tempfile.TemporaryDirectory(prefix="horizonstream_video_") as tmpdir:
        for i, src in enumerate(valid_paths):
            dst = os.path.join(tmpdir, f"frame_{i:06d}.png")
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                copy2(src, dst)

        retry_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(retry_cmd, check=True, timeout=timeout_sec)
        except FileNotFoundError:
            print(
                f"[horizonstream] warning: ffmpeg executable not found; skip video export {output_path}",
                flush=True,
            )
            return

    if bad_count > 0:
        print(
            f"[horizonstream] video export recovered: skipped {bad_count} broken frame(s) for {output_path}",
            flush=True,
        )
