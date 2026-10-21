"""3D camera trail and frustum overlays."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from ..core.configuration import OverlaySettings


def parse_color(value: str):
    if not value:
        return None
    names = {
        "black": "#000000", "white": "#ffffff", "red": "#ff0000",
        "green": "#00ff00", "blue": "#0000ff", "cyan": "#00ffff",
        "yellow": "#ffff00", "magenta": "#ff00ff", "orange": "#ffa500",
    }
    value = names.get(value.strip().lower(), value.strip())
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    raise ValueError(f"Cannot parse color {value!r}")


def _ramp(name: str, age: int, tail_len: int, total: int) -> np.ndarray:
    if name == "rainbow":
        r, g, b = colorsys.hsv_to_rgb((age / max(total - 1, 1)) * 0.8, 0.9, 1.0)
        return np.asarray([r, g, b], dtype=np.float64)
    if name == "white":
        v = 1.0 - 0.7 * (age / max(total - 1, 1))
        return np.asarray([v, v, v], dtype=np.float64)
    t = age / max(tail_len - 1, 1)
    return np.asarray([0.0, 1.0 - 0.5 * min(t, 1.0), 1.0], dtype=np.float64)


@dataclass
class CameraPathOverlay:
    c2w: np.ndarray
    intrinsics: np.ndarray
    settings: OverlaySettings

    def geometries(self, frame_idx: int, frame_start: int = 0) -> list[dict]:
        if not self.settings.camera_path_style:
            return []
        frame_idx = min(frame_idx, len(self.c2w) - 1)
        out = []
        positions = self.c2w[frame_start:frame_idx + 1, :3, 3]
        if self.settings.trail_enabled and len(positions) >= 2:
            out.append(self._trail(positions))
        if self.settings.camera_path_style == "frustum":
            out.extend(self._frustums(frame_idx, frame_start))
        return out

    def _trail(self, positions: np.ndarray) -> dict:
        points = positions.astype(np.float64)
        segs = np.asarray([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
        total = len(segs)
        colors = np.asarray([
            _ramp(self.settings.trail_color, total - 1 - i, self.settings.trail_length, total)
            for i in range(total)
        ], dtype=np.float64)
        return {
            "type": "lines",
            "name": "camera_trail",
            "points": points,
            "segments": segs,
            "colors": colors,
            "line_width": self.settings.trail_width,
        }

    def _frustums(self, frame_idx: int, frame_start: int) -> list[dict]:
        start = max(frame_start, frame_idx + 1 - self.settings.frustum_count)
        end = min(frame_idx + 1, len(self.c2w))
        total = max(end - start, 1)
        all_points = []
        all_segments = []
        all_colors = []
        offset = 0
        for fi in range(start, end):
            c2w = self.c2w[fi]
            K = self.intrinsics[fi]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            w, h = cx * 2, cy * 2
            s = self.settings.frustum_scale
            corners_cam = np.asarray([
                [-cx / fx, -cy / fy, 1.0],
                [(w - cx) / fx, -cy / fy, 1.0],
                [(w - cx) / fx, (h - cy) / fy, 1.0],
                [-cx / fx, (h - cy) / fy, 1.0],
            ], dtype=np.float64) * s
            R = c2w[:3, :3].astype(np.float64)
            t = c2w[:3, 3].astype(np.float64)
            corners_world = corners_cam @ R.T + t
            points = np.vstack([t[None], corners_world])
            lines = np.asarray([
                [0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],
            ], dtype=np.int32)
            age = frame_idx - fi
            brightness = 1.0 - 0.6 * (age / max(total - 1, 1))
            color = _ramp(self.settings.trail_color, 0, self.settings.trail_length, max(frame_idx, 1))
            all_points.append(points)
            all_segments.append(lines + offset)
            all_colors.append(np.tile(color * brightness, (len(lines), 1)))
            offset += len(points)
        if not all_points:
            return []
        return [{
            "type": "lines",
            "name": "camera_frustums",
            "points": np.vstack(all_points),
            "segments": np.vstack(all_segments),
            "colors": np.vstack(all_colors),
            "line_width": self.settings.frustum_line_width,
        }]
