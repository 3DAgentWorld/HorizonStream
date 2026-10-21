"""Simple top-down minimap renderer."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .camera_timeline import global_up
from ..core.configuration import MiniMapSettings


def _rgb(name: str) -> tuple[int, int, int]:
    names = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128)}
    if name.lower() in names:
        return names[name.lower()]
    if name.startswith("#"):
        h = name[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 255, 255, 255


def _rainbow(t: float, sat: float, val: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(t * 0.8, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


class MiniMapRenderer:
    def __init__(self, c2w: np.ndarray, settings: MiniMapSettings):
        self.c2w = c2w
        self.settings = settings
        self.positions = c2w[:, :3, 3].astype(np.float32)
        up = global_up(c2w)
        direction = self.positions[-1] - self.positions[0]
        axis_x = direction - float(np.dot(direction, up)) * up
        if np.linalg.norm(axis_x) < 1e-6:
            axis_x = np.array([1, 0, 0], dtype=np.float32)
        axis_x = axis_x / max(np.linalg.norm(axis_x), 1e-6)
        axis_y = np.cross(up, axis_x)
        axis_y = axis_y / max(np.linalg.norm(axis_y), 1e-6)
        if settings.rotation_degrees % 360 == 90:
            axis_x, axis_y = axis_y, -axis_x
        elif settings.rotation_degrees % 360 == 180:
            axis_x, axis_y = -axis_x, -axis_y
        elif settings.rotation_degrees % 360 == 270:
            axis_x, axis_y = -axis_y, axis_x
        xy = np.stack([self.positions @ axis_x, self.positions @ axis_y], axis=1)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        center = (lo + hi) * 0.5
        span = np.maximum(hi - lo, 1e-3)
        scale = min(settings.width * (1 - 2 * settings.margin) / span[0],
                    settings.height * (1 - 2 * settings.margin) / span[1])
        self.points = (xy - center) * scale + np.array([settings.width * 0.5, settings.height * 0.5])
        self.axis_x = axis_x
        self.axis_y = axis_y

    def render(self, frame_idx: int) -> np.ndarray:
        idx = min(max(frame_idx, 0), len(self.points) - 1)
        bg = _rgb(self.settings.background)
        img = np.full((self.settings.height, self.settings.width, 3), bg, dtype=np.uint8)
        end = max(idx, 1)
        for i in range(1, end + 1):
            t = i / max(len(self.points) - 1, 1)
            color = _rainbow(t, self.settings.trail_saturation, self.settings.trail_brightness)
            p0 = tuple(np.round(self.points[i - 1]).astype(int))
            p1 = tuple(np.round(self.points[i]).astype(int))
            cv2.line(img, p0, p1, color[::-1], self.settings.trail_width, cv2.LINE_AA)
        p = tuple(np.round(self.points[idx]).astype(int))
        cv2.circle(img, p, self.settings.marker_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, self.settings.marker_radius, (40, 40, 40), 2, cv2.LINE_AA)
        return img
