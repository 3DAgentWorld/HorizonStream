"""Open3D offscreen rendering engine."""

from __future__ import annotations

import contextlib
import os

import cv2
import numpy as np

from .camera_timeline import ViewCamera
from .render_overlays import parse_color
from ..core.configuration import RenderSettings


_OPEN3D = None


@contextlib.contextmanager
def _quiet_c_stdio():
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _open3d():
    global _OPEN3D
    if _OPEN3D is None:
        with _quiet_c_stdio():
            _OPEN3D = __import__("open3d")
    return _OPEN3D


def adjust_colors(rgb: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
    if brightness == 1.0 and saturation == 1.0 or len(rgb) == 0:
        return rgb
    u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8).reshape(1, -1, 3)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3).astype(np.float32) / 255.0


class Open3DPointRenderer:
    def __init__(self, settings: RenderSettings):
        self.settings = settings
        self.width = settings.width
        self.height = settings.height
        o3d = _open3d()
        with _quiet_c_stdio():
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        cg = o3d.visualization.rendering.ColorGrading(
            o3d.visualization.rendering.ColorGrading.Quality.ULTRA,
            o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.renderer.scene.view.set_color_grading(cg)
        bg = parse_color(settings.background) or [0.0, 0.0, 0.0]
        self.renderer.scene.set_background(np.asarray([bg[0], bg[1], bg[2], 1.0], dtype=np.float32))

        self.point_mat = o3d.visualization.rendering.MaterialRecord()
        self.point_mat.shader = "defaultUnlit"
        self.point_mat.point_size = settings.point_size
        self.point_mat.sRGB_color = True

        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
        self.point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
        self.renderer.scene.add_geometry("main_points", self.point_cloud, self.point_mat)
        self.active_extra = set()
        self.materials = {}

    def render(self, xyz: np.ndarray, rgb: np.ndarray, camera: ViewCamera, extras: list[dict]) -> np.ndarray:
        o3d = _open3d()
        if len(xyz) == 0:
            xyz = np.zeros((1, 3), dtype=np.float32)
            rgb = np.zeros((1, 3), dtype=np.float32)
        rgb = adjust_colors(rgb, self.settings.brightness, self.settings.saturation)
        self.point_cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        self.point_cloud.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))
        if self.renderer.scene.has_geometry("main_points"):
            self.renderer.scene.remove_geometry("main_points")
        self.renderer.scene.add_geometry("main_points", self.point_cloud, self.point_mat)
        self.renderer.setup_camera(
            camera.fov_deg,
            camera.center.astype(np.float64),
            camera.eye.astype(np.float64),
            camera.up.astype(np.float64),
        )
        self._set_extras(extras)
        return np.asarray(self.renderer.render_to_image())

    def _set_extras(self, extras: list[dict]) -> None:
        o3d = _open3d()
        current = set()
        for geom in extras:
            name = geom["name"]
            current.add(name)
            if geom["type"] == "lines":
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(geom["points"])
                line_set.lines = o3d.utility.Vector2iVector(geom["segments"])
                line_set.colors = o3d.utility.Vector3dVector(geom["colors"])
                mat = self._material(name, "unlitLine", line_width=geom.get("line_width", 2.0))
                if self.renderer.scene.has_geometry(name):
                    self.renderer.scene.remove_geometry(name)
                self.renderer.scene.add_geometry(name, line_set, mat)
            elif geom["type"] == "points":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(geom["points"])
                pcd.colors = o3d.utility.Vector3dVector(geom["colors"])
                mat = self._material(name, "defaultUnlit", point_size=geom.get("point_size", 8.0))
                if self.renderer.scene.has_geometry(name):
                    self.renderer.scene.remove_geometry(name)
                self.renderer.scene.add_geometry(name, pcd, mat)
        for stale in self.active_extra - current:
            if self.renderer.scene.has_geometry(stale):
                self.renderer.scene.remove_geometry(stale)
        self.active_extra = current

    def _material(self, name: str, shader: str, **kwargs):
        if name not in self.materials:
            o3d = _open3d()
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = shader
            for key, value in kwargs.items():
                setattr(mat, key, value)
            self.materials[name] = mat
        return self.materials[name]

    def close(self) -> None:
        return None
