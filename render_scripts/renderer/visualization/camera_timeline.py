"""Camera timeline generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.configuration import CameraSettings
from ..core.scene_graph import RenderScene


@dataclass
class ViewCamera:
    eye: np.ndarray
    center: np.ndarray
    up: np.ndarray
    fov_deg: float

    def world_to_camera(self) -> np.ndarray:
        return look_at(self.eye, self.center, self.up)

    def fov_intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        fov = np.radians(self.fov_deg)
        fy = height / (2.0 * np.tan(fov / 2.0))
        return float(fy), float(fy), width * 0.5, height * 0.5


@dataclass
class CameraTimeline:
    cameras: list[ViewCamera]
    frame_offset: int = 0

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, index: int) -> ViewCamera:
        return self.cameras[index]


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else fallback.astype(np.float32)


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 3 * t * t - 2 * t * t * t


def _lerp_camera(a: ViewCamera, b: ViewCamera, t: float) -> ViewCamera:
    t = _smoothstep(t)
    up = _unit((1 - t) * a.up + t * b.up, a.up)
    return ViewCamera(
        eye=((1 - t) * a.eye + t * b.eye).astype(np.float32),
        center=((1 - t) * a.center + t * b.center).astype(np.float32),
        up=up,
        fov_deg=(1 - t) * a.fov_deg + t * b.fov_deg,
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _unit(target - eye, np.array([0, 0, 1], dtype=np.float32))
    right = _unit(np.cross(up, forward), np.array([1, 0, 0], dtype=np.float32))
    down = np.cross(right, forward).astype(np.float32)
    R = np.stack([right, down, forward], axis=0).astype(np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R
    out[:3, 3] = -R @ eye.astype(np.float32)
    return out


def global_up(c2w: np.ndarray) -> np.ndarray:
    up = np.median(-c2w[:, :3, 1], axis=0)
    return _unit(up, np.array([0, 1, 0], dtype=np.float32))


def scene_scale(c2w: np.ndarray) -> float:
    pos = c2w[:, :3, 3]
    return max(float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))), 1e-3)


def local_scale(c2w: np.ndarray, n_frames: int) -> float:
    n = min(max(n_frames, 1), len(c2w))
    pos = c2w[:n, :3, 3]
    return max(float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))), 1e-3)


def _follow_camera_at(
    frame_idx: int,
    c2w: np.ndarray,
    full_scale: float,
    smooth_window: int,
    back_offset: float,
    up_offset: float,
    look_offset: float,
    follow_scale: Optional[float],
    up_axis: np.ndarray,
    last_forward_h: Optional[np.ndarray],
    fov: float,
) -> tuple[ViewCamera, np.ndarray]:
    S = len(c2w)
    half = max(smooth_window // 2, 1)
    start = max(0, frame_idx - half)
    end = min(S, frame_idx + half + 1)
    cam_pos = c2w[start:end, :3, 3].mean(axis=0)
    forward = _unit(c2w[start:end, :3, 2].mean(axis=0), np.array([0, 0, 1], dtype=np.float32))
    forward_h = forward - float(np.dot(forward, up_axis)) * up_axis
    if np.linalg.norm(forward_h) < 0.1:
        p0 = c2w[max(0, frame_idx - half), :3, 3]
        p1 = c2w[min(S - 1, frame_idx + half), :3, 3]
        vel = p1 - p0
        forward_h = vel - float(np.dot(vel, up_axis)) * up_axis
    forward_h = _unit(
        forward_h,
        last_forward_h if last_forward_h is not None else np.array([1, 0, 0], dtype=np.float32),
    )
    scale = follow_scale if follow_scale is not None else full_scale
    eye = cam_pos - forward_h * scale * back_offset + up_axis * scale * up_offset
    center = cam_pos + forward * scale * look_offset
    return ViewCamera(eye.astype(np.float32), center.astype(np.float32), up_axis.copy(), fov), forward_h


def _make_follow(scene: RenderScene, cfg, fov: float) -> list[ViewCamera]:
    c2w = scene.c2w
    up_axis = global_up(c2w)
    follow_scale = local_scale(c2w, cfg.scale_reference_frames) if 0 < cfg.scale_reference_frames < len(c2w) else None
    cameras = []
    last_forward = None
    for fi in range(len(c2w)):
        cam, last_forward = _follow_camera_at(
            fi,
            c2w,
            scene.scene_scale,
            cfg.smoothing_window,
            cfg.back_distance,
            cfg.height_offset,
            cfg.look_ahead,
            follow_scale,
            up_axis,
            last_forward,
            fov,
        )
        cameras.append(cam)
    return cameras


def _make_overview(scene: RenderScene, render_w: int, render_h: int, fov: float, height_mult: float) -> ViewCamera:
    c2w = scene.c2w
    up_axis = global_up(c2w)
    positions = c2w[:, :3, 3]
    traj = positions[-1] - positions[0]
    traj_h = traj - float(np.dot(traj, up_axis)) * up_axis
    traj_dir = _unit(traj_h, np.array([1, 0, 0], dtype=np.float32))
    end_up = np.cross(up_axis, traj_dir)
    end_up = _unit(end_up, np.array([1, 0, 0], dtype=np.float32))

    center = positions.mean(axis=0).astype(np.float32)
    height = scene.scene_scale * height_mult
    points = scene.lod_tree.sorted_xyz
    if len(points) > 0:
        cam_forward = -up_axis
        cam_right = _unit(np.cross(cam_forward, end_up), np.array([1, 0, 0], dtype=np.float32))
        rel = points - center
        pr = rel @ cam_right
        pu = rel @ end_up
        r_min, r_max = np.percentile(pr, [1, 99])
        u_min, u_max = np.percentile(pu, [1, 99])
        center = (center + ((r_min + r_max) * 0.5) * cam_right + ((u_min + u_max) * 0.5) * end_up).astype(np.float32)
        half_w = (r_max - r_min) * 0.5
        half_h = (u_max - u_min) * 0.5
        fov_rad = np.radians(fov)
        aspect = render_w / max(render_h, 1)
        need_v = half_h / max(np.tan(fov_rad / 2), 1e-6)
        need_h = half_w / max(aspect * np.tan(fov_rad / 2), 1e-6)
        height = max(float(need_v), float(need_h), height) * 1.15
    eye = center + up_axis * height - end_up * scene.scene_scale * 0.1
    return ViewCamera(eye.astype(np.float32), center.astype(np.float32), end_up, fov)


def build_camera_timeline(scene: RenderScene, settings: CameraSettings, render_w: int, render_h: int) -> CameraTimeline:
    follow_seg = next((s for s in settings.segments if s.mode == "follow"), settings.segments[0])
    follow = _make_follow(scene, follow_seg, settings.fov_degrees)
    cameras = list(follow)

    overview = next((s for s in settings.segments if s.mode == "overview" and s.duration_frames > 0), None)
    if overview is not None:
        bird = _make_overview(scene, render_w, render_h, settings.fov_degrees, overview.overview_height_scale)
        half = max(settings.transition_frames // 2, 1)
        last_follow = follow[-1]
        for i in range(max(0, len(follow) - half), len(follow)):
            t = (i - (len(follow) - half)) / max(2 * half, 1)
            cameras[i] = _lerp_camera(follow[i], bird, t)
        for j in range(overview.duration_frames):
            if j < half:
                t = (half + j) / max(2 * half, 1)
                cameras.append(_lerp_camera(last_follow, bird, t))
            else:
                cameras.append(bird)
    return CameraTimeline(cameras, frame_offset=0)
