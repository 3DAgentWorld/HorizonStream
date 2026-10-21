"""Typed configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _section(cls, values: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (values or {}).items() if k in names})


@dataclass
class IOSettings:
    input_path: str = ""
    output_path: str = "render.mp4"
    fps: int = 30
    keep_frame_images: bool = False
    skip_first_frames: int = 0
    max_frames: int = 0
    frame_step: int = 1


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    background: str = "white"
    point_size: float = 2.0
    near_plane: float = 0.1
    far_plane: float = 200.0
    lod_target_pixels: float = 1.5
    brightness: float = 1.4
    saturation: float = 1.2


@dataclass
class CameraSegmentSettings:
    mode: str = "follow"
    frames: list[int] = field(default_factory=lambda: [0, -1])
    duration_frames: int = 0
    back_distance: float = 0.25
    height_offset: float = 0.05
    look_ahead: float = 0.5
    smoothing_window: int = 20
    scale_reference_frames: int = 100
    overview_height_scale: float = 1.6


@dataclass
class CameraSettings:
    fov_degrees: float = 60.0
    transition_frames: int = 40
    segments: list[CameraSegmentSettings] = field(default_factory=lambda: [
        CameraSegmentSettings(mode="follow", frames=[0, -1]),
        CameraSegmentSettings(mode="overview", frames=[-1, -1], duration_frames=90),
    ])


@dataclass
class GeometrySettings:
    lod_levels: int = 16
    min_depth: float = 0.5
    max_depth: float = 80.0
    pixel_step: int = 2
    jitter_pixels: bool = True


@dataclass
class FilterSettings:
    sky_mask_enabled: bool = True
    sky_mask_model: str = "checkpoints/skyseg.onnx"
    sky_mask_batch_size: int = 64
    sky_mask_preview_dir: str = "./sky_mask_viz"
    confidence_percentile: float = 0.0
    visibility_threshold: float = 1.3
    color_filter_enabled: bool = True
    white_threshold: int = 240
    bright_channel_threshold: int = 200
    blue_channel_min: int = 155
    red_channel_max: int = 195
    green_channel_min: int = 100
    red_green_delta_max: int = 95
    blue_green_delta_min: int = 8
    blue_red_delta_min: int = 15
    statistical_filter_enabled: bool = False
    statistical_neighbors: int = 16
    statistical_std_ratio: float = 2.5
    radius_filter_enabled: bool = True
    radius_filter_radius: float = 0.20
    radius_filter_min_neighbors: int = 15


@dataclass
class OverlaySettings:
    camera_path_style: str = "frustum"
    trail_enabled: bool = True
    trail_length: int = 80
    trail_width: float = 2.0
    trail_color: str = "rainbow"
    frustum_count: int = 20000
    frustum_scale: float = 0.05
    frustum_line_width: float = 1.5
    frame_label_enabled: bool = False
    frame_label_position: str = "top_left"


@dataclass
class MiniMapSettings:
    enabled: bool = True
    width: int = 512
    height: int = 512
    background: str = "white"
    trail_width: int = 4
    marker_radius: int = 10
    margin: float = 0.20
    rotation_degrees: int = 90
    trail_saturation: float = 0.95
    trail_brightness: float = 0.85


@dataclass
class ExportSettings:
    glb_enabled: bool = True
    glb_sample_rate: int = 10


@dataclass
class RuntimeSettings:
    batch_size: int = 4
    memory_limit_gb: float = 30.0


@dataclass
class RendererSettings:
    io: IOSettings = field(default_factory=IOSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    overlays: OverlaySettings = field(default_factory=OverlaySettings)
    minimap: MiniMapSettings = field(default_factory=MiniMapSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RendererSettings":
        cfg = cls()
        if "io" in raw:
            cfg.io = _section(IOSettings, raw["io"])
        if "render" in raw:
            cfg.render = _section(RenderSettings, raw["render"])
        if "camera" in raw:
            cam = raw["camera"] or {}
            cfg.camera.fov_degrees = float(cam.get("fov_degrees", cfg.camera.fov_degrees))
            cfg.camera.transition_frames = int(cam.get("transition_frames", cfg.camera.transition_frames))
            if "segments" in cam:
                cfg.camera.segments = [
                    _section(CameraSegmentSettings, segment)
                    for segment in cam.get("segments") or []
                ]
        if "geometry" in raw:
            cfg.geometry = _section(GeometrySettings, raw["geometry"])
        if "filters" in raw:
            cfg.filters = _section(FilterSettings, raw["filters"])
        if "overlays" in raw:
            cfg.overlays = _section(OverlaySettings, raw["overlays"])
        if "minimap" in raw:
            cfg.minimap = _section(MiniMapSettings, raw["minimap"])
        if "export" in raw:
            cfg.export = _section(ExportSettings, raw["export"])
        if "runtime" in raw:
            cfg.runtime = _section(RuntimeSettings, raw["runtime"])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path | None) -> RendererSettings:
    if not path:
        return RendererSettings()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RendererSettings.from_dict(raw)
