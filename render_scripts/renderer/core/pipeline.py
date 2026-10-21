"""End-to-end scene rendering pipeline."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .configuration import RendererSettings
from .scene_graph import RenderScene
from ..geometry.depth_points import choose_device, unproject_depth_batch
from ..geometry.point_filters import (
    filter_radius_outliers,
    filter_sky_colored_points,
    filter_statistical_outliers,
)
from ..geometry.sky_masks import make_sky_masks
from ..geometry.sparse_lod import SparseLodTree
from ..io.data_sources import load_scene_source
from ..io.glb_export import export_point_cloud_glb
from ..io.media_io import encode_bgr_frames, encode_png_sequence, write_rgb_png
from ..visualization.camera_timeline import build_camera_timeline, scene_scale
from ..visualization.mini_map import MiniMapRenderer
from ..visualization.open3d_engine import Open3DPointRenderer
from ..visualization.render_overlays import CameraPathOverlay


class SceneRenderPipeline:
    def __init__(self, settings: RendererSettings, log: Callable[[str], None] | None = None):
        self.settings = settings
        self.log = log or print
        self.device = choose_device()

    def run(self) -> RenderScene:
        scene = self.build_scene()
        self.render_scene(scene)
        return scene

    def build_scene(self) -> RenderScene:
        cfg = self.settings
        self.log(f"[load] {cfg.io.input_path}")
        sequence = load_scene_source(
            cfg.io.input_path,
            skip_first=cfg.io.skip_first_frames,
            max_items=cfg.io.max_frames,
            step=cfg.io.frame_step,
        )
        self.log(f"[scene] {sequence.num_frames} frames, device={self.device.type}")

        sky_masks = None
        if cfg.filters.sky_mask_enabled:
            sky_masks = make_sky_masks(
                sequence.images,
                cfg.filters.sky_mask_model,
                cfg.filters.sky_mask_batch_size,
                cfg.filters.sky_mask_preview_dir,
                log=self.log,
            )

        xyz, rgb, frames = self._collect_points(sequence, sky_masks)
        xyz, rgb, frames = filter_sky_colored_points(xyz, rgb, frames, cfg.filters, self.log)
        xyz, rgb, frames = filter_statistical_outliers(xyz, rgb, frames, cfg.filters, self.log)
        xyz, rgb, frames = filter_radius_outliers(xyz, rgb, frames, cfg.filters, self.log)

        lod = SparseLodTree(max_level=cfg.geometry.lod_levels, device=self.device)
        lod.build(xyz, rgb, frames, log=self.log)
        scale = scene_scale(sequence.c2w)
        self.log(f"[scene] {len(lod.sorted_xyz):,} finest LOD points, scale={scale:.3f}")
        return RenderScene(sequence, lod.sorted_xyz, lod.sorted_rgb, lod.sorted_frames, lod, scale)

    def _collect_points(self, sequence, sky_masks):
        cfg = self.settings
        batch = max(1, int(cfg.runtime.batch_size))
        xyz_parts, rgb_parts, frame_parts = [], [], []
        H, W = sequence.depths[0].shape
        for start in tqdm(range(0, sequence.num_frames, batch), desc="Unproject", unit="batch"):
            end = min(start + batch, sequence.num_frames)
            depths = sequence.depths[start:end].astype(np.float32).copy()
            depths[(depths <= cfg.geometry.min_depth) | (depths >= cfg.geometry.max_depth)] = 0.0
            if sky_masks is not None:
                depths[sky_masks[start:end]] = 0.0
            if sequence.confidence is not None:
                conf = sequence.confidence[start:end]
                if cfg.filters.visibility_threshold > 0:
                    depths[conf < cfg.filters.visibility_threshold] = 0.0
                if cfg.filters.confidence_percentile > 0:
                    for j in range(end - start):
                        valid = depths[j] > 0
                        if valid.any():
                            thr = np.quantile(conf[j][valid], cfg.filters.confidence_percentile / 100.0)
                            depths[j][valid & (conf[j] < thr)] = 0.0
            d_t = torch.from_numpy(depths).to(self.device)
            img_t = torch.from_numpy(sequence.images[start:end].copy()).to(self.device)
            K_t = torch.from_numpy(sequence.intrinsics[start:end].astype(np.float32)).to(self.device)
            c2w_t = torch.from_numpy(sequence.c2w[start:end].astype(np.float32)).to(self.device)
            pts, cols, counts = unproject_depth_batch(
                d_t,
                img_t,
                K_t,
                c2w_t,
                cfg.geometry.min_depth,
                cfg.geometry.max_depth,
                cfg.geometry.pixel_step,
                cfg.geometry.jitter_pixels,
            )
            offset = 0
            for j, count_t in enumerate(counts):
                count = int(count_t.item())
                if count > 0:
                    xyz_parts.append(pts[offset:offset + count].detach().cpu())
                    rgb_parts.append(cols[offset:offset + count].detach().cpu())
                    frame_parts.append(torch.full((count,), start + j, dtype=torch.int32))
                offset += count
            del d_t, img_t, K_t, c2w_t, pts, cols
        if not xyz_parts:
            return (
                torch.empty((0, 3), dtype=torch.float32),
                torch.empty((0, 3), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int32),
            )
        return torch.cat(xyz_parts), torch.cat(rgb_parts), torch.cat(frame_parts)

    def render_scene(self, scene: RenderScene) -> None:
        cfg = self.settings
        output = Path(cfg.io.output_path)
        frame_dir = output.with_suffix("").with_name(output.stem + "_frames")
        shutil.rmtree(frame_dir, ignore_errors=True)
        frame_dir.mkdir(parents=True, exist_ok=True)
        cameras = build_camera_timeline(scene, cfg.camera, cfg.render.width, cfg.render.height)
        overlay = CameraPathOverlay(scene.c2w, scene.intrinsics, cfg.overlays)
        renderer = Open3DPointRenderer(cfg.render)
        t0 = time.time()
        for out_i in tqdm(range(len(cameras)), desc="Render", unit="frame"):
            cam = cameras[out_i]
            reveal = min(out_i + cameras.frame_offset, scene.num_frames - 1)
            vis_xyz, vis_rgb = scene.lod_tree.select_visible(
                cam.world_to_camera(),
                *cam.fov_intrinsics(cfg.render.width, cfg.render.height),
                cfg.render.width,
                cfg.render.height,
                cfg.render.near_plane,
                cfg.render.far_plane,
                reveal,
                cfg.render.lod_target_pixels,
                frame_start=cameras.frame_offset,
            )
            extras = overlay.geometries(reveal, cameras.frame_offset)
            frame = renderer.render(vis_xyz, vis_rgb, cam, extras)
            write_rgb_png(frame_dir / f"frame_{out_i:06d}.png", frame)
        self.log(f"[render] {len(cameras)} frames in {time.time() - t0:.1f}s")
        encode_png_sequence(frame_dir, output, cfg.io.fps)
        self.log(f"[video] {output}")
        if not cfg.io.keep_frame_images:
            shutil.rmtree(frame_dir, ignore_errors=True)
        if cfg.export.glb_enabled:
            glb_path = output.with_suffix("").with_name(output.stem + "_cloud.glb")
            export_point_cloud_glb(scene.point_xyz, scene.point_rgb, scene.c2w, glb_path, cfg.export.glb_sample_rate)
            self.log(f"[glb] {glb_path}")
        if cfg.minimap.enabled:
            mm = MiniMapRenderer(scene.c2w, cfg.minimap)
            frames = [mm.render(i) for i in range(len(cameras))]
            mm_path = output.with_suffix("").with_name(output.stem + "_minimap.mp4")
            encode_bgr_frames(frames, mm_path, cfg.io.fps)
            self.log(f"[minimap] {mm_path}")
