"""Dependency-free GLB point-cloud export."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _pack_glb(gltf: dict, binary: bytes) -> bytes:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A)
    out += json_bytes
    out += struct.pack("<II", len(binary), 0x004E4942)
    out += binary
    return bytes(out)


def export_point_cloud_glb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    c2w: np.ndarray,
    output_path: str | Path,
    downsample: int = 10,
) -> None:
    rng = np.random.default_rng(0)
    if downsample > 1 and len(xyz) > downsample:
        keep = rng.choice(len(xyz), max(1, len(xyz) // downsample), replace=False)
        keep.sort()
        xyz = xyz[keep]
        rgb = rgb[keep]
    if len(xyz) == 0:
        xyz = np.zeros((1, 3), dtype=np.float32)
        rgb = np.zeros((1, 3), dtype=np.float32)
    xyz = xyz.astype(np.float32)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    cam = c2w[:, :3, 3].astype(np.float32)
    data_pos = xyz.tobytes()
    data_col = rgb.tobytes()
    data_cam = cam.tobytes()
    off_pos = 0
    off_col = len(data_pos)
    off_cam = off_col + len(data_col)
    binary = data_pos + data_col + data_cam

    gltf = {
        "asset": {"version": "2.0", "generator": "render_scripts/renderer"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "PointCloud", "mesh": 0}, {"name": "CameraTrajectory", "mesh": 1}],
        "meshes": [
            {"name": "PointCloud", "primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]},
            {"name": "CameraTrajectory", "primitives": [{"attributes": {"POSITION": 2}, "mode": 3}]},
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": len(xyz), "type": "VEC3",
                "min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5126, "count": len(rgb), "type": "VEC3"},
            {
                "bufferView": 2, "componentType": 5126, "count": len(cam), "type": "VEC3",
                "min": cam.min(axis=0).tolist(), "max": cam.max(axis=0).tolist(),
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_pos, "byteLength": len(data_pos), "target": 34962},
            {"buffer": 0, "byteOffset": off_col, "byteLength": len(data_col), "target": 34962},
            {"buffer": 0, "byteOffset": off_cam, "byteLength": len(data_cam), "target": 34962},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    Path(output_path).write_bytes(_pack_glb(gltf, binary))
