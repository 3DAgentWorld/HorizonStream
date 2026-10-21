"""Shared in-memory scene types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameSequence:
    images: np.ndarray
    depths: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    confidence: Optional[np.ndarray] = None
    frame_ids: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    def subset(self, indices: np.ndarray) -> "FrameSequence":
        conf = self.confidence[indices] if self.confidence is not None else None
        frame_ids = self.frame_ids[indices] if self.frame_ids is not None else None
        return FrameSequence(
            images=self.images[indices],
            depths=self.depths[indices],
            c2w=self.c2w[indices],
            intrinsics=self.intrinsics[indices],
            confidence=conf,
            frame_ids=frame_ids,
        )


@dataclass
class RenderScene:
    sequence: FrameSequence
    point_xyz: np.ndarray
    point_rgb: np.ndarray
    point_frames: np.ndarray
    lod_tree: object
    scene_scale: float

    @property
    def num_frames(self) -> int:
        return self.sequence.num_frames

    @property
    def c2w(self) -> np.ndarray:
        return self.sequence.c2w

    @property
    def intrinsics(self) -> np.ndarray:
        return self.sequence.intrinsics
