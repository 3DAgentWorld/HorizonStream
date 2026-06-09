import random
import torch
from typing import Optional, Tuple


class KeyframeSelector:
    def __init__(
        self,
        min_interval: int = 8,
        max_interval: int = 8,
        force_first: bool = True,
        motion_threshold: Optional[float] = None,
        mode: str = "fixed",
    ):
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.force_first = bool(force_first)
        self.motion_threshold = motion_threshold
        self.mode = mode

    def select_keyframes(
        self,
        sequence_length: int,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or torch.device("cpu")
        is_keyframe = torch.zeros(
            batch_size, sequence_length, dtype=torch.bool, device=device
        )
        keyframe_indices = torch.zeros(
            batch_size, sequence_length, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            last_keyframe_idx = 0
            next_keyframe_target = None

            if self.force_first or sequence_length == 1:
                is_keyframe[b, 0] = True
                keyframe_indices[b, 0] = 0
                if self.mode == "random":
                    interval = random.randint(self.min_interval, self.max_interval)
                    next_keyframe_target = interval

            for s in range(1, sequence_length):
                keyframe_indices[b, s] = last_keyframe_idx
                frames_since_last = s - last_keyframe_idx

                if self.mode == "random" and next_keyframe_target is not None:
                    if s >= next_keyframe_target:
                        is_keyframe[b, s] = True
                        last_keyframe_idx = s
                        interval = random.randint(self.min_interval, self.max_interval)
                        next_keyframe_target = s + interval
                elif frames_since_last >= self.max_interval:
                    is_keyframe[b, s] = True
                    last_keyframe_idx = s
                    if self.mode == "random":
                        interval = random.randint(self.min_interval, self.max_interval)
                        next_keyframe_target = s + interval
                elif (
                    frames_since_last >= self.min_interval
                    and poses is not None
                    and self.motion_threshold is not None
                ):
                    motion = torch.norm(
                        poses[b, s, :3] - poses[b, last_keyframe_idx, :3]
                    ).item()
                    if motion > self.motion_threshold:
                        is_keyframe[b, s] = True
                        last_keyframe_idx = s
                        if self.mode == "random":
                            interval = random.randint(
                                self.min_interval, self.max_interval
                            )
                            next_keyframe_target = s + interval

        return is_keyframe, keyframe_indices
