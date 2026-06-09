from typing import Optional

import torch


def create_attn_mask(
    S: int,
    P: int = 1,
    mode: str = "full",
    device: torch.device | None = None,
    window_size: int = 5,
) -> Optional[torch.Tensor]:
    if mode == "full":
        return None

    N = S * P
    mask = torch.zeros((N, N), dtype=torch.float32, device=device)

    if mode == "causal":
        for i in range(S):
            curr_view_start = i * P
            curr_view_end = (i + 1) * P
            mask[curr_view_start:curr_view_end, curr_view_end:] = float("-inf")
    elif mode == "window":
        for i in range(S):
            curr_view_start = i * P
            curr_view_end = (i + 1) * P
            start_view = max(0, i - window_size + 1)
            mask[curr_view_start:curr_view_end, : start_view * P] = float("-inf")
            mask[curr_view_start:curr_view_end, curr_view_end:] = float("-inf")
    else:
        raise NotImplementedError(f"Unknown attention mode: {mode}")

    return mask

