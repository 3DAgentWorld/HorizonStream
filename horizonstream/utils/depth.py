import numpy as np
import torch
import matplotlib.cm as cm


def colorize_depth(depth: torch.Tensor, cmap: str = "plasma") -> np.ndarray:
    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = depth
    d_min = np.nanmin(depth_np)
    d_max = np.nanmax(depth_np)
    if d_max - d_min < 1e-6:
        d_max = d_min + 1e-6
    norm = (depth_np - d_min) / (d_max - d_min)
    norm = np.clip(norm, 0.0, 1.0)
    mapper = cm.get_cmap(cmap)
    colored = mapper(norm)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def unproject_depth_to_points(depth: torch.Tensor, intri: torch.Tensor) -> torch.Tensor:
    B, H, W = depth.shape
    fx = intri[:, 0, 0].view(B, 1, 1)
    fy = intri[:, 1, 1].view(B, 1, 1)
    cx = intri[:, 0, 2].view(B, 1, 1)
    cy = intri[:, 1, 2].view(B, 1, 1)

    ys = torch.arange(H, device=depth.device).view(1, H, 1).float()
    xs = torch.arange(W, device=depth.device).view(1, 1, W).float()

    x = (xs - cx) * depth / fx
    y = (ys - cy) * depth / fy
    z = depth
    pts = torch.stack([x, y, z], dim=-1)
    return pts
