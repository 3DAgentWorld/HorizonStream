import os
import numpy as np


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_w2c_txt(path, extri, frames):
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("# w2c\n")
        for i, frame in enumerate(frames):
            mat = extri[i]
            r = mat[:3, :3].reshape(-1)
            t = mat[:3, 3].reshape(-1)
            vals = [frame] + r.tolist() + t.tolist()
            f.write(" ".join([str(v) for v in vals]) + "\n")


def save_intri_txt(path, intri, frames):
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("# fx fy cx cy\n")
        for i, frame in enumerate(frames):
            k = intri[i]
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            f.write(f"{frame} {fx} {fy} {cx} {cy}\n")


def save_rel_pose_txt(path, rel_pose_enc, frames):
    _ensure_dir(path)
    arr = rel_pose_enc
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    with open(path, "w") as f:
        f.write("# tx ty tz qx qy qz qw fov_h fov_w\n")
        for i, frame in enumerate(frames):
            vals = [frame] + arr[i].tolist()
            f.write(" ".join([str(v) for v in vals]) + "\n")
