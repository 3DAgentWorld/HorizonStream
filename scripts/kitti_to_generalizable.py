import argparse
import os
import shutil
from glob import glob

import numpy as np


def _read_list(path: str):
    with open(path, "r") as f:
        return [
            ln.strip()
            for ln in f.readlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]


def _write_list(path: str, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str, copy: bool):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _link_tree(src_dir: str, dst_dir: str, copy: bool):
    _ensure_dir(dst_dir)
    files = sorted(glob(os.path.join(src_dir, "*")))
    for p in files:
        if os.path.isdir(p):
            continue
        _link_or_copy(p, os.path.join(dst_dir, os.path.basename(p)), copy=copy)


def _opencv_matrix(name: str, mat: np.ndarray) -> str:
    mat = np.asarray(mat, dtype=np.float64)
    rows, cols = mat.shape
    data = ", ".join(f"{float(v):.12g}" for v in mat.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        f"   rows: {rows}\n"
        f"   cols: {cols}\n"
        f"   dt: d\n"
        f"   data: [ {data} ]\n"
    )


def _load_kitti_calib_k(seq_dir: str, camera: str) -> np.ndarray:
    calib_path = os.path.join(seq_dir, "calib.txt")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(calib_path)

    key = f"P{int(camera):d}:"
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(key):
                continue
            values = [float(v) for v in line.split()[1:]]
            return np.asarray(values, dtype=np.float64).reshape(3, 4)[:, :3]
    raise KeyError(f"{key} not found in {calib_path}")


def _load_kitti_poses(src_root: str, seq: str):
    candidates = [
        os.path.join(src_root, "poses", f"{seq}.txt"),
        os.path.join(os.path.dirname(src_root), "poses", f"{seq}.txt"),
    ]
    pose_path = next((p for p in candidates if os.path.exists(p)), None)
    if pose_path is None:
        return None

    c2w_list = []
    with open(pose_path, "r") as f:
        for line in f:
            values = [float(v) for v in line.split()]
            if len(values) != 12:
                continue
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
            c2w_list.append(c2w)
    return c2w_list


def _write_camera_yml(dst_scene: str, camera: str, image_files, k_mat: np.ndarray, c2w_list):
    cam_dir = os.path.join(dst_scene, "cameras", camera)
    _ensure_dir(cam_dir)
    names = [
        os.path.splitext(os.path.basename(img_path))[0]
        for img_path in image_files[: len(c2w_list)]
    ]

    with open(os.path.join(cam_dir, "extri.yml"), "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for name in names:
            f.write(f'   - "{name}"\n')
        for idx, name in enumerate(names):
            w2c = np.linalg.inv(c2w_list[idx])
            f.write(_opencv_matrix(f"Rot_{name}", w2c[:3, :3]))
            f.write(_opencv_matrix(f"T_{name}", w2c[:3, 3:4]))

    with open(os.path.join(cam_dir, "intri.yml"), "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for name in names:
            f.write(f'   - "{name}"\n')
        for name in names:
            f.write(_opencv_matrix(f"K_{name}", k_mat))


def _is_generalizable_meta_root(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "00", "images"))


def _is_kitti_sequences_root(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    if os.path.isdir(os.path.join(path, "sequences")):
        return True
    return os.path.isdir(os.path.join(path, "00")) and (
        os.path.isdir(os.path.join(path, "00", "image_2"))
        or os.path.isdir(os.path.join(path, "00", "image_02"))
    )


def _resolve_kitti_seq_dir(sequences_root: str, seq: str) -> str:
    if os.path.isdir(os.path.join(sequences_root, "sequences")):
        return os.path.join(sequences_root, "sequences", seq)
    return os.path.join(sequences_root, seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        help="KITTI source: generalizable meta_root OR KITTI odometry sequences root.",
    )
    parser.add_argument(
        "--out", required=True, help="Output meta_root in GeneralizableDataset format."
    )
    parser.add_argument(
        "--seqs",
        default=None,
        help="Comma-separated seq ids (e.g. 00,01,02). Default: use data_roots.txt or scan dirs.",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlinking."
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    copy = bool(args.copy)

    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    _ensure_dir(out)

    if _is_generalizable_meta_root(src):
        roots_file = os.path.join(src, "data_roots.txt")
        seqs = (
            _read_list(roots_file)
            if os.path.exists(roots_file)
            else sorted(
                [
                    d
                    for d in os.listdir(src)
                    if os.path.isdir(os.path.join(src, d, "images"))
                ]
            )
        )
        if args.seqs:
            seqs = [s for s in args.seqs.split(",") if s]
        for seq in seqs:
            src_scene = os.path.join(src, seq)
            dst_scene = os.path.join(out, seq)
            _ensure_dir(dst_scene)
            for sub in ["images", "cameras", "depths", "masks", "vis_depths"]:
                sp = os.path.join(src_scene, sub)
                if not os.path.exists(sp):
                    continue
                dp = os.path.join(dst_scene, sub)
                if os.path.isdir(sp):
                    if copy:
                        shutil.copytree(sp, dp, dirs_exist_ok=True)
                    else:
                        if not os.path.exists(dp):
                            os.symlink(sp, dp)
                else:
                    _link_or_copy(sp, dp, copy=copy)
        _write_list(os.path.join(out, "data_roots.txt"), seqs)
        return

    if not _is_kitti_sequences_root(src):
        raise RuntimeError("Unsupported KITTI source layout.")

    if args.seqs:
        seqs = [s for s in args.seqs.split(",") if s]
    else:
        sequences_root = (
            os.path.join(src, "sequences")
            if os.path.isdir(os.path.join(src, "sequences"))
            else src
        )
        seqs = sorted(
            [
                d
                for d in os.listdir(sequences_root)
                if os.path.isdir(os.path.join(sequences_root, d))
            ]
        )

    for seq in seqs:
        seq_dir = _resolve_kitti_seq_dir(src, seq)
        if not os.path.isdir(seq_dir):
            continue

        img2 = os.path.join(seq_dir, "image_2")
        if not os.path.isdir(img2):
            img2 = os.path.join(seq_dir, "image_02")
        img3 = os.path.join(seq_dir, "image_3")
        if not os.path.isdir(img3):
            img3 = os.path.join(seq_dir, "image_03")

        dst_scene = os.path.join(out, seq)
        dst_img2 = os.path.join(dst_scene, "images", "02")
        dst_img3 = os.path.join(dst_scene, "images", "03")
        image_files = []
        if os.path.isdir(img2):
            _link_tree(img2, dst_img2, copy=copy)
            image_files = sorted(glob(os.path.join(img2, "*")))
        if os.path.isdir(img3):
            _link_tree(img3, dst_img3, copy=copy)
        c2w_list = _load_kitti_poses(src, seq)
        if image_files and c2w_list:
            k_mat = _load_kitti_calib_k(seq_dir, "02")
            _write_camera_yml(dst_scene, "02", image_files, k_mat, c2w_list)

    _write_list(os.path.join(out, "data_roots.txt"), seqs)


if __name__ == "__main__":
    main()
