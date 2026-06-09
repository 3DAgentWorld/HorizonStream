#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


PROCESSED_SUFFIX = "_processed_aligned"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_lines(path: str, lines: Iterable[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def _link_or_copy(src: str, dst: str, copy: bool) -> None:
    if os.path.lexists(dst):
        return
    _ensure_dir(os.path.dirname(dst))
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _opencv_matrix(name: str, mat: np.ndarray) -> str:
    mat = np.asarray(mat, dtype=np.float64)
    data = ", ".join(f"{x:.17g}" for x in mat.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        f"   rows: {mat.shape[0]}\n"
        f"   cols: {mat.shape[1]}\n"
        "   dt: d\n"
        f"   data: [ {data} ]\n"
    )


def _write_camera_yml(
    cam_dir: str,
    names: List[str],
    w2c_by_name: Dict[str, np.ndarray],
    k_mat: np.ndarray,
    resolution: Tuple[int, int],
) -> None:
    _ensure_dir(cam_dir)
    width, height = resolution

    with open(os.path.join(cam_dir, "extri.yml"), "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for name in names:
            f.write(f'   - "{name}"\n')
        for name in names:
            w2c = w2c_by_name[name]
            f.write(_opencv_matrix(f"Rot_{name}", w2c[:3, :3]))
            f.write(_opencv_matrix(f"T_{name}", w2c[:3, 3:4]))

    with open(os.path.join(cam_dir, "intri.yml"), "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for name in names:
            f.write(f'   - "{name}"\n')
        for name in names:
            f.write(_opencv_matrix(f"K_{name}", k_mat))
            f.write(f"H_{name}: {height}\n")
            f.write(f"W_{name}: {width}\n")


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_from_tum_row(row: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = row[1:4]
    mat[:3, :3] = _quat_xyzw_to_rot(row[4:8])
    return mat


def _read_tum(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()[:8]]
            if len(vals) == 8:
                rows.append(vals)
    if not rows:
        raise RuntimeError(f"No TUM poses found in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], [_pose_from_tum_row(row) for row in arr]


def _resolve_src_root(src: str) -> str:
    nested = os.path.join(src, "vbr")
    if os.path.isdir(nested):
        return nested
    return src


def _seq_name_from_dir(seq_dir: str) -> str:
    name = os.path.basename(seq_dir.rstrip(os.sep))
    if name.endswith(PROCESSED_SUFFIX):
        return name[: -len(PROCESSED_SUFFIX)]
    return name


def _collect_seq_dirs(src_root: str, seqs_arg: Optional[str]) -> List[str]:
    wanted = None
    if seqs_arg:
        wanted = {s.strip() for s in seqs_arg.split(",") if s.strip()}
    seq_dirs = []
    for name in sorted(os.listdir(src_root)):
        path = os.path.join(src_root, name)
        if not os.path.isdir(path) or not name.endswith(PROCESSED_SUFFIX):
            continue
        seq = _seq_name_from_dir(path)
        if wanted is not None and seq not in wanted and name not in wanted:
            continue
        seq_dirs.append(path)
    return seq_dirs


def _read_camera_pose_matrix(path: str) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise RuntimeError(f"Expected 4x4 camera pose in {path}, got {pose.shape}")
    return pose


def _invert_rigid(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot_t = mat[:3, :3].T
    inv[:3, :3] = rot_t
    inv[:3, 3] = -(rot_t @ mat[:3, 3])
    return inv


def _read_intrinsics(seq_dir: str) -> np.ndarray:
    intrinsics_path = os.path.join(seq_dir, "intrinsics.txt")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Missing processed intrinsics: {intrinsics_path}")
    k_mat = np.loadtxt(intrinsics_path, dtype=np.float64)
    if k_mat.shape != (3, 3):
        raise RuntimeError(f"Expected 3x3 intrinsics in {intrinsics_path}, got {k_mat.shape}")
    return k_mat


def _load_w2c_from_camera_pose_txt(
    seq_dir: str,
    names: List[str],
    require_exact_count: bool,
) -> Dict[str, np.ndarray]:
    pose_txt = os.path.join(seq_dir, "camera_pose.txt")
    if not os.path.exists(pose_txt):
        raise FileNotFoundError(f"Missing required pose file: {pose_txt}")

    _, poses = _read_tum(pose_txt)
    if require_exact_count and len(poses) != len(names):
        raise RuntimeError(
            f"Pose/image count mismatch in {seq_dir}: camera_pose.txt={len(poses)} images={len(names)}"
        )
    if len(poses) < len(names):
        raise RuntimeError(
            f"Not enough poses in {pose_txt}: camera_pose.txt={len(poses)} images={len(names)}"
        )

    # Junyi42/vbr_processed stores camera_pose.txt in sorted image order. The
    # first column is not reliable as the original sparse frame id for all sequences.
    return {name: _invert_rigid(pose) for name, pose in zip(names, poses[: len(names)])}


def _check_rotation(name: str, w2c: np.ndarray) -> None:
    rot = w2c[:3, :3]
    det = float(np.linalg.det(rot))
    ortho_err = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
    if abs(det - 1.0) > 1e-5 or ortho_err > 1e-5:
        raise RuntimeError(f"Invalid rotation for {name}: det={det:.9g}, ortho_err={ortho_err:.9g}")


def _verify_pose_dir_alignment(
    seq_dir: str,
    names: List[str],
    w2c_by_name: Dict[str, np.ndarray],
    max_checks: int,
    atol: float,
) -> None:
    if max_checks <= 0:
        return
    pose_dir = os.path.join(seq_dir, "camera_pose")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"Missing required pose validation dir: {pose_dir}")

    if len(names) <= max_checks:
        indices = list(range(len(names)))
    else:
        indices = sorted(set(np.linspace(0, len(names) - 1, num=max_checks, dtype=np.int64).tolist()))

    for idx in indices:
        name = names[idx]
        pose_path = os.path.join(pose_dir, f"{name}.txt")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing required per-frame pose for validation: {pose_path}")
        per_frame_c2w = _read_camera_pose_matrix(pose_path)
        expected_w2c = _invert_rigid(per_frame_c2w)
        max_err = float(np.max(np.abs(expected_w2c - w2c_by_name[name])))
        if max_err > atol:
            raise RuntimeError(
                f"camera_pose.txt order does not match camera_pose/{name}.txt in {seq_dir}: max_err={max_err:.9g}"
            )


def _validate_sequence_inputs(
    seq_dir: str,
    images: List[str],
    names: List[str],
    require_exact_count: bool,
) -> Dict[str, np.ndarray]:
    if not images:
        raise RuntimeError(f"No images found in {seq_dir}/rgb")

    rgb_dir = os.path.join(seq_dir, "rgb")
    depth_dir = os.path.join(seq_dir, "depthmap")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Missing required rgb dir: {rgb_dir}")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Missing required depthmap dir: {depth_dir}")

    depth_names = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(depth_dir)
        if os.path.splitext(fn)[1].lower() == ".npy"
    }
    missing_depths = [name for name in names if name not in depth_names]
    if missing_depths:
        raise FileNotFoundError(
            f"Missing {len(missing_depths)} depth files in {depth_dir}, first missing: {missing_depths[0]}.npy"
        )
    if require_exact_count and depth_names != set(names):
        extras = sorted(depth_names - set(names))
        raise RuntimeError(
            f"Depth/image name mismatch in {seq_dir}: images={len(names)} depths={len(depth_names)} "
            f"first_extra_depth={extras[0] if extras else 'none'}"
        )

    _read_intrinsics(seq_dir)
    w2c_by_name = _load_w2c_from_camera_pose_txt(seq_dir, names, require_exact_count=require_exact_count)
    for name, w2c in w2c_by_name.items():
        _check_rotation(name, w2c)
    return w2c_by_name


def _write_poses(
    seq_dir: str,
    images: List[str],
    names: List[str],
    dst_scene: str,
    w2c_by_name: Dict[str, np.ndarray],
) -> bool:
    first = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read image: {images[0]}")
    height, width = first.shape[:2]
    _write_camera_yml(
        os.path.join(dst_scene, "cameras", "02"),
        names,
        w2c_by_name,
        _read_intrinsics(seq_dir),
        (width, height),
    )
    return True


def _convert_depth_one(task: Tuple[str, str, bool]) -> int:
    src, dst, skip_existing = task
    if skip_existing and os.path.exists(dst):
        return 0
    _ensure_dir(os.path.dirname(dst))
    depth = np.load(src)
    depth = np.nan_to_num(depth.astype(np.float32, copy=False), 0.0)
    if not cv2.imwrite(dst, depth):
        raise RuntimeError(f"Failed to write {dst}")
    return 1


def _convert_depths(
    seq: str,
    depth_dir: str,
    dst_dir: str,
    names: List[str],
    depth_jobs: int,
    skip_existing: bool,
    progress_every: int,
    src_name_map: Optional[Dict[str, str]] = None,
) -> int:
    tasks = []
    for name in names:
        src_name = src_name_map[name] if src_name_map is not None else name
        src = os.path.join(depth_dir, f"{src_name}.npy")
        if os.path.exists(src):
            tasks.append((src, os.path.join(dst_dir, f"{name}.exr"), skip_existing))

    if depth_jobs <= 1:
        written = 0
        for done, task in enumerate(tasks, start=1):
            written += _convert_depth_one(task)
            if progress_every > 0 and (done % progress_every == 0 or done == len(tasks)):
                print(f"[{seq}] depth {done}/{len(tasks)} written={written}", flush=True)
        return written

    written = 0
    with ThreadPoolExecutor(max_workers=depth_jobs) as executor:
        futures = [executor.submit(_convert_depth_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            written += int(future.result())
            if progress_every > 0 and (done % progress_every == 0 or done == len(tasks)):
                print(f"[{seq}] depth {done}/{len(tasks)} written={written}", flush=True)
    return written


def _image_files(rgb_dir: str) -> List[str]:
    return [
        os.path.join(rgb_dir, fn)
        for fn in sorted(os.listdir(rgb_dir))
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
    ]


def _copy_images(images: List[str], dst_dir: str, copy: bool) -> Tuple[List[str], Dict[str, str]]:
    names = []
    name_map = {}
    for idx, path in enumerate(images):
        old_name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        new_name = f"{idx:06d}"
        _link_or_copy(path, os.path.join(dst_dir, f"{new_name}{ext}"), copy=copy)
        names.append(new_name)
        name_map[old_name] = new_name
    return names, name_map


def _convert_sequence(task: Tuple) -> Dict[str, object]:
    (
        idx,
        total,
        seq_dir,
        out,
        copy_images,
        skip_depth,
        skip_existing_depth,
        depth_jobs,
        max_frames,
        progress_every,
        pose_check_frames,
        pose_check_atol,
        verify_only,
    ) = task
    seq = _seq_name_from_dir(seq_dir)
    rgb_dir = os.path.join(seq_dir, "rgb")
    depth_dir = os.path.join(seq_dir, "depthmap")
    dst_scene = os.path.join(out, seq)
    stats = {
        "sequence": seq,
        "root": None,
        "images": 0,
        "depths": 0,
        "poses": 0,
        "message": "",
    }

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Missing required rgb dir: {rgb_dir}")

    images = _image_files(rgb_dir)
    if max_frames is not None:
        images = images[:max_frames]
    if not images:
        raise RuntimeError(f"No images found in {rgb_dir}")

    src_names = [os.path.splitext(os.path.basename(path))[0] for path in images]
    require_exact_count = max_frames is None
    src_w2c_by_name = _validate_sequence_inputs(seq_dir, images, src_names, require_exact_count=require_exact_count)
    _verify_pose_dir_alignment(seq_dir, src_names, src_w2c_by_name, max_checks=pose_check_frames, atol=pose_check_atol)

    stats["images"] = len(src_names)
    stats["root"] = seq
    stats["poses"] = 1

    if verify_only:
        stats["message"] = f"[{idx}/{total}] verified {seq}: images={stats['images']} poses=1 depths=ok"
        return stats

    names, src_to_dst_name = _copy_images(images, os.path.join(dst_scene, "images", "02"), copy=copy_images)
    dst_to_src_name = {dst: src for src, dst in src_to_dst_name.items()}
    w2c_by_name = {dst: src_w2c_by_name[src] for dst, src in dst_to_src_name.items()}

    if not skip_depth:
        stats["depths"] = _convert_depths(
            seq,
            depth_dir,
            os.path.join(dst_scene, "depths", "02"),
            names,
            depth_jobs=depth_jobs,
            skip_existing=skip_existing_depth,
            progress_every=progress_every,
            src_name_map=dst_to_src_name,
        )

    _write_poses(seq_dir, images, names, dst_scene, w2c_by_name)

    metadata = {
        "source": seq_dir,
        "sequence": seq,
        "camera": "02",
        "rgb_dir": rgb_dir,
        "depth_dir": depth_dir,
        "pose_source": os.path.join(seq_dir, "camera_pose.txt"),
        "pose_convention": "source camera_pose.txt is camera-to-world; output extri.yml is world-to-camera",
        "intrinsics": os.path.join(seq_dir, "intrinsics.txt"),
        "renumbering": {dst: src for dst, src in dst_to_src_name.items()},
    }
    with open(os.path.join(dst_scene, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    stats["message"] = (
        f"[{idx}/{total}] done {seq}: images={stats['images']} "
        f"depth_written={stats['depths']} poses={stats['poses']}"
    )
    return stats


def convert(args: argparse.Namespace) -> None:
    src_root = _resolve_src_root(os.path.abspath(args.src))
    out = os.path.abspath(args.out)
    if not os.path.isdir(src_root):
        raise FileNotFoundError(src_root)
    if not args.verify_only:
        _ensure_dir(out)

    seq_dirs = _collect_seq_dirs(src_root, args.seqs)
    if args.limit is not None:
        seq_dirs = seq_dirs[: args.limit]
    if not seq_dirs:
        raise RuntimeError(f"No *{PROCESSED_SUFFIX} sequences found in {src_root}")

    tasks = [
        (
            idx,
            len(seq_dirs),
            seq_dir,
            out,
            args.copy,
            args.skip_depth,
            args.skip_existing_depth,
            max(1, args.depth_jobs),
            args.max_frames,
            args.progress_every,
            args.pose_check_frames,
            args.pose_check_atol,
            args.verify_only,
        )
        for idx, seq_dir in enumerate(seq_dirs, start=1)
    ]

    action = "Verifying" if args.verify_only else "Converting"
    print(
        f"{action} {len(tasks)} sequences with seq_jobs={args.seq_jobs}, "
        f"depth_jobs={args.depth_jobs}, src={src_root}, out={out}",
        flush=True,
    )

    roots = []
    totals = {"images": 0, "depths": 0, "poses": 0}
    if args.seq_jobs <= 1:
        results = []
        for task in tasks:
            result = _convert_sequence(task)
            results.append(result)
            print(result["message"], flush=True)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.seq_jobs) as executor:
            futures = [executor.submit(_convert_sequence, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                print(results[-1]["message"], flush=True)

    for result in results:
        if result["root"]:
            roots.append(str(result["root"]))
        for key in totals:
            totals[key] += int(result[key])

    roots = sorted(set(roots))
    if not args.verify_only:
        _write_lines(os.path.join(out, "data_roots.txt"), roots)
        _write_lines(os.path.join(out, "train_data_roots.txt"), roots)
        _write_lines(os.path.join(out, "test_data_roots.txt"), [])
        _write_lines(os.path.join(out, "validate_data_roots.txt"), [])

    print("Summary:", flush=True)
    for key, value in totals.items():
        print(f"  {key}: {value}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Junyi42/vbr_processed into the GeneralizableDataset/KITTI-like layout."
    )
    parser.add_argument("--src", required=True, help="Extracted vbr_processed root, e.g. .../vbr_slam_process")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--seqs", default=None, help="Comma-separated sequence names")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences, mainly for debugging")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sequence, mainly for debugging")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--skip-depth", action="store_true", help="Skip .npy depthmap to .exr conversion")
    parser.add_argument("--skip-existing-depth", action="store_true", help="Skip existing output .exr depth files")
    parser.add_argument("--seq-jobs", type=int, default=1, help="Parallel sequence workers")
    parser.add_argument("--depth-jobs", type=int, default=8, help="Per-sequence depth conversion threads")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print depth progress every N frames")
    parser.add_argument("--pose-check-frames", type=int, default=16, help="Compare camera_pose.txt against sampled camera_pose/<frame>.txt files")
    parser.add_argument("--pose-check-atol", type=float, default=1e-4, help="Absolute tolerance for camera_pose.txt vs camera_pose/<frame>.txt checks")
    parser.add_argument("--verify-only", action="store_true", help="Only validate required files and pose convention; do not write output")
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
