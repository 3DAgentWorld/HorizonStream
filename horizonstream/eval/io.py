import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def frame_stems(image_paths):
    stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    if len(set(stems)) == len(stems):
        return stems
    parents = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    if len(set(parents)) == len(parents):
        return parents
    return stems


def read_pred_w2c_txt(path):
    frames = []
    poses = []
    if not os.path.exists(path):
        return frames, poses
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                continue
            frame = int(vals[0])
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = np.asarray(vals[1:10], dtype=np.float64).reshape(3, 3)
            mat[:3, 3] = np.asarray(vals[10:13], dtype=np.float64)
            frames.append(frame)
            poses.append(mat)
    return frames, poses


def read_opencv_camera_yml(extri_path, intri_path=None):
    if not os.path.exists(extri_path):
        return {}, {}, {}

    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    names_node = fs_extri.getNode("names")
    names = []
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())

    extri = {}
    for name in names:
        rot = fs_extri.getNode(f"Rot_{name}").mat()
        t = fs_extri.getNode(f"T_{name}").mat()
        if rot is None or t is None:
            continue
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = np.asarray(rot, dtype=np.float64)
        mat[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        extri[name] = mat
    fs_extri.release()

    intri = {}
    image_sizes = {}
    if intri_path is not None and os.path.exists(intri_path):
        fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        for name in names:
            K = fs_intri.getNode(f"K_{name}").mat()
            if K is None:
                continue
            intri[name] = np.asarray(K, dtype=np.float64)
            h_node = fs_intri.getNode(f"H_{name}")
            w_node = fs_intri.getNode(f"W_{name}")
            if not h_node.empty() and not w_node.empty():
                image_sizes[name] = (int(h_node.real()), int(w_node.real()))
        fs_intri.release()

    return extri, intri, image_sizes


def read_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(path)
    return depth.astype(np.float32)


def read_ply_xyz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    header = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported PLY format: {path}")

        vertex_count = None
        property_specs = []
        in_vertex_block = False
        for line in header:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex_block = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex_block = False
            if in_vertex_block and line.startswith("property "):
                _, dtype_name, prop_name = line.split()
                property_specs.append((dtype_name, prop_name))

        if vertex_count is None:
            raise ValueError(f"Missing vertex count in PLY: {path}")

        dtype_map = {
            "float": "<f4",
            "float32": "<f4",
            "uchar": "u1",
            "uint8": "u1",
        }
        vertex_dtype = []
        for dtype_name, prop_name in property_specs:
            if dtype_name not in dtype_map:
                raise ValueError(
                    f"Unsupported PLY property type {dtype_name} in {path}"
                )
            vertex_dtype.append((prop_name, dtype_map[dtype_name]))

        data = np.fromfile(f, dtype=np.dtype(vertex_dtype), count=vertex_count)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(
        np.float32, copy=False
    )


def read_pointcloud_xyz(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
        return np.asarray(data, dtype=np.float32).reshape(-1, 3)
    if ext == ".npz":
        data = np.load(path)
        if "points" in data:
            points = data["points"]
        else:
            first_key = next(iter(data.files))
            points = data[first_key]
        return np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return read_ply_xyz(path)
