import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from horizonstream.data.undistort_utils import colmap_undistort_numpy
from horizonstream.utils.vendor.dust3r.utils.image import load_images_for_eval

dataset_metadata: Dict[str, Dict[str, Any]] = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,
    },
    "kitti": {
        "img_path": "data/kitti/sequences",
        "anno_path": "data/kitti/poses",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "image_2"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            anno_path, f"{seq}.txt"
        )
        if os.path.exists(os.path.join(anno_path, f"{seq}.txt"))
        else None,
        "traj_format": "kitti",
        "seq_list": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": None,
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "waymo": {
        "img_path": "toyourdata/scatt3r_evaluation/waymo_open_dataset_v1_4_3",
        "anno_path": None,
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "images",
            seq.split("_cam")[1] if "_cam" in seq else "00",
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "cameras",
            seq.split("_cam")[1] if "_cam" in seq else "00",
            "extri.yml",
        ),
        "traj_format": "waymo",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
}


@dataclass
class HorizonStreamSequenceInfo:
    name: str
    scene_root: str
    image_dir: str
    image_paths: List[str]
    camera: Optional[str]


class HorizonStreamSequence:
    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        image_paths: List[str],
        scene_root: Optional[str] = None,
        image_dir: Optional[str] = None,
        camera: Optional[str] = None,
    ):
        self.name = name
        self.images = images
        self.image_paths = image_paths
        self.scene_root = scene_root
        self.image_dir = image_dir
        self.camera = camera


def _read_list_file(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _is_generalizable_scene_root(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "images"))


def _direct_image_files(dir_path: str) -> List[str]:
    filelist = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpeg")))
    return filelist


def _frame_stems(image_paths: List[str]) -> List[str]:
    stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    if len(set(stems)) == len(stems):
        return stems
    parents = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    if len(set(parents)) == len(parents):
        return parents
    return stems


def _format_fps_for_name(fps: float) -> str:
    return f"{float(fps):g}".replace(".", "p")


def _safe_video_scene_name(path: str, stride: int, target_fps: Optional[float]) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in stem).strip("._")
    if not safe:
        safe = "video"
    if target_fps is not None:
        return f"{safe}_fps{_format_fps_for_name(target_fps)}"
    return f"{safe}_stride{int(stride)}"


class HorizonStreamDataLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.dataset = cfg.get("dataset", None)
        meta = dataset_metadata.get(self.dataset, {})
        self.img_path = cfg.get("img_path", meta.get("img_path"))
        self.mask_path = cfg.get("mask_path", meta.get("mask_path"))
        self.dir_path_func = meta.get("dir_path_func", lambda p, s: os.path.join(p, s))
        self.mask_path_seq_func = meta.get("mask_path_seq_func", lambda p, s: None)
        self.full_seq = bool(cfg.get("full_seq", meta.get("full_seq", True)))
        self.seq_list = cfg.get("seq_list", None)
        self.stride = int(cfg.get("stride", 1))
        self.max_frames = cfg.get("max_frames", None)
        self.size = int(cfg.get("size", 518))
        self.crop = bool(cfg.get("crop", False))
        self.patch_size = int(cfg.get("patch_size", 14))
        self.format = cfg.get("format", "auto")
        self.data_roots_file = cfg.get("data_roots_file", None)
        self.split = cfg.get("split", None)
        self.camera = cfg.get("camera", None)
        self.seq_match_mode = str(cfg.get("seq_match_mode", "exact")).lower()
        self.video_path = cfg.get("video_path", None)
        self.image_paths_cfg = cfg.get("image_paths", None)
        self.image_scene_name = cfg.get("image_scene_name", None)
        self.video_stride = max(1, int(cfg.get("video_stride", 1)))
        self.video_sample_fps = cfg.get("video_sample_fps", None)
        if self.video_sample_fps in ("", "null"):
            self.video_sample_fps = None
        if self.video_sample_fps is not None:
            self.video_sample_fps = float(self.video_sample_fps)
        self.video_scene_name = cfg.get("video_scene_name", None)

        # camera VAL-aligned preprocess: undistort -> resize/crop with K sync.
        self.camera_preprocess = bool(cfg.get("camera_preprocess", True))
        self.camera_preprocess_strict = bool(
            cfg.get("camera_preprocess_strict", True)
        )
        self.undistort_backend = str(
            cfg.get("undistort_backend", "colmap")
        ).lower()
        self.undistort_dist_opt_k = bool(cfg.get("undistort_dist_opt_k", False))
        self.undistort_safe_bound = int(cfg.get("undistort_safe_bound", 4))
        self.undistort_center_crop = bool(cfg.get("undistort_center_crop", True))
        self._undistort_warned_no_pycolmap = False

    def _infer_format(self) -> str:
        if self.image_paths_cfg not in (None, "", "null"):
            return "image_list"
        if self.video_path not in (None, "", "null"):
            return "video"
        if self.format in ["relpose", "generalizable"]:
            return self.format
        if self.img_path is None:
            return "relpose"
        if _is_generalizable_scene_root(self.img_path):
            return "generalizable"
        default_list = self.data_roots_file or "data_roots.txt"
        if os.path.exists(os.path.join(self.img_path, default_list)):
            return "generalizable"
        return "relpose"

    def _resolve_seq_list_generalizable(self) -> List[str]:
        if self.seq_list is not None and self.seq_match_mode not in ["contains", "in"]:
            return list(self.seq_list)
        if self.img_path is None or not os.path.isdir(self.img_path):
            return []

        if _is_generalizable_scene_root(self.img_path):
            return self._filter_seq_candidates([self.img_path])

        candidates = []
        if isinstance(self.data_roots_file, str) and self.data_roots_file:
            candidates.append(self.data_roots_file)
        if isinstance(self.split, str) and self.split:
            split_name = self.split.lower()
            if split_name in ["val", "valid", "validate"]:
                split_name = "validate"
            candidates.append(f"{split_name}_data_roots.txt")
        candidates.append("data_roots.txt")
        candidates.append("train_data_roots.txt")
        candidates.append("validate_data_roots.txt")

        for fname in candidates:
            path = os.path.join(self.img_path, fname)
            if os.path.exists(path):
                return self._filter_seq_candidates(_read_list_file(path))

        img_dirs = sorted(
            glob.glob(os.path.join(self.img_path, "**", "images"), recursive=True)
        )
        scene_roots = [os.path.dirname(p) for p in img_dirs]

        rels = []
        for p in scene_roots:
            try:
                rels.append(os.path.relpath(p, self.img_path))
            except ValueError:
                rels.append(p)
        return self._filter_seq_candidates(sorted(set(rels)))

    def _resolve_seq_list_relpose(self) -> List[str]:
        if self.seq_list is not None and self.seq_match_mode not in ["contains", "in"]:
            return list(self.seq_list)
        meta = dataset_metadata.get(self.dataset, {})
        if self.full_seq:
            if self.img_path is None or not os.path.isdir(self.img_path):
                return []
            seqs = [
                s
                for s in os.listdir(self.img_path)
                if os.path.isdir(os.path.join(self.img_path, s))
            ]
            return self._filter_seq_candidates(sorted(seqs))
        seqs = meta.get("seq_list", []) or []
        return self._filter_seq_candidates(list(seqs))

    def _filter_seq_candidates(self, candidates: List[str]) -> List[str]:
        if self.seq_list is None or self.seq_match_mode not in ["contains", "in"]:
            return candidates
        needles = [str(x).strip() for x in self.seq_list if str(x).strip()]
        if not needles:
            return candidates
        matched = []
        for seq in candidates:
            seq_text = str(seq)
            seq_name = os.path.basename(os.path.normpath(seq_text))
            if any(needle in seq_text or needle in seq_name for needle in needles):
                matched.append(seq)
        return matched

    def _resolve_seq_list(self) -> List[str]:
        fmt = self._infer_format()
        if fmt == "generalizable":
            result = self._resolve_seq_list_generalizable()
        else:
            result = self._resolve_seq_list_relpose()
        if not result:
            print(f"[horizonstream] warning: no sequences resolved (img_path={self.img_path}, seq_list={self.seq_list}, seq_match_mode={self.seq_match_mode})", flush=True)
        return result

    def _sequence_name_from_scene_root(self, seq_entry: str, scene_root: str) -> str:
        scene_root_abs = os.path.abspath(scene_root)
        if self.img_path is not None:
            try:
                rel = os.path.relpath(scene_root_abs, os.path.abspath(self.img_path))
            except ValueError:
                rel = None
            if rel and rel not in [os.curdir, os.pardir] and not rel.startswith(os.pardir + os.path.sep):
                return rel

        if os.path.isabs(seq_entry):
            return os.path.basename(os.path.normpath(seq_entry))

        name = os.path.normpath(seq_entry)
        if name in ["", os.curdir, os.pardir] or name.startswith(os.pardir + os.path.sep):
            return os.path.basename(os.path.normpath(scene_root))
        return name

    def _resolve_scene_root(self, seq_entry: str) -> Tuple[str, str]:
        if os.path.isabs(seq_entry):
            scene_root = seq_entry
        elif os.path.sep in seq_entry:
            scene_root = os.path.join(self.img_path, seq_entry)
        else:
            scene_root = os.path.join(self.img_path, seq_entry)
        name = self._sequence_name_from_scene_root(seq_entry, scene_root)
        return name, scene_root

    def _resolve_image_dir_generalizable(self, scene_root: str) -> Optional[str]:
        images_root = os.path.join(scene_root, "images")
        if not os.path.isdir(images_root):
            return None

        if isinstance(self.camera, str) and self.camera:
            cam_dir = os.path.join(images_root, self.camera)
            if os.path.isdir(cam_dir):
                return cam_dir

        if _direct_image_files(images_root):
            return images_root

        cams = [
            d
            for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        if not cams:
            return None
        cams = sorted(cams)

        frame_dirs = []
        for name in cams:
            child_dir = os.path.join(images_root, name)
            child_images = _direct_image_files(child_dir)
            if child_images:
                frame_dirs.append((name, len(child_images)))

        if (
            len(cams) > 10
            and len(frame_dirs) == len(cams)
            and max(count for _, count in frame_dirs) == 1
        ):
            return images_root

        return os.path.join(images_root, cams[0])

    def _camera_from_image_dir(self, image_dir: str) -> Optional[str]:
        parent = os.path.basename(os.path.dirname(image_dir))
        if parent != "images":
            return None
        return os.path.basename(image_dir)

    def _has_multiple_camera_dirs(self, scene_root: str) -> bool:
        images_root = os.path.join(scene_root, "images")
        if not os.path.isdir(images_root):
            return False
        camera_dirs = []
        for name in os.listdir(images_root):
            child_dir = os.path.join(images_root, name)
            if os.path.isdir(child_dir) and _direct_image_files(child_dir):
                camera_dirs.append(name)
        return len(camera_dirs) > 1

    def _collect_filelist(self, dir_path: str) -> List[str]:
        filelist = _direct_image_files(dir_path)
        if not filelist:
            nested = []
            child_dirs = sorted(
                d for d in glob.glob(os.path.join(dir_path, "*")) if os.path.isdir(d)
            )
            for child_dir in child_dirs:
                child_images = _direct_image_files(child_dir)
                if child_images:
                    nested.append(child_images[0])
            filelist = nested
        if self.stride > 1:
            filelist = filelist[:: self.stride]
        if self.max_frames is not None and self.max_frames > 0:
            filelist = filelist[: self.max_frames]
        return filelist

    def _load_intrinsics_distortion(
        self, scene_root: str, camera: Optional[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
        intri_candidates = []
        if isinstance(camera, str) and camera:
            intri_candidates.append(os.path.join(scene_root, "cameras", camera, "intri.yml"))
        intri_candidates.append(os.path.join(scene_root, "intri.yml"))

        intri_path = None
        for p in intri_candidates:
            if os.path.exists(p):
                intri_path = p
                break
        if intri_path is None:
            raise FileNotFoundError(
                f"intri.yml not found under scene_root={scene_root} camera={camera}"
            )

        fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        names_node = fs.getNode("names")
        names = []
        if not names_node.empty():
            for i in range(names_node.size()):
                names.append(names_node.at(i).string())

        K_map: Dict[str, np.ndarray] = {}
        D_map: Dict[str, np.ndarray] = {}
        for name in names:
            K = fs.getNode(f"K_{name}").mat()
            if K is None:
                continue
            D = fs.getNode(f"D_{name}").mat()
            if D is None:
                D_arr = np.zeros((5, 1), dtype=np.float64)
            else:
                D_arr = np.asarray(D, dtype=np.float64).reshape(-1)
                if D_arr.size < 5:
                    D_arr = np.pad(D_arr, (0, 5 - D_arr.size), mode="constant")
                elif D_arr.size > 5:
                    D_arr = D_arr[:5]
                D_arr = D_arr.reshape(5, 1)

            K_map[name] = np.asarray(K, dtype=np.float64)
            D_map[name] = D_arr

        fs.release()
        if not K_map:
            raise ValueError(f"No intrinsics K_* found in {intri_path}")
        return K_map, D_map, intri_path

    def _undistort_image(
        self,
        img: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = np.asarray(K, dtype=np.float64).copy()
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        if D.size < 5:
            D = np.pad(D, (0, 5 - D.size), mode="constant")
        elif D.size > 5:
            D = D[:5]

        if np.sum(np.abs(D)) == 0.0:
            return img, K

        backend = self.undistort_backend
        if backend == "colmap" and os.name != "nt":
            try:
                undist, new_K = colmap_undistort_numpy(
                    img, K, D.reshape(5, 1), blank_pixels=not self.undistort_dist_opt_k
                )
                undist = np.rint(undist).clip(0, 255).astype(np.uint8)
                return undist, np.asarray(new_K, dtype=np.float64)
            except Exception as exc:
                if not self._undistort_warned_no_pycolmap:
                    print(
                        f"[horizonstream] warning: camera preprocess colmap undistort unavailable ({exc}); fallback to cv2",
                        flush=True,
                    )
                    self._undistort_warned_no_pycolmap = True

        H, W = img.shape[:2]
        if self.undistort_dist_opt_k:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H))
            undist = cv2.undistort(img, K, D, newCameraMatrix=new_K)
            return undist, np.asarray(new_K, dtype=np.float64)

        undist = cv2.undistort(img, K, D)
        return undist, K

    def _resize_crop_with_k(
        self,
        img: np.ndarray,
        K: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape[:2]
        K = np.asarray(K, dtype=np.float64).copy()

        safe_bound = max(int(self.undistort_safe_bound), 0)
        ratio = max(
            float(target_h + safe_bound) / max(H, 1),
            float(target_w + safe_bound) / max(W, 1),
        )
        if ratio <= 0:
            ratio = 1.0

        h = int(H * ratio)
        w = int(W * ratio)
        if h != H or w != W:
            K[0:1] *= float(w) / float(max(W, 1))
            K[1:2] *= float(h) / float(max(H, 1))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            h, w = H, W

        # Match camera strict-center crop behavior:
        # use principal point for crop center and pad with zeros when out-of-bound.
        if self.undistort_center_crop:
            crop_h = int(round(float(K[1, 2]))) - target_h // 2
            crop_w = int(round(float(K[0, 2]))) - target_w // 2
        else:
            crop_h = 0
            crop_w = 0

        end_h = crop_h + target_h
        end_w = crop_w + target_w
        offset_h = crop_h
        offset_w = crop_w

        if self.undistort_center_crop and (
            crop_h < 0 or crop_w < 0 or end_h > h or end_w > w
        ):
            pad_top = max(0, -crop_h)
            pad_bottom = max(0, end_h - h)
            pad_left = max(0, -crop_w)
            pad_right = max(0, end_w - w)
            if pad_top or pad_bottom or pad_left or pad_right:
                img = cv2.copyMakeBorder(
                    img,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                crop_h += pad_top
                crop_w += pad_left
                end_h = crop_h + target_h
                end_w = crop_w + target_w

        img = img[crop_h:end_h, crop_w:end_w]

        K[0, 2] -= offset_w
        K[1, 2] -= offset_h
        return img, K

    def _load_images_preprocessed(self, info: HorizonStreamSequenceInfo) -> torch.Tensor:
        K_map, D_map, intri_path = self._load_intrinsics_distortion(
            info.scene_root, info.camera
        )

        stems = _frame_stems(info.image_paths)
        missing = [s for s in stems if s not in K_map]
        if missing:
            sample = ", ".join(missing[:5])
            msg = (
                f"[horizonstream] preprocess: stem mismatch in {intri_path}, "
                f"missing={len(missing)} sample=[{sample}]"
            )
            if self.camera_preprocess_strict:
                raise KeyError(msg)
            print(msg, flush=True)

        processed = []
        target_h = None
        target_w = None

        for i, path in enumerate(info.image_paths):
            stem = stems[i]
            if stem not in K_map:
                if self.camera_preprocess_strict:
                    raise KeyError(f"Missing K/D for frame stem={stem} path={path}")
                continue

            K = K_map[stem]
            D = D_map.get(stem, np.zeros((5, 1), dtype=np.float64))

            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise IOError(f"Failed to read image: {path}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            img, K = self._undistort_image(img, K, D)

            if target_h is None or target_w is None:
                H0, W0 = img.shape[:2]
                if self.size > 0:
                    h = int(float(H0) / float(max(W0, 1)) * float(self.size))
                    w = int(self.size)
                else:
                    h, w = H0, W0
                target_h = max((h // self.patch_size) * self.patch_size, self.patch_size)
                target_w = max((w // self.patch_size) * self.patch_size, self.patch_size)

            img, _ = self._resize_crop_with_k(img, K, target_h, target_w)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = (img_t - 0.5) / 0.5
            processed.append(img_t[None])

        if not processed:
            raise RuntimeError(f"preprocess produced 0 frames for {info.name}")

        imgs = torch.cat(processed, dim=0)
        images = imgs.unsqueeze(0)
        images = (images + 1.0) / 2.0
        return images

    def _load_images(self, info: HorizonStreamSequenceInfo) -> torch.Tensor:
        if self.camera_preprocess:
            try:
                return self._load_images_preprocessed(info)
            except Exception as exc:
                if self.camera_preprocess_strict and not isinstance(exc, FileNotFoundError):
                    raise
                print(
                    f"[horizonstream] warning: preprocess failed for {info.name}, fallback to load_images_for_eval: {exc}",
                    flush=True,
                )

        views = load_images_for_eval(
            info.image_paths,
            size=self.size,
            verbose=False,
            crop=self.crop,
            patch_size=self.patch_size,
        )
        imgs = torch.cat([view["img"] for view in views], dim=0)
        images = imgs.unsqueeze(0)
        images = (images + 1.0) / 2.0
        return images

    def _preprocess_rgb_frame_for_eval(self, rgb: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        w1, h1 = img.size
        if self.size == 224:
            long_edge = round(self.size * max(w1 / h1, h1 / w1))
        else:
            long_edge = self.size

        scale = float(long_edge) / float(max(w1, h1))
        new_size = tuple(int(round(x * scale)) for x in img.size)
        interp = Image.Resampling.LANCZOS if max(w1, h1) > long_edge else Image.Resampling.BICUBIC
        img = img.resize(new_size, interp)

        w, h = img.size
        cx, cy = w // 2, h // 2
        if self.size == 224:
            half = min(cx, cy)
            if self.crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                img = img.resize((2 * half, 2 * half), Image.Resampling.LANCZOS)
        else:
            halfw = ((2 * cx) // self.patch_size) * (self.patch_size // 2)
            halfh = ((2 * cy) // self.patch_size) * (self.patch_size // 2)
            if w == h:
                halfh = int(3 * halfw / 4)
            if self.crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:
                img = img.resize((2 * halfw, 2 * halfh), Image.Resampling.LANCZOS)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def _load_video_sequence(self) -> HorizonStreamSequence:
        video_path = os.path.abspath(os.path.expanduser(str(self.video_path)))
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if self.video_sample_fps is not None and self.video_sample_fps <= 0.0:
            raise ValueError(f"video_sample_fps must be positive, got {self.video_sample_fps}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0.0:
            source_fps = None
        if self.video_sample_fps is not None and source_fps is None:
            cap.release()
            raise ValueError(f"Cannot sample video by FPS because source FPS is unavailable: {video_path}")

        scene_name = self.video_scene_name or _safe_video_scene_name(
            video_path,
            self.video_stride,
            self.video_sample_fps,
        )
        effective_fps = (
            min(float(self.video_sample_fps), float(source_fps))
            if self.video_sample_fps is not None and source_fps is not None
            else (None if source_fps is None else source_fps / float(self.video_stride))
        )
        tensors = []
        source_idx = 0
        out_idx = 0

        def should_keep_frame(frame_idx: int, output_idx: int) -> bool:
            if self.video_sample_fps is None:
                return frame_idx % self.video_stride == 0
            if source_fps is None or self.video_sample_fps >= source_fps:
                return True
            target_source_idx = int(round(float(output_idx) * float(source_fps) / float(self.video_sample_fps)))
            return frame_idx >= target_source_idx

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if should_keep_frame(source_idx, out_idx):
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    tensors.append(self._preprocess_rgb_frame_for_eval(rgb)[None])
                    out_idx += 1
                    if self.max_frames is not None and self.max_frames > 0 and out_idx >= self.max_frames:
                        break
                source_idx += 1
        finally:
            cap.release()

        if not tensors:
            raise RuntimeError(f"No frames read from video: {video_path}")

        imgs = torch.cat(tensors, dim=0)
        images = ((imgs.unsqueeze(0)) + 1.0) / 2.0
        print(
            f"[horizonstream] loaded video {scene_name}: path={video_path} "
            f"frames={out_idx} last_source_frame={source_idx} stride={self.video_stride} "
            f"target_fps={self.video_sample_fps} source_fps={source_fps} "
            f"effective_fps={effective_fps} shape={tuple(images.shape)}",
            flush=True,
        )
        return HorizonStreamSequence(
            scene_name,
            images,
            [],
            scene_root=os.path.dirname(video_path),
            image_dir=None,
            camera="video",
        )

    def _load_image_list_sequence(self) -> HorizonStreamSequence:
        image_paths = self.image_paths_cfg
        if isinstance(image_paths, str):
            image_paths = _read_list_file(image_paths) if os.path.isfile(image_paths) else [image_paths]
        image_paths = [os.path.abspath(os.path.expanduser(str(p))) for p in (image_paths or [])]
        image_paths = [p for p in image_paths if p and os.path.isfile(p)]
        if self.max_frames is not None and self.max_frames > 0:
            image_paths = image_paths[: int(self.max_frames)]
        if not image_paths:
            raise RuntimeError("No images found for data.image_paths")
        scene_name = self.image_scene_name or "images"
        info = HorizonStreamSequenceInfo(
            name=scene_name,
            scene_root=os.path.dirname(image_paths[0]),
            image_dir=os.path.dirname(image_paths[0]),
            image_paths=image_paths,
            camera="image_list",
        )
        print(
            f"[horizonstream] loading image list {scene_name}: {len(image_paths)} frames",
            flush=True,
        )
        images = self._load_images(info)
        print(
            f"[horizonstream] loaded image list {scene_name}: {tuple(images.shape)}",
            flush=True,
        )
        return HorizonStreamSequence(
            scene_name,
            images,
            image_paths,
            scene_root=info.scene_root,
            image_dir=info.image_dir,
            camera=info.camera,
        )

    def iter_sequence_infos(self) -> Iterator[HorizonStreamSequenceInfo]:
        fmt = self._infer_format()
        seqs = self._resolve_seq_list()
        for seq_entry in seqs:
            if fmt == "generalizable":
                seq, scene_root = self._resolve_scene_root(seq_entry)
                dir_path = self._resolve_image_dir_generalizable(scene_root)
                if dir_path is None or not os.path.isdir(dir_path):
                    print(f"[horizonstream] skip seq={seq_entry}: image dir not found under {scene_root}", flush=True)
                    continue
                camera = self._camera_from_image_dir(dir_path)
                if camera is not None and self._has_multiple_camera_dirs(scene_root):
                    seq = os.path.join(seq, camera)
            else:
                seq = seq_entry
                scene_root = os.path.join(self.img_path, seq)
                dir_path = self.dir_path_func(self.img_path, seq)
                if not os.path.isdir(dir_path):
                    continue
                camera = None

            filelist = self._collect_filelist(dir_path)
            if not filelist:
                print(f"[horizonstream] skip seq={seq_entry}: no images found in {dir_path}", flush=True)
                continue
            yield HorizonStreamSequenceInfo(
                name=seq,
                scene_root=scene_root,
                image_dir=dir_path,
                image_paths=filelist,
                camera=camera,
            )

    def __iter__(self) -> Iterator[HorizonStreamSequence]:
        if self._infer_format() == "image_list":
            yield self._load_image_list_sequence()
            return
        if self._infer_format() == "video":
            yield self._load_video_sequence()
            return

        for info in self.iter_sequence_infos():
            print(
                f"[horizonstream] loading sequence {info.name}: {len(info.image_paths)} frames",
                flush=True,
            )
            images = self._load_images(info)
            print(
                f"[horizonstream] loaded sequence {info.name}: {tuple(images.shape)}",
                flush=True,
            )
            yield HorizonStreamSequence(
                info.name,
                images,
                info.image_paths,
                scene_root=info.scene_root,
                image_dir=info.image_dir,
                camera=info.camera,
            )
