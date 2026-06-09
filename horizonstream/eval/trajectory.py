import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from horizonstream.eval.io import read_pred_w2c_txt
from horizonstream.loop.runtime import (
    compact_gt_metrics,
    evaluate_trajectory_against_gt,
    load_ground_truth_trajectory,
)


POSE_VARIANT_FILES = {
    "main": "abs_pose.txt",
    "online": "abs_pose.txt",
    "abs": "abs_pose.txt",
    "offline": "offline_abs_pose.txt",
    "loop": "loop_abs_pose.txt",
    "lc": "loop_abs_pose.txt",
}


def _read_pose_variant(seq_dir: str, variant: str):
    filename = POSE_VARIANT_FILES.get(variant, f"{variant}_abs_pose.txt")
    pose_path = os.path.join(seq_dir, "poses", filename)
    frame_ids, w2c = read_pred_w2c_txt(pose_path)
    if not w2c:
        return None
    w2c = np.asarray(w2c, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    return pose_path, frame_ids, c2w


def _candidate_sequence_dirs(output_root: str) -> List[str]:
    if os.path.exists(os.path.join(output_root, "poses", "abs_pose.txt")):
        return [output_root]
    candidates = []
    for root, dirs, files in os.walk(output_root):
        if "abs_pose.txt" in files and os.path.basename(root) == "poses":
            candidates.append(os.path.dirname(root))
            dirs[:] = []
    return sorted(set(candidates))


def _sequence_name(output_root: str, seq_dir: str) -> str:
    try:
        rel = os.path.relpath(seq_dir, output_root)
    except ValueError:
        rel = os.path.basename(os.path.normpath(seq_dir))
    if rel in ("", "."):
        return os.path.basename(os.path.normpath(seq_dir))
    return rel


def evaluate_sequence_dir(
    seq_dir: str,
    *,
    data_cfg: Optional[dict],
    seq_name_hint: Optional[str] = None,
    variants: Iterable[str] = ("main", "offline", "loop"),
    verbose: bool = True,
    save: bool = True,
) -> Dict[str, object]:
    gt = load_ground_truth_trajectory(
        seq_dir,
        data_cfg,
        verbose=verbose,
        seq_name_hint=seq_name_hint,
    )
    metrics: Dict[str, object] = {}
    pose_paths: Dict[str, str] = {}

    for variant in variants:
        variant = str(variant).strip()
        if not variant:
            continue
        pose = _read_pose_variant(seq_dir, variant)
        if pose is None:
            continue
        pose_path, frame_ids, c2w = pose
        pose_paths[variant] = pose_path
        full_metrics = evaluate_trajectory_against_gt(c2w, frame_ids, gt)
        metrics[variant] = compact_gt_metrics(full_metrics)

    summary = {
        "sequence_dir": seq_dir,
        "sequence": seq_name_hint or os.path.basename(os.path.normpath(seq_dir)),
        "gt_pose": {
            "available": gt is not None,
            "scene_root": None if gt is None else gt.scene_root,
            "camera": None if gt is None else gt.camera,
        },
        "pose_paths": pose_paths,
        "metrics": metrics,
    }
    if save:
        out_dir = os.path.join(seq_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "trajectory_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def evaluate_output_root(
    output_root: str,
    *,
    data_cfg: Optional[dict],
    variants: Iterable[str] = ("main", "online", "offline", "last", "lag", "loop"),
    verbose: bool = True,
    save: bool = True,
) -> Dict[str, object]:
    output_root = os.path.abspath(os.path.expanduser(output_root))
    seq_dirs = _candidate_sequence_dirs(output_root)
    summaries = {}
    for seq_dir in seq_dirs:
        name = _sequence_name(output_root, seq_dir)
        summaries[name] = evaluate_sequence_dir(
            seq_dir,
            data_cfg=data_cfg,
            seq_name_hint=name,
            variants=variants,
            verbose=verbose,
            save=save,
        )

    root_summary = {
        "output_root": output_root,
        "num_sequences": len(summaries),
        "sequences": summaries,
    }
    if save:
        os.makedirs(output_root, exist_ok=True)
        with open(os.path.join(output_root, "eval_summary.json"), "w") as f:
            json.dump(root_summary, f, indent=2)
    return root_summary
