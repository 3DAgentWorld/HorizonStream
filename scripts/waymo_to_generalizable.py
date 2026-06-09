import argparse
import os
import shutil
from glob import glob


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


def _is_generalizable_scene_root(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "images"))


def _is_generalizable_meta_root(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "data_roots.txt")):
        return True

    scenes = sorted(glob(os.path.join(path, "*")))
    for p in scenes[:50]:
        if os.path.isdir(p) and _is_generalizable_scene_root(p):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", required=True, help="Waymo meta_root in GeneralizableDataset format."
    )
    parser.add_argument(
        "--out", required=True, help="Output meta_root in GeneralizableDataset format."
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene names. Default: use data_roots.txt or scan dirs.",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlinking."
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    copy = bool(args.copy)

    if not _is_generalizable_meta_root(src):
        raise RuntimeError(
            "Unsupported Waymo source layout (expect GeneralizableDataset format)."
        )

    roots_file = os.path.join(src, "data_roots.txt")
    if args.scenes:
        scenes = [s for s in args.scenes.split(",") if s]
    elif os.path.exists(roots_file):
        scenes = _read_list(roots_file)
    else:
        scenes = sorted(
            [
                os.path.basename(p)
                for p in glob(os.path.join(src, "*"))
                if os.path.isdir(p) and _is_generalizable_scene_root(p)
            ]
        )

    _ensure_dir(out)
    for name in scenes:
        src_scene = os.path.join(src, name)
        dst_scene = os.path.join(out, name)
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
                if copy:
                    shutil.copy2(sp, dp)
                else:
                    if not os.path.exists(dp):
                        os.symlink(sp, dp)

    _write_list(os.path.join(out, "data_roots.txt"), scenes)


if __name__ == "__main__":
    main()
