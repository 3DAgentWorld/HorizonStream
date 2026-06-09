import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HFSpec:
    repo_id: str
    filename: str
    revision: Optional[str] = None
    local_dir: str = "checkpoints"


def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and len(x) > 0


def resolve_checkpoint_path(
    checkpoint: Optional[str], hf: Optional[dict]
) -> Optional[str]:
    if _is_nonempty_str(checkpoint):
        return checkpoint
    if not isinstance(hf, dict):
        return None

    repo_id = hf.get("repo_id")
    filename = hf.get("filename")
    revision = hf.get("revision", None)
    local_dir = hf.get("local_dir", "checkpoints")

    if not _is_nonempty_str(repo_id) or not _is_nonempty_str(filename):
        return None

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is required for auto-download") from e

    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, local_dir=local_dir
    )
