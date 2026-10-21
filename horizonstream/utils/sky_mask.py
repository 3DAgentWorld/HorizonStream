import os
import copy
import ctypes
import glob
import cv2
import numpy as np
import urllib.error
import urllib.request

try:
    import onnxruntime
except Exception as exc:
    onnxruntime = None
    ONNXRUNTIME_IMPORT_ERROR = exc
else:
    ONNXRUNTIME_IMPORT_ERROR = None

SKYSEG_THRESHOLD = 0.5
SKYSEG_REMOTE_URL = os.environ.get(
    "LONGSTREAM_SKYSEG_REMOTE_URL",
    "https://huggingface.co/NicolasCC/HorizonStream/resolve/main/skyseg.onnx",
).strip()
SKYSEG_DOWNLOAD_TIMEOUT_SEC = int(os.environ.get("LONGSTREAM_SKYSEG_DOWNLOAD_TIMEOUT_SEC", "120"))
SKYSEG_TRT_ENGINE_CACHE_DIR = os.environ.get(
    "LONGSTREAM_SKYSEG_TRT_CACHE_DIR",
    os.path.join("checkpoints", "trt_cache"),
).strip()


def _preload_tensorrt_libs() -> bool:
    """Make pip-installed TensorRT visible to onnxruntime.

    ``pip install tensorrt`` drops the shared objects into
    ``site-packages/tensorrt_libs/``, which is not on the dynamic loader path,
    so onnxruntime fails to dlopen its TensorRT provider unless the caller sets
    LD_LIBRARY_PATH. Loading them here with RTLD_GLOBAL registers them under
    their sonames, which is enough for that dlopen to resolve.
    """
    try:
        import tensorrt_libs
    except Exception:
        try:
            import tensorrt  # noqa: F401  (system install; already on the path)
        except Exception:
            return False
        return True

    lib_dir = os.path.dirname(os.path.abspath(tensorrt_libs.__file__))
    loaded = False
    for pattern in ("libnvinfer.so.*", "libnvonnxparser.so.*"):
        for path in sorted(glob.glob(os.path.join(lib_dir, pattern))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
    return loaded


def _build_skyseg_session(model_path: str):
    """Build an onnxruntime session, preferring TensorRT > CUDA > CPU.

    onnxruntime silently falls back to CPU-only when ``providers`` is omitted,
    which makes sky segmentation the slowest stage of a long sequence
    (~92 ms/frame on CPU vs ~6.7 ms on CUDA and ~2.3 ms on TensorRT).
    Each provider is attempted in turn so that a missing or mismatched GPU
    runtime degrades instead of breaking inference.
    """
    available = set(onnxruntime.get_available_providers())
    attempts = []

    if "TensorrtExecutionProvider" in available and _preload_tensorrt_libs():
        # Parsing skyseg.onnx emits ~150 harmless "Empty initializer ... was
        # provided with non-empty data" warnings on every run, cached engine or
        # not. They come from the global logger, so SessionOptions cannot mute
        # them. Drop to ERROR so real failures still surface.
        try:
            onnxruntime.set_default_logger_severity(3)
        except Exception:
            pass
        cache_dir = os.path.abspath(os.path.expanduser(SKYSEG_TRT_ENGINE_CACHE_DIR))
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = None
        if cache_dir is not None:
            # Engine compilation costs ~15 s on first use; caching it makes
            # every later run start in well under a second.
            attempts.append((
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                [
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache_dir,
                        "trt_fp16_enable": True,
                    },
                    {},
                    {},
                ],
            ))

    if "CUDAExecutionProvider" in available:
        attempts.append((["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]))

    attempts.append((["CPUExecutionProvider"], [{}]))

    for providers, provider_options in attempts:
        try:
            session = onnxruntime.InferenceSession(
                model_path, providers=providers, provider_options=provider_options
            )
        except Exception as exc:
            print(
                f"[horizonstream] sky mask: {providers[0]} unavailable "
                f"({type(exc).__name__}: {str(exc).splitlines()[0][:160]}), trying next provider",
                flush=True,
            )
            continue
        print(f"[horizonstream] sky mask providers: {session.get_providers()}", flush=True)
        return session

    raise RuntimeError(f"Failed to create an onnxruntime session for {model_path}")


def run_skyseg(session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    input_name = session.get_inputs()[0].name
    result_map = session.run(None, {input_name: x})[0]
    return result_map[0, 0]


def _normalize_skyseg_output(result_map):
    result_map = np.asarray(result_map, dtype=np.float32)
    if result_map.size == 0:
        return result_map
    finite = np.isfinite(result_map)
    if not np.any(finite):
        return np.zeros_like(result_map, dtype=np.float32)
    result_map = np.nan_to_num(result_map, nan=0.0, posinf=1.0, neginf=0.0)
    max_value = float(result_map.max())
    min_value = float(result_map.min())
    if min_value >= 0.0 and max_value > 1.5:
        result_map = result_map / 255.0
    return np.clip(result_map, 0.0, 1.0)


def sky_mask_filename(image_path, index=None):
    if not isinstance(image_path, str):
        return f"frame_{0 if index is None else int(index):06d}.png"
    parent = os.path.basename(os.path.dirname(image_path))
    name = os.path.basename(image_path)
    if "#frame=" in name:
        frame_text = name.rsplit("#frame=", 1)[-1]
        return f"video_frame_{int(frame_text):06d}.png" if frame_text.isdigit() else f"video_{index:06d}.png"
    if parent:
        return f"{parent}__{name}"
    return name


def segment_sky(image_path, session, mask_filename=None):
    if isinstance(image_path, np.ndarray):
        image = np.asarray(image_path)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(image_path)
    if image is None:
        return None
    result_map = run_skyseg(session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))
    result_map_original = _normalize_skyseg_output(result_map_original)
    output_mask = np.zeros(result_map_original.shape, dtype=np.uint8)
    output_mask[result_map_original < SKYSEG_THRESHOLD] = 255
    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    return output_mask


def _download_skyseg_model(remote_url: str, model_path: str, timeout_sec: int) -> bool:
    if not remote_url:
        print("[horizonstream] skyseg remote url is empty, skip sky mask download", flush=True)
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        tmp_path = model_path + ".tmp"
        print(
            f"[horizonstream] downloading skyseg.onnx from {remote_url} to {model_path} "
            f"(timeout={timeout_sec}s)",
            flush=True,
        )
        with urllib.request.urlopen(remote_url, timeout=timeout_sec) as resp:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(tmp_path, model_path)
        return True
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[horizonstream] failed to download skyseg.onnx, continue without sky mask: {exc}", flush=True)
        return False


def compute_sky_mask(image_paths, model_path: str, target_dir: str = None):
    if onnxruntime is None:
        print(
            f"[horizonstream] onnxruntime unavailable, skip sky mask: {ONNXRUNTIME_IMPORT_ERROR}",
            flush=True,
        )
        return None
    if not os.path.exists(model_path):
        ok = _download_skyseg_model(
            remote_url=SKYSEG_REMOTE_URL,
            model_path=model_path,
            timeout_sec=SKYSEG_DOWNLOAD_TIMEOUT_SEC,
        )
        if not ok:
            return None
    if not os.path.exists(model_path):
        print(f"[horizonstream] skyseg.onnx not found: {model_path}", flush=True)
        return None
    session = _build_skyseg_session(model_path)
    masks = []
    for idx, image_path in enumerate(image_paths):
        mask_filepath = None
        if target_dir is not None:
            name = sky_mask_filename(image_path, idx)
            mask_filepath = os.path.join(target_dir, name)
            if os.path.exists(mask_filepath):
                sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            else:
                sky_mask = segment_sky(image_path, session, mask_filepath)
        else:
            sky_mask = segment_sky(image_path, session, None)
        masks.append(sky_mask)
    return masks
