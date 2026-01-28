# worker.py (RunPod Serverless)
import os
import time
import gc
import base64
import binascii
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# --- ENV hardening ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- hf cached_download compatibility ---
import huggingface_hub as h
if not hasattr(h, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download
    def _cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)
    h.cached_download = _cached_download

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RMSNorm fallback (some builds) ---
if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__()
            self.dim = dim
            self.eps = eps
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            var = x.pow(2).mean(-1, keepdim=True)
            x_norm = x / torch.sqrt(var + self.eps)
            if getattr(self, "weight", None) is not None:
                x_norm = x_norm * self.weight
            return x_norm

    nn.RMSNorm = RMSNorm

# --- Some torch builds don't like enable_gqa kwarg ---
if hasattr(F, "scaled_dot_product_attention"):
    _orig_sdp = F.scaled_dot_product_attention
    def patched_sdp_attention(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdp(*args, **kwargs)
    F.scaled_dot_product_attention = patched_sdp_attention

import runpod

# ---------------------------
# Paths / Config
# ---------------------------
def _normalize_model_path(p: str) -> str:
    if not p:
        return p
    p = p.strip()
    if p.startswith("workspace/"):
        p = "/" + p
    if p.startswith("./workspace/"):
        p = p[1:]
    while "//" in p:
        p = p.replace("//", "/")
    return p

DEFAULT_T2V_PATH = "/runpod-volume/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/runpod-volume/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_pipe_t2v = None
_pipe_i2v = None

# ---------------------------
# Utils
# ---------------------------
def _cuda_cleanup():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    _cuda_cleanup()

def _gpu_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        try:
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_mb"] = int(props.total_memory / (1024 * 1024))
        except Exception:
            pass
    return info

def _diffusers_info():
    try:
        import diffusers
        return {"diffusers_version": getattr(diffusers, "__version__", "unknown")}
    except Exception as e:
        return {"diffusers_version": None, "diffusers_import_error": str(e)}

def _assert_model_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise RuntimeError(f"{label} model path not found: {path}. (En serverless puede ser /runpod-volume/...)")

def _pipe_memory_tweaks(pipe):
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe

def _list_dir_safe(path: str, limit: int = 200):
    try:
        items = sorted(os.listdir(path))
        if len(items) > limit:
            return items[:limit] + [f"...(+{len(items)-limit} more)"]
        return items
    except Exception as e:
        return [f"<cannot list: {e}>"]

def _probe_model_tree(base_path: str):
    out = {
        "path": base_path,
        "exists": os.path.isdir(base_path),
        "top": [],
        "checks": {},
        "subfolders": {},
    }
    if not out["exists"]:
        return out

    out["top"] = _list_dir_safe(base_path, limit=200)

    must_files = ["model_index.json"]
    optional = ["config.json", "scheduler", "tokenizer", "text_encoder", "transformer", "unet", "vae"]

    for f in must_files:
        out["checks"][f] = os.path.exists(os.path.join(base_path, f))

    for name in optional:
        p = os.path.join(base_path, name)
        out["subfolders"][name] = {
            "exists": os.path.exists(p),
            "is_dir": os.path.isdir(p),
        }

    for sf in ["vae", "unet", "transformer", "scheduler", "text_encoder", "tokenizer"]:
        p = os.path.join(base_path, sf)
        if os.path.isdir(p):
            out["subfolders"][sf]["items"] = _list_dir_safe(p, limit=80)

    return out

def _lazy_import_wan():
    try:
        from diffusers import WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
        return WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline, None
    except Exception as e:
        return None, None, None, str(e)

# ---------- Robust base64 image decode ----------
def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("image_b64 vacío")

    s = str(s).strip()

    # DataURL
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()

    # urlsafe base64
    s = s.replace("-", "+").replace("_", "/")

    # padding
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad

    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image_b64 inválido: {e}")

def _b64_to_pil_image(image_b64: str):
    from PIL import Image
    raw = _decode_b64(image_b64)
    img = Image.open(BytesIO(raw))
    img.load()
    return img.convert("RGB")

# ---------------------------
# Frames -> MP4 bytes
# ---------------------------
def _to_uint8_hwc(frame):
    import numpy as np

    # PIL
    if hasattr(frame, "convert"):
        arr = np.array(frame.convert("RGB"), dtype=np.uint8)
        return arr

    if torch.is_tensor(frame):
        t = frame.detach().float().cpu()
        arr = t.numpy()
    else:
        arr = np.asarray(frame)

    # squeeze leading 1
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]

    # (C,H,W) -> (H,W,C)
    if arr.ndim == 3:
        c_first = arr.shape[0] in (1, 3, 4)
        c_last_ok = arr.shape[-1] in (1, 3, 4)
        if c_first and not c_last_ok:
            arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4):
        raise RuntimeError(f"BAD_FRAME_CHANNELS: shape={arr.shape} dtype={arr.dtype}")

    return arr

def _normalize_frames(frames):
    import numpy as np

    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()

    if isinstance(frames, np.ndarray):
        while frames.ndim >= 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim == 4:
            out = []
            for i in range(frames.shape[0]):
                out.append(_to_uint8_hwc(frames[i]))
            return out

    out = []
    for f in frames:
        out.append(_to_uint8_hwc(f))
    return out

def _frames_to_mp4_bytes(frames, fps: int = 24) -> bytes:
    import imageio.v2 as imageio
    import tempfile

    frames_u8 = _normalize_frames(frames)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        writer = imageio.get_writer(tmp.name, fps=fps, codec="libx264", quality=8)
        try:
            for arr in frames_u8:
                writer.append_data(arr)
        finally:
            writer.close()

        tmp.seek(0)
        return tmp.read()

def _extract_frames(result):
    if isinstance(result, dict):
        for k in ("frames", "videos", "video"):
            if k in result:
                v = result[k]
                if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                    return v[0]
                return v

    for k in ("frames", "videos", "video"):
        if hasattr(result, k):
            v = getattr(result, k)
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                return v[0]
            return v

    try:
        return result[0]
    except Exception:
        pass

    raise RuntimeError(f"Could not extract frames from result type={type(result)}")

# ---------------------------
# Timing + Dims helpers (3s/5s + /16)
# ---------------------------
def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(round(float(v)))
    except Exception:
        return default
    return max(lo, min(hi, n))

def _snap16(n: int) -> int:
    n = int(n)
    # round to nearest multiple of 16
    r = int(round(n / 16.0) * 16)
    return max(16, r)

def _pick_dims(width, height, aspect_ratio: str = "", platform_ref: str = "") -> Tuple[int, int]:
    # 1) Si vienen width/height válidos, respétalos
    try:
        w = int(width) if width is not None else None
        h = int(height) if height is not None else None
        if w and h and w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    ar = (aspect_ratio or "").strip()

    # 2) aspect_ratio explícito
    if ar == "9:16":
        return 1080, 1920
    if ar == "1:1":
        return 1080, 1080
    if ar == "16:9":
        return 1920, 1080
    if ar == "4:5":
        return 1080, 1350
    if ar == "4:3":
        return 1440, 1080

    # 3) platform_ref
    p = (platform_ref or "").lower().strip()
    if p == "tiktok":
        return 1080, 1920
    if p == "instagram":
        return 1080, 1920
    if p == "youtube":
        return 1920, 1080
    if p == "facebook":
        return 1440, 1080

    # 4) fallback
    return 768, 432

def _normalize_timing(inp: Dict[str, Any]) -> Tuple[int, int, int]:
    seconds_raw = inp.get("duration_s", None)
    if seconds_raw is None:
        seconds_raw = inp.get("seconds", None)
    if seconds_raw is None:
        seconds_raw = 4

    seconds = _clamp_int(seconds_raw, 3, 5, 3)
    seconds = 3 if seconds < 4 else 5

    fps = _clamp_int(inp.get("fps", 24), 8, 60, 24)

    nf_raw = inp.get("num_frames", None)
    if nf_raw is None:
        nf_raw = inp.get("frames", None)
    if nf_raw is None:
        nf_raw = seconds * fps

    num_frames = _clamp_int(nf_raw, 1, 9999, seconds * fps)
    return seconds, fps, num_frames

# ---------------------------
# Pipelines
# ---------------------------
def _unload_pipes(keep: Optional[str] = None):
    global _pipe_t2v, _pipe_i2v
    if keep != "t2v" and _pipe_t2v is not None:
        try:
            del _pipe_t2v
        except Exception:
            pass
        _pipe_t2v = None
    if keep != "i2v" and _pipe_i2v is not None:
        try:
            del _pipe_i2v
        except Exception:
            pass
        _pipe_i2v = None
    _hard_cleanup()

def _load_t2v(unload_other: bool = True):
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    if unload_other:
        _unload_pipes(keep="t2v")

    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")

    WanPipeline, AutoencoderKLWan, _, err = _lazy_import_wan()
    if err:
        raise RuntimeError(
            "WAN_DIFFUSERS_IMPORT_FAILED: No se pudo importar WanPipeline/AutoencoderKLWan desde diffusers. "
            f"Detalle: {err}"
        )

    t0 = time.time()
    print(f"[WAN_LOAD] Loading T2V LOCAL from: {MODEL_T2V_LOCAL}")

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_T2V_LOCAL,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    pipe = WanPipeline.from_pretrained(
        MODEL_T2V_LOCAL,
        vae=vae,
        torch_dtype=DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    _pipe_t2v = _pipe_memory_tweaks(pipe)
    print(f"[WAN_LOAD] T2V loaded in {time.time() - t0:.2f}s")
    return _pipe_t2v

def _load_i2v(unload_other: bool = True):
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    if unload_other:
        _unload_pipes(keep="i2v")

    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")

    _, AutoencoderKLWan, WanImageToVideoPipeline, err = _lazy_import_wan()
    if err:
        raise RuntimeError(
            "WAN_DIFFUSERS_IMPORT_FAILED: No se pudo importar WanImageToVideoPipeline/AutoencoderKLWan desde diffusers. "
            f"Detalle: {err}"
        )

    t0 = time.time()
    print(f"[WAN_LOAD] Loading I2V LOCAL from: {MODEL_I2V_LOCAL}")

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_I2V_LOCAL,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_I2V_LOCAL,
        vae=vae,
        torch_dtype=DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    _pipe_i2v = _pipe_memory_tweaks(pipe)
    print(f"[WAN_LOAD] I2V loaded in {time.time() - t0:.2f}s")
    return _pipe_i2v

# ---------------------------
# Generators
# ---------------------------
def _t2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_t2v(unload_other=True)

    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    negative = str(inp.get("negative_prompt") or "").strip()
    platform_ref = str(inp.get("platform_ref") or "")
    aspect_ratio = str(inp.get("aspect_ratio") or "")

    seconds, fps, num_frames = _normalize_timing(inp)
    width_raw, height_raw = _pick_dims(inp.get("width"), inp.get("height"), aspect_ratio, platform_ref)

    # ✅ SNAP /16 (esto evita el error: width/height must be divisible by 16)
    width = _snap16(width_raw)
    height = _snap16(height_raw)

    steps = _clamp_int(inp.get("steps", 20), 1, 80, 20)
    guidance_scale = float(inp.get("guidance_scale", 5.0) or 5.0)

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative if negative else None,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(result)
    mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
    mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    return {
        "ok": True,
        "mode": "t2v",
        "width": width,
        "height": height,
        "seconds": seconds,
        "fps": fps,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "elapsed_s": round(time.time() - t0, 3),
        # ✅ lo importante:
        "video_b64": mp4_b64,
        "video_mime": "video/mp4",
    }

def _i2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_i2v(unload_other=True)

    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    image_b64 = inp.get("image_b64") or inp.get("image") or inp.get("init_image_b64")
    if not image_b64:
        raise RuntimeError("Falta image_b64")

    init_img = _b64_to_pil_image(str(image_b64))

    negative = str(inp.get("negative_prompt") or "").strip()
    platform_ref = str(inp.get("platform_ref") or "")
    aspect_ratio = str(inp.get("aspect_ratio") or "")

    seconds, fps, num_frames = _normalize_timing(inp)
    width_raw, height_raw = _pick_dims(inp.get("width"), inp.get("height"), aspect_ratio, platform_ref)

    width = _snap16(width_raw)
    height = _snap16(height_raw)

    steps = _clamp_int(inp.get("steps", 20), 1, 80, 20)
    guidance_scale = float(inp.get("guidance_scale", 5.0) or 5.0)

    # (opcional) resize init image al tamaño que pedimos
    try:
        init_img = init_img.resize((width, height))
    except Exception:
        pass

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=init_img,
            negative_prompt=negative if negative else None,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(result)
    mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
    mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    return {
        "ok": True,
        "mode": "i2v",
        "width": width,
        "height": height,
        "seconds": seconds,
        "fps": fps,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": mp4_b64,
        "video_mime": "video/mp4",
    }

# ---------------------------
# Handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    ping = str(inp.get("ping") or "").strip().lower()

    # ✅ compat con tu backend: mode=t2v/i2v
    mode = str(inp.get("mode") or "").strip().lower()
    if not ping and mode:
        if mode == "t2v":
            ping = "t2v_generate"
        elif mode == "i2v":
            ping = "i2v_generate"

    # ---- debug ----
    if ping in ("echo", "debug"):
        ftfy_ok = True
        try:
            import ftfy  # noqa
        except Exception:
            ftfy_ok = False

        return {
            "ok": True,
            "msg": "ECHO_OK",
            "job_keys": list(job.keys()),
            "input": inp,
            "gpu_info": _gpu_info(),
            "ftfy_available": ftfy_ok,
            **_diffusers_info(),
            "env": {
                "WAN_T2V_PATH": os.environ.get("WAN_T2V_PATH"),
                "WAN_I2V_PATH": os.environ.get("WAN_I2V_PATH"),
            },
            "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
        }

    if ping == "smoke":
        return {"ok": True, "msg": "SMOKE_OK", "gpu_info": _gpu_info(), **_diffusers_info()}

    if ping == "gpu_sanity":
        return {"ok": True, **_gpu_info(), **_diffusers_info()}

    if ping == "list_paths":
        candidates = [
            "/",
            "/workspace",
            "/workspace/models",
            "/workspace/models/wan22",
            "/runpod-volume",
            "/runpod-volume/models",
            "/runpod-volume/models/wan22",
        ]
        out = {"ok": True, "msg": "LIST_PATHS_OK", "paths": {}}
        for p in candidates:
            out["paths"][p] = {
                "exists": os.path.exists(p),
                "is_dir": os.path.isdir(p),
                "items": _list_dir_safe(p, limit=200) if os.path.isdir(p) else []
            }
        out["gpu_info"] = _gpu_info()
        out.update(_diffusers_info())
        out["resolved_paths"] = {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL}
        return out

    if ping in ("probe_models", "probe_model", "models_probe"):
        return {
            "ok": True,
            "msg": "PROBE_OK",
            "paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
            "t2v": _probe_model_tree(MODEL_T2V_LOCAL),
            "i2v": _probe_model_tree(MODEL_I2V_LOCAL),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }

    # ---- load only ----
    if ping == "wan_load":
        which = str(inp.get("which") or "i2v").strip().lower()
        if which not in ("t2v", "i2v"):
            which = "i2v"
        t0 = time.time()
        if which == "i2v":
            _load_i2v(unload_other=True)
        else:
            _load_t2v(unload_other=True)
        return {"ok": True, "msg": f"WAN_LOAD_OK:{which}", "elapsed_s": round(time.time() - t0, 3), "gpu_info": _gpu_info()}

    # ---- GENERATION ----
    try:
        if ping == "t2v_generate":
            out = _t2v_generate(inp)
            return out

        if ping == "i2v_generate":
            out = _i2v_generate(inp)
            return out

        return {"ok": False, "error": f"UNKNOWN_PING: {ping or '<empty>'}", "input_keys": list(inp.keys())}

    except Exception as e:
        # Important: liberar VRAM si algo truena
        _hard_cleanup()
        return {
            "ok": False,
            "error": str(e),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }
    finally:
        # cleanup suave post-job (opcional)
        _cuda_cleanup()

runpod.serverless.start({"handler": handler})