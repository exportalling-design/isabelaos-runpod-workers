# worker.py — IsabelaOS Video Worker (WAN 2.2) [SERVERLESS FINAL]
# ============================================================
# OBJETIVO:
# - MISMO resultado "POD" (WAN 2.2)
# - SIN errores aleatorios (OOM / mismatch frames)
# - Limpieza REAL de VRAM entre requests
# - NO inventa parámetros raros
# ============================================================

import os
import time
import gc
import base64
import binascii
import traceback
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

# ------------------------------------------------------------
# ENV hardening (ANTES de importar torch)
# ------------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 CRÍTICO: reduce fragmentación en serverless
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
)

# ------------------------------------------------------------
# HF cached_download compatibility (huggingface_hub)
# ------------------------------------------------------------
import huggingface_hub as h
if not hasattr(h, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download
    def _cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)
    h.cached_download = _cached_download

import torch
import torch.nn as nn
import torch.nn.functional as F

# Buenas prácticas (no rompe nada)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ------------------------------------------------------------
# RMSNorm fallback (COMPAT con elementwise_affine)
# ------------------------------------------------------------
if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__()
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

# ------------------------------------------------------------
# scaled_dot_product_attention patch (enable_gqa)
# ------------------------------------------------------------
if hasattr(F, "scaled_dot_product_attention"):
    _orig_sdp = F.scaled_dot_product_attention
    def patched_sdp_attention(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdp(*args, **kwargs)
    F.scaled_dot_product_attention = patched_sdp_attention

import runpod

# ------------------------------------------------------------
# Paths / Config
# ------------------------------------------------------------
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
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # WAN estable en fp16

_pipe_t2v = None
_pipe_i2v = None

# ------------------------------------------------------------
# VRAM CLEANUP REAL
# ------------------------------------------------------------
def _cuda_cleanup(sync=True):
    if not torch.cuda.is_available():
        return
    try:
        if sync:
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

def _hard_cleanup(sync=True):
    try:
        gc.collect()
    except Exception:
        pass
    _cuda_cleanup(sync=sync)

def _safe_pipe_to_cpu(pipe):
    try:
        pipe.to("cpu")
    except Exception:
        pass

def _unload_pipes(keep: Optional[str] = None):
    global _pipe_t2v, _pipe_i2v

    if keep != "t2v" and _pipe_t2v is not None:
        _safe_pipe_to_cpu(_pipe_t2v)
        try:
            del _pipe_t2v
        except Exception:
            pass
        _pipe_t2v = None

    if keep != "i2v" and _pipe_i2v is not None:
        _safe_pipe_to_cpu(_pipe_i2v)
        try:
            del _pipe_i2v
        except Exception:
            pass
        _pipe_i2v = None

    _hard_cleanup(sync=True)

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
        raise RuntimeError(f"{label} model path not found: {path}")

def _pipe_memory_tweaks(pipe):
    # Ajustes seguros (no cambian “look”, solo ayudan memoria)
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

def _lazy_import_wan():
    # Importa solo cuando se necesita (más rápido en cold start)
    from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
    return WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan

# ------------------------------------------------------------
# Robust base64 decode
# ------------------------------------------------------------
def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("image_b64 vacío")
    s = str(s).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = s.replace("-", "+").replace("_", "/")
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

# ------------------------------------------------------------
# Frames -> MP4 bytes
# ------------------------------------------------------------
def _to_uint8_hwc(frame):
    import numpy as np
    if hasattr(frame, "convert"):  # PIL
        arr = np.array(frame.convert("RGB"), dtype=np.uint8)
        return arr
    if torch.is_tensor(frame):
        t = frame.detach().float().cpu().numpy()
        arr = t
    else:
        arr = np.asarray(frame)

    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]

    # (C,H,W)->(H,W,C)
    if arr.ndim == 3:
        c_first = arr.shape[0] in (1, 3, 4)
        c_last_ok = arr.shape[-1] in (1, 3, 4)
        if c_first and not c_last_ok:
            arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr

def _normalize_frames(frames):
    import numpy as np
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        while frames.ndim >= 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim == 4:
            return [_to_uint8_hwc(frames[i]) for i in range(frames.shape[0])]

    return [_to_uint8_hwc(f) for f in frames]

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

# ------------------------------------------------------------
# Timing + Dims (POD-like)
# ------------------------------------------------------------
def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(round(float(v)))
    except Exception:
        return default
    return max(lo, min(hi, n))

def _snap16(n: int) -> int:
    n = int(n)
    r = int(round(n / 16.0) * 16)
    return max(16, r)

# ✅ FIX REAL: WAN exige (num_frames - 1) % 4 == 0
def _fix_frames_for_wan(n: int) -> int:
    """
    WAN: (num_frames - 1) MUST be divisible by 4.
    Evitamos que diffusers "redondee" adentro (eso causó mismatch).
    IMPORTANTE: NO redondear "nearest", siempre subir al siguiente válido.
    """
    n = int(n)
    if n < 5:
        return 5
    r = (n - 1) % 4
    if r == 0:
        return n
    return n + (4 - r)

# ✅ POD default landscape
POD_W, POD_H = 1280, 720

# ✅ Reels 9:16 (rápido)
REELS_W, REELS_H = 576, 1024

def _pick_dims(inp: Dict[str, Any]) -> Tuple[int, int]:
    ar = str(inp.get("aspect_ratio") or "").strip()
    if ar == "9:16":
        return REELS_W, REELS_H
    # default = POD-like
    return POD_W, POD_H

def _normalize_timing(inp: Dict[str, Any]) -> Tuple[int, int, int]:
    fps = _clamp_int(inp.get("fps", 24), 8, 30, 24)

    seconds_raw = inp.get("duration_s", inp.get("seconds", 3))
    seconds = _clamp_int(seconds_raw, 1, 10, 3)

    # ✅ REGLA: si hay duration_s, calculamos frames desde seconds*fps
    # (ignoramos num_frames del cliente para evitar inconsistencias)
    raw_frames = int(seconds * fps)

    # pero si NO mandan duration, sí podemos tomar num_frames
    if "duration_s" not in inp and "seconds" not in inp:
        raw_frames = inp.get("num_frames", inp.get("frames", raw_frames))
        try:
            raw_frames = int(raw_frames)
        except Exception:
            raw_frames = int(seconds * fps)

    num_frames = _fix_frames_for_wan(raw_frames)
    return int(seconds), int(fps), int(num_frames)

# ------------------------------------------------------------
# Load pipelines (con cleanup real)
# ------------------------------------------------------------
def _load_t2v(unload_other=True):
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    if unload_other:
        _unload_pipes(keep="t2v")

    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")
    WanPipeline, _, AutoencoderKLWan = _lazy_import_wan()

    t0 = time.time()
    print(f"[WAN_LOAD] Loading T2V LOCAL from: {MODEL_T2V_LOCAL}")
    print(f"[WAN_LOAD] dtype={DTYPE} device={DEVICE}")

    # VAE float32 estable, resto fp16
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
        try:
            pipe = pipe.to("cuda")
        except torch.cuda.OutOfMemoryError:
            print("[WAN_LOAD] OOM on pipe.to(cuda) -> hard cleanup and retry")
            _hard_cleanup(sync=True)
            pipe = pipe.to("cuda")

    pipe = _pipe_memory_tweaks(pipe)
    _pipe_t2v = pipe
    print(f"[WAN_LOAD] T2V loaded OK in {time.time() - t0:.2f}s")
    return _pipe_t2v

def _load_i2v(unload_other=True):
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    if unload_other:
        _unload_pipes(keep="i2v")

    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")
    _, WanImageToVideoPipeline, AutoencoderKLWan = _lazy_import_wan()

    t0 = time.time()
    print(f"[WAN_LOAD] Loading I2V LOCAL from: {MODEL_I2V_LOCAL}")
    print(f"[WAN_LOAD] dtype={DTYPE} device={DEVICE}")

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
        try:
            pipe = pipe.to("cuda")
        except torch.cuda.OutOfMemoryError:
            print("[WAN_LOAD] OOM on pipe.to(cuda) -> hard cleanup and retry")
            _hard_cleanup(sync=True)
            pipe = pipe.to("cuda")

    pipe = _pipe_memory_tweaks(pipe)
    _pipe_i2v = pipe
    print(f"[WAN_LOAD] I2V loaded OK in {time.time() - t0:.2f}s")
    return _pipe_i2v

# ------------------------------------------------------------
# Generators
# ------------------------------------------------------------
def _t2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_t2v(unload_other=True)

    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    negative = str(inp.get("negative_prompt") or "").strip()
    negative = negative if negative else None

    seconds, fps, num_frames = _normalize_timing(inp)
    raw_frames = int(seconds * fps)

    w_raw, h_raw = _pick_dims(inp)
    width, height = _snap16(w_raw), _snap16(h_raw)

    # Respetar valores POD si vienen:
    steps = _clamp_int(inp.get("steps", 34), 1, 80, 34)
    guidance_scale = float(inp.get("guidance_scale", 6.5) or 6.5)

    print(f"[T2V] w={width} h={height} fps={fps} raw_frames={raw_frames} FIXED_frames={num_frames} steps={steps} cfg={guidance_scale}")

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,              # ✅ YA arreglado al formato WAN
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(result)
    mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
    mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    # Limpieza ligera entre jobs (sin matar pipeline)
    _cuda_cleanup(sync=False)

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
        "video_b64": mp4_b64,
        "video_mime": "video/mp4",
        **_diffusers_info(),
        "gpu_info": _gpu_info(),
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
    negative = negative if negative else None

    seconds, fps, num_frames = _normalize_timing(inp)
    raw_frames = int(seconds * fps)

    w_raw, h_raw = _pick_dims(inp)
    width, height = _snap16(w_raw), _snap16(h_raw)

    # Resize init image al tamaño estable elegido
    try:
        init_img = init_img.resize((width, height))
    except Exception:
        pass

    steps = _clamp_int(inp.get("steps", 34), 1, 80, 34)
    guidance_scale = float(inp.get("guidance_scale", 6.5) or 6.5)

    print(f"[I2V] w={width} h={height} fps={fps} raw_frames={raw_frames} FIXED_frames={num_frames} steps={steps} cfg={guidance_scale}")

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=init_img,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,              # ✅ YA arreglado al formato WAN
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(result)
    mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
    mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    _cuda_cleanup(sync=False)

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
        **_diffusers_info(),
        "gpu_info": _gpu_info(),
    }

# ------------------------------------------------------------
# Debug helpers (no los quito)
# ------------------------------------------------------------
def _list_dir_safe(path: str, limit: int = 200):
    try:
        items = sorted(os.listdir(path))
        if len(items) > limit:
            return items[:limit] + [f"...(+{len(items)-limit} more)"]
        return items
    except Exception as e:
        return [f"<cannot list: {e}>"]

# ------------------------------------------------------------
# RunPod handler
# ------------------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}

        ping = str(inp.get("ping") or "").strip().lower()
        mode = str(inp.get("mode") or "").strip().lower()

        # Compat: mode=t2v/i2v
        if not ping and mode:
            if mode == "t2v":
                ping = "t2v_generate"
            elif mode == "i2v":
                ping = "i2v_generate"

        # Debug endpoints
        if ping in ("echo", "debug"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "input": inp,
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
                "env": {
                    "WAN_T2V_PATH": os.environ.get("WAN_T2V_PATH"),
                    "WAN_I2V_PATH": os.environ.get("WAN_I2V_PATH"),
                },
                "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                "sizes": {
                    "pod_default": {"w": POD_W, "h": POD_H},
                    "reels_9_16": {"w": REELS_W, "h": REELS_H},
                },
                "notes": "Frames se ajustan a WAN: (num_frames-1)%4==0 (siempre sube al siguiente válido).",
            }

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
                MODEL_T2V_LOCAL,
                MODEL_I2V_LOCAL,
            ]
            return {
                "ok": True,
                "candidates": {p: _list_dir_safe(p, limit=120) for p in candidates},
                "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                **_diffusers_info(),
                "gpu_info": _gpu_info(),
            }

        # Generación
        if ping == "t2v_generate":
            return _t2v_generate(inp)
        if ping == "i2v_generate":
            return _i2v_generate(inp)

        return {"ok": False, "error": "Ping/Modo inválido", "gpu_info": _gpu_info(), **_diffusers_info()}

    except Exception as e:
        # 🔥 Limpieza fuerte si algo falla
        _hard_cleanup(sync=True)
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }

runpod.serverless.start({"handler": handler})