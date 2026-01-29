# worker.py — IsabelaOS Video Worker (WAN)
# ============================================================
# OBJETIVO:
# - MISMO resultado que en POD (mismos defaults: 576x1024, fps 24, seconds 3)
# - SIN OOM aleatorios en serverless
# - Limpieza REAL de VRAM (evita fragmentación)
# - NO inventa parámetros de calidad
#
# ✅ FIX CRÍTICO APLICADO:
# - VAE en fp16 (ANTES estaba en float32 y causaba OOM/fragmentación)
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

# 🔥 CRÍTICO: reduce fragmentación VRAM en serverless
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
)

# ------------------------------------------------------------
# HF cached_download compatibility
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

# ------------------------------------------------------------
# RMSNorm fallback
# ------------------------------------------------------------
if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    nn.RMSNorm = RMSNorm

# ------------------------------------------------------------
# scaled_dot_product_attention patch
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
    p = str(p).strip()
    if p.startswith("workspace/"):
        p = "/" + p
    return p.replace("//", "/")

DEFAULT_T2V_PATH = "/runpod-volume/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/runpod-volume/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# cache
_pipe_t2v = None
_pipe_i2v = None

# ------------------------------------------------------------
# 🔥 VRAM CLEANUP REAL
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

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    _cuda_cleanup(sync=True)

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

    _hard_cleanup()

# ------------------------------------------------------------
# Load WAN pipelines
# ------------------------------------------------------------
def _lazy_import_wan():
    from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
    return WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan

def _pipe_memory_tweaks(pipe):
    # No cambia output, solo memoria/estabilidad
    try: pipe.enable_attention_slicing("max")
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    return pipe

def _assert_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise RuntimeError(f"{label} model path not found: {path}")

def _load_t2v():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    _unload_pipes(keep="t2v")
    WanPipeline, _, AutoencoderKLWan = _lazy_import_wan()

    _assert_dir(MODEL_T2V_LOCAL, "T2V")

    # ✅ FIX CRÍTICO: VAE en fp16 (NO float32)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_T2V_LOCAL,
        subfolder="vae",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
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
            _hard_cleanup()
            pipe = pipe.to("cuda")

    _pipe_t2v = _pipe_memory_tweaks(pipe)
    return _pipe_t2v

def _load_i2v():
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    _unload_pipes(keep="i2v")
    _, WanImageToVideoPipeline, AutoencoderKLWan = _lazy_import_wan()

    _assert_dir(MODEL_I2V_LOCAL, "I2V")

    # ✅ FIX CRÍTICO: VAE en fp16 (NO float32)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_I2V_LOCAL,
        subfolder="vae",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
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
            _hard_cleanup()
            pipe = pipe.to("cuda")

    _pipe_i2v = _pipe_memory_tweaks(pipe)
    return _pipe_i2v

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _snap16(x: int) -> int:
    return max(16, int(round(int(x) / 16) * 16))

# ✅ Defaults EXACTOS como tu POD screenshot
DEFAULT_W, DEFAULT_H = 576, 1024

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

def _frames_to_mp4(frames, fps: int) -> bytes:
    import imageio.v2 as imageio
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio.get_writer(f.name, fps=fps, codec="libx264", quality=8)
        try:
            for fr in frames:
                writer.append_data(fr)
        finally:
            writer.close()
        f.seek(0)
        return f.read()

def _extract_frames(out):
    # diffusers WAN suele tener .frames
    if hasattr(out, "frames"):
        return out.frames
    # fallback dict
    if isinstance(out, dict):
        if "frames" in out:
            return out["frames"]
        if "video" in out:
            return out["video"]
    raise RuntimeError("No pude extraer frames del resultado (out.frames no existe).")

# ------------------------------------------------------------
# GENERATORS (mismos defaults, sin inventar)
# ------------------------------------------------------------
def _t2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_t2v()

    prompt = str(inp.get("prompt", "")).strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    negative = str(inp.get("negative_prompt", "")).strip()
    negative = negative if negative else None

    fps = int(inp.get("fps", 24))
    seconds = int(inp.get("duration_s", inp.get("seconds", 3)))
    num_frames = int(inp.get("num_frames", fps * seconds))

    width = _snap16(DEFAULT_W)
    height = _snap16(DEFAULT_H)

    steps = int(inp.get("steps", 25))
    guidance_scale = float(inp.get("guidance_scale", 7.5))

    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(out)
    mp4 = _frames_to_mp4(frames, fps=fps)

    # Limpieza ligera post-job (no borra pipes por defecto)
    _cuda_cleanup(sync=False)

    return {
        "ok": True,
        "mode": "t2v",
        "width": width,
        "height": height,
        "fps": fps,
        "seconds": seconds,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4).decode("utf-8"),
        "video_mime": "video/mp4",
    }

def _i2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _load_i2v()

    prompt = str(inp.get("prompt", "")).strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    img_b64 = inp.get("image_b64") or inp.get("image") or inp.get("init_image_b64")
    if not img_b64:
        raise RuntimeError("Falta image_b64")

    from PIL import Image
    raw = _decode_b64(str(img_b64))
    img = Image.open(BytesIO(raw)).convert("RGB")

    fps = int(inp.get("fps", 24))
    seconds = int(inp.get("duration_s", inp.get("seconds", 3)))
    num_frames = int(inp.get("num_frames", fps * seconds))

    width = _snap16(DEFAULT_W)
    height = _snap16(DEFAULT_H)

    steps = int(inp.get("steps", 25))
    guidance_scale = float(inp.get("guidance_scale", 7.5))

    # Resize igual a tu default
    try:
        img = img.resize((width, height))
    except Exception:
        pass

    negative = str(inp.get("negative_prompt", "")).strip()
    negative = negative if negative else None

    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            image=img,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

    frames = _extract_frames(out)
    mp4 = _frames_to_mp4(frames, fps=fps)

    _cuda_cleanup(sync=False)

    return {
        "ok": True,
        "mode": "i2v",
        "width": width,
        "height": height,
        "fps": fps,
        "seconds": seconds,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4).decode("utf-8"),
        "video_mime": "video/mp4",
    }

# ------------------------------------------------------------
# RunPod handler
# ------------------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input", {}) or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode == "t2v":
            return _t2v_generate(inp)
        if mode == "i2v":
            return _i2v_generate(inp)

        return {"ok": False, "error": "Modo inválido (usa mode='t2v' o mode='i2v')."}

    except Exception as e:
        # Limpieza fuerte si falla (para que el siguiente job no herede basura)
        _unload_pipes(keep=None)
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

# RunPod entrypoint
runpod.serverless.start({"handler": handler})