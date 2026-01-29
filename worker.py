# worker.py — IsabelaOS Video Worker (WAN)
# ============================================================
# OBJETIVO:
# - MISMO resultado que en POD
# - SIN OOM aleatorios en serverless
# - Limpieza REAL de VRAM
# - NO cambia calidad / NO inventa parámetros
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

# 🔥 CRÍTICO: evita fragmentación de VRAM en serverless
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
    p = p.strip()
    if p.startswith("workspace/"):
        p = "/" + p
    return p.replace("//", "/")

DEFAULT_T2V_PATH = "/runpod-volume/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/runpod-volume/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

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

def _unload_pipes(keep=None):
    global _pipe_t2v, _pipe_i2v

    if keep != "t2v" and _pipe_t2v is not None:
        _safe_pipe_to_cpu(_pipe_t2v)
        del _pipe_t2v
        _pipe_t2v = None

    if keep != "i2v" and _pipe_i2v is not None:
        _safe_pipe_to_cpu(_pipe_i2v)
        del _pipe_i2v
        _pipe_i2v = None

    _hard_cleanup()

# ------------------------------------------------------------
# Load WAN pipelines
# ------------------------------------------------------------
def _lazy_import_wan():
    from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
    return WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan

def _pipe_memory_tweaks(pipe):
    try: pipe.enable_attention_slicing("max")
    except: pass
    try: pipe.enable_vae_slicing()
    except: pass
    try: pipe.enable_vae_tiling()
    except: pass
    return pipe

def _load_t2v():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    _unload_pipes(keep="t2v")
    WanPipeline, _, AutoencoderKLWan = _lazy_import_wan()

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
def _snap16(x):
    return max(16, int(round(x / 16) * 16))

DEFAULT_W, DEFAULT_H = 576, 1024

def _frames_to_mp4(frames, fps):
    import imageio.v2 as imageio
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio.get_writer(f.name, fps=fps, codec="libx264", quality=8)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
        f.seek(0)
        return f.read()

# ------------------------------------------------------------
# GENERATORS
# ------------------------------------------------------------
def _t2v_generate(inp):
    pipe = _load_t2v()

    prompt = inp.get("prompt", "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    negative = inp.get("negative_prompt", "").strip() or None

    fps = int(inp.get("fps", 24))
    seconds = int(inp.get("duration_s", 3))
    num_frames = fps * seconds

    width = _snap16(DEFAULT_W)
    height = _snap16(DEFAULT_H)

    steps = int(inp.get("steps", 16))
    guidance_scale = float(inp.get("guidance_scale", 5.0))

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

    frames = out.frames
    mp4 = _frames_to_mp4(frames, fps)

    _cuda_cleanup(sync=False)

    return {
        "ok": True,
        "video_b64": base64.b64encode(mp4).decode(),
        "video_mime": "video/mp4",
    }

def _i2v_generate(inp):
    pipe = _load_i2v()

    prompt = inp.get("prompt", "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    img_b64 = inp.get("image_b64")
    if not img_b64:
        raise RuntimeError("Falta image_b64")

    from PIL import Image
    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    img = img.resize((_snap16(DEFAULT_W), _snap16(DEFAULT_H)))

    fps = int(inp.get("fps", 24))
    seconds = int(inp.get("duration_s", 3))
    num_frames = fps * seconds

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            image=img,
            num_frames=num_frames,
        )

    frames = out.frames
    mp4 = _frames_to_mp4(frames, fps)

    _cuda_cleanup(sync=False)

    return {
        "ok": True,
        "video_b64": base64.b64encode(mp4).decode(),
        "video_mime": "video/mp4",
    }

# ------------------------------------------------------------
# RunPod handler
# ------------------------------------------------------------
def handler(job):
    try:
        inp = job.get("input", {})
        mode = inp.get("mode")

        if mode == "t2v":
            return _t2v_generate(inp)
        if mode == "i2v":
            return _i2v_generate(inp)

        return {"ok": False, "error": "Modo inválido"}
    except Exception as e:
        _hard_cleanup()
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }