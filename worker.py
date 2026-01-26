import os
import io
import time
import base64
import uuid
import urllib.request
from typing import Any, Dict, Optional

# --- ENV hardening ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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

from diffusers import WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

import runpod


# ---------------------------
# Helpers / Paths
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

DEFAULT_T2V_PATH = "/workspace/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/workspace/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

DEFAULT_H = int(os.environ.get("DEFAULT_H", "512"))
DEFAULT_W = int(os.environ.get("DEFAULT_W", "896"))
DEFAULT_FRAMES = int(os.environ.get("DEFAULT_FRAMES", "49"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "24"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "6.0"))

MAX_H = int(os.environ.get("MAX_H", "512"))
MAX_W = int(os.environ.get("MAX_W", "896"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "49"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))

MAX_RETURN_MB = int(os.environ.get("MAX_RETURN_MB", "200"))

_pipe_t2v = None
_pipe_i2v = None


def _assert_model_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise RuntimeError(f"{label} model path not found: {path} (check volume mount)")

def _pipe_memory_tweaks(pipe):
    try: pipe.enable_attention_slicing("max")
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    return pipe


def _load_t2v():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")
    print(f"[T2V] Loading LOCAL model from {MODEL_T2V_LOCAL}")

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
    ).to(DEVICE)

    _pipe_t2v = _pipe_memory_tweaks(pipe)
    return _pipe_t2v


def _load_i2v():
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")
    print(f"[I2V] Loading LOCAL model from {MODEL_I2V_LOCAL}")

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
    ).to(DEVICE)

    _pipe_i2v = _pipe_memory_tweaks(pipe)
    return _pipe_i2v


def _decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _fetch_image_url(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as r:
        raw = r.read()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_frames(result):
    if hasattr(result, "frames"):
        return result.frames[0]
    if hasattr(result, "videos"):
        return result.videos[0]
    raise RuntimeError("No frames/videos in result")


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _too_big(nbytes: int) -> bool:
    return nbytes > (MAX_RETURN_MB * 1024 * 1024)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def _cuda_cleanup():
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache()
        except Exception: pass
        try: torch.cuda.ipc_collect()
        except Exception: pass


def _normalize_mode(m: Any) -> str:
    m = (str(m or "").strip().lower())
    if m in ("i2v", "img2video", "image2video", "img_to_video", "image_to_video"):
        return "i2v"
    if m in ("t2v", "txt2video", "text2video", "text_to_video", "prompt", "prompt2video"):
        return "t2v"
    return "t2v"


def _render_mp4_bytes(payload: Dict[str, Any]) -> bytes:
    mode = _normalize_mode(payload.get("mode") or os.getenv("WORKER_MODE", "t2v"))

    prompt = str(payload.get("prompt") or "")
    negative = str(payload.get("negative_prompt") or payload.get("negative") or "")

    h = _clamp_int(payload.get("height", DEFAULT_H), 128, MAX_H)
    w = _clamp_int(payload.get("width", DEFAULT_W), 128, MAX_W)
    frames = _clamp_int(payload.get("num_frames", payload.get("frames", DEFAULT_FRAMES)), 8, MAX_FRAMES)
    fps = _clamp_int(payload.get("fps", DEFAULT_FPS), 8, 30)
    steps = _clamp_int(payload.get("steps", DEFAULT_STEPS), 1, MAX_STEPS)

    try:
        guidance = float(payload.get("guidance_scale", payload.get("guidance", DEFAULT_GUIDANCE)))
    except Exception:
        guidance = DEFAULT_GUIDANCE
    guidance = max(1.0, min(8.0, guidance))

    if not prompt:
        raise RuntimeError("MISSING_PROMPT")

    try:
        if mode == "t2v":
            pipe = _load_t2v()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                height=h,
                width=w,
                num_frames=frames,
                guidance_scale=guidance,
                num_inference_steps=steps,
            )
        else:
            img_b64 = payload.get("image_base64") or payload.get("imageB64") or payload.get("image_b64")
            img_url = payload.get("image_url") or payload.get("imageUrl")

            image = None
            if img_b64:
                image = _decode_image(img_b64)
            elif img_url:
                image = _fetch_image_url(img_url)

            if image is None:
                raise RuntimeError("MISSING_IMAGE: i2v requires image_base64 or image_url")

            pipe = _load_i2v()
            image = image.resize((w, h))
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative,
                height=h,
                width=w,
                num_frames=frames,
                guidance_scale=guidance,
                num_inference_steps=steps,
            )

        frames_out = _extract_frames(result)
        out_local = f"/tmp/wan_{mode}_{uuid.uuid4().hex}.mp4"
        export_to_video(frames_out, out_local, fps=fps)

        mp4_bytes = _read_file_bytes(out_local)
        try:
            os.remove(out_local)
        except Exception:
            pass
        return mp4_bytes

    except torch.cuda.OutOfMemoryError as e:
        _cuda_cleanup()
        raise RuntimeError(f"CUDA OOM: {str(e)}. Lower frames/size/steps.")
    except Exception:
        _cuda_cleanup()
        raise


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "NO_CUDA"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA_UNKNOWN"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}

    mp4 = _render_mp4_bytes(inp)

    if _too_big(len(mp4)):
        return {"ok": False, "error": "MP4_TOO_LARGE", "detail": f"{len(mp4)} bytes > MAX_RETURN_MB"}

    return {
        "ok": True,
        "gpu": _gpu_name(),
        "video_b64": base64.b64encode(mp4).decode("utf-8"),
        "meta": {
            "t2v_path": MODEL_T2V_LOCAL,
            "i2v_path": MODEL_I2V_LOCAL,
            "device": DEVICE,
            "dtype": str(DTYPE)
        }
    }


runpod.serverless.start({"handler": handler})
