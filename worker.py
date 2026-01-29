# worker.py — IsabelaOS Video Worker (WAN 2.2) — SERVERLESS SAFE
# ============================================================
# FIXES:
# ✅ RMSNorm patch compatible con elementwise_affine
# ✅ Anti-fragmentación VRAM
# ✅ NO REUSAR PIPE ENTRE JOBS (evita mismatch 16720 vs 17480 y OOM sucio)
# ✅ Defaults = POD: 576x1024, 24fps, 72 frames, steps 25, cfg 7.5
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

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
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
# RMSNorm fallback (WAN pasa elementwise_affine=True)
# ------------------------------------------------------------
import inspect


def _needs_rmsnorm_patch() -> bool:
    if not hasattr(nn, "RMSNorm"):
        return True
    try:
        sig = inspect.signature(nn.RMSNorm.__init__)
        return "elementwise_affine" not in sig.parameters
    except Exception:
        return True


if _needs_rmsnorm_patch():

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True, **kwargs):
            super().__init__()
            self.eps = eps
            self.dim = dim
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                x = x * self.weight
            return x

    nn.RMSNorm = RMSNorm

# ------------------------------------------------------------
# scaled_dot_product_attention patch (remove enable_gqa kwarg)
# ------------------------------------------------------------
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
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------
# Cleanup helpers (VRAM REAL)
# ---------------------------
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
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    _cuda_cleanup(sync=True)


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


def _lazy_import_wan():
    # Import aquí para que el RMSNorm patch ya exista
    from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan

    return WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan


def _pipe_memory_tweaks(pipe):
    # reduce VRAM, no cambia “calidad artística”
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


# ---------------------------
# Base64 image decode (robusto)
# ---------------------------
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


# ---------------------------
# Frames -> MP4 bytes
# ---------------------------
def _to_uint8_hwc(frame):
    import numpy as np

    if hasattr(frame, "convert"):
        return np.array(frame.convert("RGB"), dtype=np.uint8)

    if torch.is_tensor(frame):
        arr = frame.detach().float().cpu().numpy()
    else:
        arr = np.asarray(frame)

    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]

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
        raise RuntimeError(f"Could not extract frames from result type={type(result)}")


# ---------------------------
# Timing + Dims (MISMO POD)
# ---------------------------
def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(round(float(v)))
    except Exception:
        return default
    return max(lo, min(hi, n))


def _snap16(n: int) -> int:
    r = int(round(int(n) / 16.0) * 16)
    return max(16, r)


# POD defaults
POD_W, POD_H = 576, 1024
POD_FPS = 24
POD_SECONDS = 3
POD_FRAMES = 72
POD_STEPS = 25
POD_CFG = 7.5


def _pick_dims(inp: Dict[str, Any]) -> Tuple[int, int]:
    w = inp.get("width", None)
    h = inp.get("height", None)
    if w is not None and h is not None:
        return _snap16(int(w)), _snap16(int(h))
    return _snap16(POD_W), _snap16(POD_H)


def _normalize_timing(inp: Dict[str, Any]) -> Tuple[int, int, int]:
    fps = _clamp_int(inp.get("fps", POD_FPS), 8, 30, POD_FPS)

    if inp.get("num_frames") is not None:
        num_frames = _clamp_int(inp.get("num_frames", POD_FRAMES), 1, 300, POD_FRAMES)
        seconds = max(1, int(round(num_frames / max(1, fps))))
        return seconds, fps, num_frames

    seconds = _clamp_int(inp.get("duration_s", inp.get("seconds", POD_SECONDS)), 1, 10, POD_SECONDS)
    num_frames = int(round(seconds * fps))
    return seconds, fps, num_frames


# ---------------------------
# LOAD PIPE PER JOB (NO CACHE)
# ---------------------------
def _load_t2v_fresh():
    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")
    WanPipeline, _, AutoencoderKLWan = _lazy_import_wan()

    print(f"[WAN_LOAD] T2V path: {MODEL_T2V_LOCAL}")
    print(f"[WAN_LOAD] dtype={DTYPE} device={DEVICE}")

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

    pipe = _pipe_memory_tweaks(pipe)

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    return pipe


def _load_i2v_fresh():
    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")
    _, WanImageToVideoPipeline, AutoencoderKLWan = _lazy_import_wan()

    print(f"[WAN_LOAD] I2V path: {MODEL_I2V_LOCAL}")
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

    pipe = _pipe_memory_tweaks(pipe)

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    return pipe


def _destroy_pipe(pipe):
    try:
        pipe.to("cpu")
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    _hard_cleanup()


# ---------------------------
# Generators
# ---------------------------
def _t2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = None
    try:
        # Limpieza antes de cargar (serverless warm)
        _hard_cleanup()

        pipe = _load_t2v_fresh()

        prompt = str(inp.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Falta prompt")

        negative = str(inp.get("negative_prompt") or "").strip() or None

        seconds, fps, num_frames = _normalize_timing(inp)
        width, height = _pick_dims(inp)

        steps = _clamp_int(inp.get("steps", POD_STEPS), 1, 80, POD_STEPS)
        guidance_scale = float(inp.get("guidance_scale", POD_CFG) or POD_CFG)

        t0 = time.time()
        print(f"[T2V] w={width} h={height} fps={fps} frames={num_frames} steps={steps} cfg={guidance_scale}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
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
            "video_b64": mp4_b64,
            "video_mime": "video/mp4",
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }
    finally:
        if pipe is not None:
            _destroy_pipe(pipe)


def _i2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    pipe = None
    try:
        _hard_cleanup()

        pipe = _load_i2v_fresh()

        prompt = str(inp.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Falta prompt")

        image_b64 = inp.get("image_b64") or inp.get("image") or inp.get("init_image_b64")
        if not image_b64:
            raise RuntimeError("Falta image_b64")

        init_img = _b64_to_pil_image(str(image_b64))

        negative = str(inp.get("negative_prompt") or "").strip() or None

        seconds, fps, num_frames = _normalize_timing(inp)
        width, height = _pick_dims(inp)

        steps = _clamp_int(inp.get("steps", POD_STEPS), 1, 80, POD_STEPS)
        guidance_scale = float(inp.get("guidance_scale", POD_CFG) or POD_CFG)

        try:
            init_img = init_img.resize((width, height))
        except Exception:
            pass

        t0 = time.time()
        print(f"[I2V] w={width} h={height} fps={fps} frames={num_frames} steps={steps} cfg={guidance_scale}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=init_img,
                negative_prompt=negative,
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
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }
    finally:
        if pipe is not None:
            _destroy_pipe(pipe)


# ---------------------------
# RunPod handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        ping = str(inp.get("ping") or "").strip().lower()
        if ping in ("echo", "debug"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "input": inp,
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
                "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                "pod_defaults": {"w": POD_W, "h": POD_H, "fps": POD_FPS, "frames": POD_FRAMES, "steps": POD_STEPS, "cfg": POD_CFG},
            }

        if mode == "t2v":
            return _t2v_generate(inp)
        if mode == "i2v":
            return _i2v_generate(inp)

        return {"ok": False, "error": "Modo inválido (usa mode='t2v' o mode='i2v').", "gpu_info": _gpu_info()}
    except Exception as e:
        _hard_cleanup()
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }


runpod.serverless.start({"handler": handler})