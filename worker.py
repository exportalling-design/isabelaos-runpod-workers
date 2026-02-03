# worker.py (RunPod Serverless) — IsabelaOS Video Worker (WAN)
# ✅ FIX: anti-OOM load + cleanup BEFORE load + retry load + WAN frames + cold per job

import os
import time
import gc
import base64
import binascii
import traceback
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

# --- ENV hardening ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ NO expandable_segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.8"

# ✅ serverless: por defecto COLD por job
WAN_COLD_EACH_JOB = os.environ.get("WAN_COLD_EACH_JOB", "1").strip() not in ("0", "false", "False")

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

# --- RMSNorm fallback ---
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
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_pipe_t2v = None
_pipe_i2v = None
_last_sig_t2v: Optional[Tuple[int, int, int]] = None
_last_sig_i2v: Optional[Tuple[int, int, int]] = None

# ---------------------------
# Utils
# ---------------------------
def _cuda_cleanup():
    if torch.cuda.is_available():
        try:
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
            info["mem_alloc_mb"] = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
            info["mem_reserved_mb"] = int(torch.cuda.memory_reserved(0) / (1024 * 1024))
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
        raise RuntimeError(f"{label} model path not found: {path}. (serverless suele ser /runpod-volume/...)")

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
        arr = np.array(frame.convert("RGB"), dtype=np.uint8)
        return arr
    if torch.is_tensor(frame):
        t = frame.detach().float().cpu()
        arr = t.numpy()
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

# ---------------------------
# Timing + Dims helpers
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

def _fix_frames_for_wan(num_frames: int) -> int:
    num_frames = int(num_frames)
    if num_frames < 5:
        return 5
    r = (num_frames - 1) % 4
    if r == 0:
        return num_frames
    return num_frames + (4 - r)

DEFAULT_W, DEFAULT_H = 576, 512
REELS_W, REELS_H = 576, 1024

def _pick_dims_simple(inp: Dict[str, Any]) -> Tuple[int, int]:
    ar = str(inp.get("aspect_ratio") or "").strip()
    if ar == "9:16":
        return REELS_W, REELS_H
    return DEFAULT_W, DEFAULT_H

def _normalize_timing(inp: Dict[str, Any]) -> Tuple[int, int, int]:
    seconds_raw = inp.get("duration_s", None)
    if seconds_raw is None:
        seconds_raw = inp.get("seconds", None)
    if seconds_raw is None:
        seconds_raw = 3

    seconds = _clamp_int(seconds_raw, 3, 5, 3)
    seconds = 3 if seconds < 4 else 5

    fps = _clamp_int(inp.get("fps", 12), 8, 30, 12)

    num_frames = seconds * fps
    num_frames = _fix_frames_for_wan(num_frames)
    return seconds, fps, num_frames

# ---------------------------
# Pipelines
# ---------------------------
def _unload_pipes():
    global _pipe_t2v, _pipe_i2v, _last_sig_t2v, _last_sig_i2v
    if _pipe_t2v is not None:
        try:
            del _pipe_t2v
        except Exception:
            pass
    if _pipe_i2v is not None:
        try:
            del _pipe_i2v
        except Exception:
            pass
    _pipe_t2v = None
    _pipe_i2v = None
    _last_sig_t2v = None
    _last_sig_i2v = None
    _hard_cleanup()

def _ensure_signature(which: str, width: int, height: int, num_frames: int):
    global _last_sig_t2v, _last_sig_i2v
    sig = (int(width), int(height), int(num_frames))

    if which == "t2v":
        if _last_sig_t2v is None:
            _last_sig_t2v = sig
            return
        if sig != _last_sig_t2v:
            print(f"[SIG] T2V changed {_last_sig_t2v} -> {sig} | forcing unload")
            _unload_pipes()
            _last_sig_t2v = sig
            return

    if which == "i2v":
        if _last_sig_i2v is None:
            _last_sig_i2v = sig
            return
        if sig != _last_sig_i2v:
            print(f"[SIG] I2V changed {_last_sig_i2v} -> {sig} | forcing unload")
            _unload_pipes()
            _last_sig_i2v = sig
            return

def _load_t2v_once():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")
    WanPipeline, AutoencoderKLWan, _, err = _lazy_import_wan()
    if err:
        raise RuntimeError(f"WAN_DIFFUSERS_IMPORT_FAILED (T2V): {err}")

    t0 = time.time()
    print(f"[WAN_LOAD] T2V from: {MODEL_T2V_LOCAL} dtype={DTYPE} device={DEVICE}")
    print("[WAN_LOAD] gpu before load:", _gpu_info())

    # ✅ IMPORTANT: limpiar antes de cargar (evita OOM por residuos de warm container)
    _hard_cleanup()

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_T2V_LOCAL, subfolder="vae",
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

    # ✅ mover a cuda solo al final
    if DEVICE == "cuda":
        pipe = _pipe_memory_tweaks(pipe)
        pipe = pipe.to("cuda")

    _pipe_t2v = _pipe_memory_tweaks(pipe)
    print(f"[WAN_LOAD] T2V loaded in {time.time()-t0:.2f}s")
    print("[WAN_LOAD] gpu after load:", _gpu_info())
    return _pipe_t2v

def _load_t2v():
    # ✅ retry inteligente si OOM en .to("cuda")
    try:
        return _load_t2v_once()
    except torch.cuda.OutOfMemoryError as e:
        print("[WAN_LOAD] OOM on load. Forcing unload + cleanup + retry once.")
        _unload_pipes()
        _hard_cleanup()
        return _load_t2v_once()

def _load_i2v_once():
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")
    _, AutoencoderKLWan, WanImageToVideoPipeline, err = _lazy_import_wan()
    if err:
        raise RuntimeError(f"WAN_DIFFUSERS_IMPORT_FAILED (I2V): {err}")

    t0 = time.time()
    print(f"[WAN_LOAD] I2V from: {MODEL_I2V_LOCAL} dtype={DTYPE} device={DEVICE}")
    print("[WAN_LOAD] gpu before load:", _gpu_info())

    _hard_cleanup()

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_I2V_LOCAL, subfolder="vae",
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
        pipe = _pipe_memory_tweaks(pipe)
        pipe = pipe.to("cuda")

    _pipe_i2v = _pipe_memory_tweaks(pipe)
    print(f"[WAN_LOAD] I2V loaded in {time.time()-t0:.2f}s")
    print("[WAN_LOAD] gpu after load:", _gpu_info())
    return _pipe_i2v

def _load_i2v():
    try:
        return _load_i2v_once()
    except torch.cuda.OutOfMemoryError:
        print("[WAN_LOAD] OOM on load. Forcing unload + cleanup + retry once.")
        _unload_pipes()
        _hard_cleanup()
        return _load_i2v_once()

# ---------------------------
# Generators
# ---------------------------
def _t2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    negative = str(inp.get("negative_prompt") or "").strip()

    seconds, fps, num_frames = _normalize_timing(inp)
    width_raw, height_raw = _pick_dims_simple(inp)
    width = _snap16(width_raw)
    height = _snap16(height_raw)

    _ensure_signature("t2v", width, height, num_frames)

    steps = _clamp_int(inp.get("steps", 16), 1, 80, 16)
    guidance_scale = float(inp.get("guidance_scale", 5.0) or 5.0)

    t0 = time.time()
    print(f"[T2V] w={width} h={height} frames={num_frames} fps={fps} steps={steps}")
    print("[T2V] gpu pre:", _gpu_info())

    result = None
    frames = None
    mp4_bytes = None
    mp4_b64 = None

    try:
        pipe = _load_t2v()
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
            "video_b64": mp4_b64,
            "video_mime": "video/mp4",
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }

    finally:
        if WAN_COLD_EACH_JOB:
            _unload_pipes()
        else:
            try:
                del result, frames, mp4_bytes, mp4_b64
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            _cuda_cleanup()

def _i2v_generate(inp: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Falta prompt")

    image_b64 = inp.get("image_b64") or inp.get("image") or inp.get("init_image_b64")
    if not image_b64:
        raise RuntimeError("Falta image_b64")

    init_img = _b64_to_pil_image(str(image_b64))
    negative = str(inp.get("negative_prompt") or "").strip()

    seconds, fps, num_frames = _normalize_timing(inp)
    width_raw, height_raw = _pick_dims_simple(inp)
    width = _snap16(width_raw)
    height = _snap16(height_raw)

    try:
        init_img = init_img.resize((width, height))
    except Exception:
        pass

    _ensure_signature("i2v", width, height, num_frames)

    steps = _clamp_int(inp.get("steps", 16), 1, 80, 16)
    guidance_scale = float(inp.get("guidance_scale", 5.0) or 5.0)

    t0 = time.time()
    print(f"[I2V] w={width} h={height} frames={num_frames} fps={fps} steps={steps}")
    print("[I2V] gpu pre:", _gpu_info())

    result = None
    frames = None
    mp4_bytes = None
    mp4_b64 = None

    try:
        pipe = _load_i2v()
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
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }

    finally:
        if WAN_COLD_EACH_JOB:
            _unload_pipes()
        else:
            try:
                del result, frames, mp4_bytes, mp4_b64, init_img
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            _cuda_cleanup()

# ---------------------------
# Handler (ANTI-OOM desde el inicio)
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        ping = str(inp.get("ping") or "").strip().lower()
        mode = str(inp.get("mode") or "").strip().lower()

        # ✅ CRÍTICO: limpiar SIEMPRE antes de cargar/generar (serverless warm containers)
        if ping not in ("echo", "debug", "smoke", "list_paths"):
            # si estamos en cold mode, descargamos todo de una
            if WAN_COLD_EACH_JOB:
                _unload_pipes()
            else:
                _hard_cleanup()

        if not ping and mode:
            if mode == "t2v":
                ping = "t2v_generate"
            elif mode == "i2v":
                ping = "i2v_generate"

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
                    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    "WAN_COLD_EACH_JOB": str(WAN_COLD_EACH_JOB),
                },
                "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                "sizes": {"default": {"w": DEFAULT_W, "h": DEFAULT_H}, "reels_9_16": {"w": REELS_W, "h": REELS_H}},
            }

        if ping == "smoke":
            return {"ok": True, "msg": "SMOKE_OK", "gpu_info": _gpu_info(), **_diffusers_info()}

        if ping in ("t2v_generate", "t2v"):
            return _t2v_generate(inp)

        if ping in ("i2v_generate", "i2v"):
            return _i2v_generate(inp)

        if ping == "list_paths":
            candidates = [
                "/",
                "/workspace",
                "/runpod-volume",
                "/runpod-volume/models",
                "/runpod-volume/models/wan22",
                MODEL_T2V_LOCAL,
                MODEL_I2V_LOCAL,
            ]
            return {
                "ok": True,
                "candidates": candidates,
                "listing": {p: _list_dir_safe(p) for p in candidates},
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
            }

        return {"ok": False, "error": f"Unknown ping/mode ping='{ping}' mode='{mode}'", "gpu_info": _gpu_info(), **_diffusers_info()}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
        }

runpod.serverless.start({"handler": handler})
