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

# --- ftfy optional (Wan/tokenizers sometimes expect it) ---
try:
    import ftfy  # noqa: F401
except Exception:
    ftfy = None

def _fix_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    try:
        if ftfy is not None:
            return ftfy.fix_text(s)
    except Exception:
        pass
    return s

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

def _frames_to_mp4_bytes(frames, fps: int = 24) -> bytes:
    """
    Convierte frames a mp4 bytes en memoria.
    Requiere imageio + ffmpeg. Si falla, devolvemos un error claro.
    """
    import numpy as np
    import imageio.v2 as imageio
    import tempfile
    import os as _os

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        _os.close(fd)

        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
        try:
            for f in frames:
                if hasattr(f, "convert"):  # PIL
                    arr = np.array(f.convert("RGB"))
                else:
                    arr = f
                writer.append_data(arr)
        finally:
            writer.close()

        with open(tmp_path, "rb") as f:
            return f.read()

    except Exception as e:
        # Usualmente aquí cae si falta ffmpeg / imageio-ffmpeg
        raise RuntimeError(
            f"MP4_ENCODE_FAILED: {e}. "
            "Si estás en tu propio Dockerfile, asegurate de instalar: imageio-ffmpeg y/o ffmpeg."
        )
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            try:
                _os.remove(tmp_path)
            except Exception:
                pass

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
        v = result[0]
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            return v[0]
        return v
    except Exception:
        pass

    raise RuntimeError(f"Could not extract frames from result type={type(result)}")

def _frames_multiple_of_4(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 16
    # Wan exige múltiplo de 4
    if n % 4 != 0:
        n = 4 * round(n / 4)
        if n <= 0:
            n = 16
    return n

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
# Find / list volume paths
# ---------------------------
def _walk_find_model_index(root: str, max_depth: int = 4, limit: int = 50) -> List[str]:
    hits = []
    root = root.rstrip("/")
    if not os.path.isdir(root):
        return hits

    queue: List[Tuple[str, int]] = [(root, 0)]
    while queue and len(hits) < limit:
        cur, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            items = os.listdir(cur)
        except Exception:
            continue

        if "model_index.json" in items:
            hits.append(cur)
            continue

        for name in items:
            p = os.path.join(cur, name)
            if os.path.isdir(p):
                queue.append((p, depth + 1))
    return hits

# ---------------------------
# Handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    ping = str(inp.get("ping") or "").strip().lower()

    # ---- debug ----
    if ping in ("echo", "debug"):
        return {
            "ok": True,
            "msg": "ECHO_OK",
            "job_keys": list(job.keys()),
            "input": inp,
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
            "env": {
                "WAN_T2V_PATH": os.environ.get("WAN_T2V_PATH"),
                "WAN_I2V_PATH": os.environ.get("WAN_I2V_PATH"),
            },
            "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
            "ftfy_available": ftfy is not None,
        }

    if ping == "smoke":
        return {"ok": True, "msg": "SMOKE_OK", "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}

    if ping == "gpu_sanity":
        return {"ok": True, **_gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}

    # ---- list paths ----
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
        out["ftfy_available"] = ftfy is not None
        return out

    if ping == "find_wan_models":
        roots = inp.get("roots") or ["/runpod-volume/models", "/runpod-volume", "/workspace/models", "/workspace"]
        max_depth = int(inp.get("max_depth") or 6)
        limit = int(inp.get("limit") or 80)

        found = {}
        for r in roots:
            found[str(r)] = _walk_find_model_index(str(r), max_depth=max_depth, limit=limit)

        return {
            "ok": True,
            "msg": "FIND_WAN_MODELS_OK",
            "roots": roots,
            "max_depth": max_depth,
            "limit": limit,
            "found_model_dirs": found,
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
            "ftfy_available": ftfy is not None,
        }

    if ping in ("probe_models", "probe_model", "models_probe"):
        return {
            "ok": True,
            "msg": "PROBE_OK",
            "paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
            "t2v": _probe_model_tree(MODEL_T2V_LOCAL),
            "i2v": _probe_model_tree(MODEL_I2V_LOCAL),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
            "ftfy_available": ftfy is not None,
        }

    # ---- load only ----
    if ping == "wan_load":
        which = str(inp.get("which") or "i2v").strip().lower()
        if which not in ("t2v", "i2v"):
            which = "i2v"

        try:
            t0 = time.time()
            if which == "i2v":
                _load_i2v(unload_other=True)
            else:
                _load_t2v(unload_other=True)

            return {
                "ok": True,
                "ping": "wan_load",
                "which": which,
                "paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
                "seconds": round(time.time() - t0, 3),
                "note": "Loaded pipeline only. No inference performed.",
                "ftfy_available": ftfy is not None,
            }
        except Exception as e:
            _hard_cleanup()
            return {"ok": False, "ping": "wan_load", "error": str(e), "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}

    # ---- image debug ----
    if ping == "image_debug":
        try:
            from PIL import Image
            s = str(inp.get("image_b64") or "")
            raw = _decode_b64(s)
            head_hex = raw[:32].hex()
            img = Image.open(BytesIO(raw))
            img.load()
            return {
                "ok": True,
                "msg": "IMAGE_DEBUG_OK",
                "bytes_len": len(raw),
                "head_hex": head_hex,
                "pil_format": img.format,
                "mode": img.mode,
                "size": [img.size[0], img.size[1]],
            }
        except Exception as e:
            preview = None
            try:
                preview = (str(inp.get("image_b64") or ""))[:140]
            except Exception:
                pass
            return {"ok": False, "msg": "IMAGE_DEBUG_FAIL", "error": str(e), "image_b64_preview": preview}

    # ---- I2V generate ----
    if ping in ("i2v_generate", "wan_i2v_generate"):
        image_b64 = inp.get("image_b64") or inp.get("image")
        prompt = _fix_text(str(inp.get("prompt") or "cinematic, ultra realistic, high quality"))
        negative = _fix_text(str(inp.get("negative") or ""))

        num_frames = _frames_multiple_of_4(int(inp.get("frames") or inp.get("num_frames") or 16))
        fps = int(inp.get("fps") or 24)
        steps = int(inp.get("steps") or inp.get("num_inference_steps") or 14)
        guidance = float(inp.get("guidance") or inp.get("guidance_scale") or 4.0)
        height = int(inp.get("height") or 512)
        width = int(inp.get("width") or 512)
        seed = inp.get("seed", None)

        if not image_b64:
            return {"ok": False, "ping": ping, "error": "missing image_b64", "gpu_info": _gpu_info(), **_diffusers_info()}

        try:
            pipe = _load_i2v(unload_other=True)
            init_image = _b64_to_pil_image(str(image_b64))

            generator = None
            if seed is not None and str(seed) != "":
                seed = int(seed)
                generator = torch.Generator(device="cuda").manual_seed(seed) if DEVICE == "cuda" else torch.Generator().manual_seed(seed)

            t0 = time.time()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative if negative else None,
                image=init_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            )

            frames = _extract_frames(result)
            mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
            video_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

            return {
                "ok": True,
                "ping": ping,
                "video_b64": video_b64,
                "video_bytes_len": len(mp4_bytes),
                "seconds": round(time.time() - t0, 3),
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
                "params": {
                    "frames": num_frames,
                    "fps": fps,
                    "steps": steps,
                    "guidance": guidance,
                    "height": height,
                    "width": width,
                    "seed": seed,
                },
                "ftfy_available": ftfy is not None,
                "note": "video_b64 es mp4 en base64. Súbelo a Supabase desde tu API (como Flux).",
            }

        except torch.cuda.OutOfMemoryError as e:
            _hard_cleanup()
            return {"ok": False, "ping": ping, "error": f"CUDA OOM: {str(e)}", "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}
        except Exception as e:
            _hard_cleanup()
            return {"ok": False, "ping": ping, "error": str(e), "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}

    # ---- T2V generate (prompt -> video) ----
    if ping in ("t2v_generate", "wan_t2v_generate"):
        prompt = _fix_text(str(inp.get("prompt") or ""))
        negative = _fix_text(str(inp.get("negative") or ""))

        if not prompt:
            return {"ok": False, "ping": ping, "error": "missing prompt", "gpu_info": _gpu_info(), **_diffusers_info()}

        num_frames = _frames_multiple_of_4(int(inp.get("frames") or inp.get("num_frames") or 16))
        fps = int(inp.get("fps") or 24)
        steps = int(inp.get("steps") or inp.get("num_inference_steps") or 14)
        guidance = float(inp.get("guidance") or inp.get("guidance_scale") or 4.0)
        height = int(inp.get("height") or 512)
        width = int(inp.get("width") or 512)
        seed = inp.get("seed", None)

        try:
            pipe = _load_t2v(unload_other=True)

            generator = None
            if seed is not None and str(seed) != "":
                seed = int(seed)
                generator = torch.Generator(device="cuda").manual_seed(seed) if DEVICE == "cuda" else torch.Generator().manual_seed(seed)

            t0 = time.time()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative if negative else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            )

            frames = _extract_frames(result)
            mp4_bytes = _frames_to_mp4_bytes(frames, fps=fps)
            video_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

            return {
                "ok": True,
                "ping": ping,
                "video_b64": video_b64,
                "video_bytes_len": len(mp4_bytes),
                "seconds": round(time.time() - t0, 3),
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
                "params": {
                    "frames": num_frames,
                    "fps": fps,
                    "steps": steps,
                    "guidance": guidance,
                    "height": height,
                    "width": width,
                    "seed": seed,
                },
                "ftfy_available": ftfy is not None,
                "note": "video_b64 es mp4 en base64. Súbelo a Supabase desde tu API (como Flux).",
            }

        except torch.cuda.OutOfMemoryError as e:
            _hard_cleanup()
            return {"ok": False, "ping": ping, "error": f"CUDA OOM: {str(e)}", "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}
        except Exception as e:
            _hard_cleanup()
            return {"ok": False, "ping": ping, "error": str(e), "gpu_info": _gpu_info(), **_diffusers_info(), "ftfy_available": ftfy is not None}

    # Default
    return {
        "ok": True,
        "msg": "DEFAULT_ECHO",
        "input": inp,
        "resolved_paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
        "gpu_info": _gpu_info(),
        **_diffusers_info(),
        "ftfy_available": ftfy is not None,
    }

runpod.serverless.start({"handler": handler})
