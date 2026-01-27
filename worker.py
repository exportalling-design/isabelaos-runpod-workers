import os
import time
from typing import Any, Dict

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

DEFAULT_T2V_PATH = "/workspace/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/workspace/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))


# ---------------------------
# Lazy Torch + Diffusers state
# ---------------------------
_torch = None
_nn = None
_F = None
DEVICE = None
DTYPE = None

# diffusers classes (lazy)
WanPipeline = None
AutoencoderKLWan = None
WanImageToVideoPipeline = None

_pipe_t2v = None
_pipe_i2v = None


def _lazy_import_torch():
    global _torch, _nn, _F, DEVICE, DTYPE
    if _torch is not None:
        return

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

    _torch, _nn, _F = torch, nn, F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _lazy_import_diffusers():
    global WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
    if WanPipeline is not None:
        return

    # IMPORTANT: do not import diffusers until we actually need WAN
    from diffusers import WanPipeline as _WanPipeline
    from diffusers import AutoencoderKLWan as _AutoencoderKLWan
    from diffusers import WanImageToVideoPipeline as _WanImageToVideoPipeline

    WanPipeline = _WanPipeline
    AutoencoderKLWan = _AutoencoderKLWan
    WanImageToVideoPipeline = _WanImageToVideoPipeline


def _assert_model_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise RuntimeError(f"{label} model path not found: {path} (check volume mount)")


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


def _cuda_cleanup():
    _lazy_import_torch()
    if _torch.cuda.is_available():
        try: _torch.cuda.empty_cache()
        except Exception: pass
        try: _torch.cuda.ipc_collect()
        except Exception: pass


def _gpu_info():
    _lazy_import_torch()
    info = {
        "cuda_available": _torch.cuda.is_available(),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "torch_version": _torch.__version__,
    }
    if _torch.cuda.is_available():
        try:
            info["gpu"] = _torch.cuda.get_device_name(0)
            props = _torch.cuda.get_device_properties(0)
            info["vram_mb"] = int(props.total_memory / (1024 * 1024))
        except Exception:
            pass
    return info


def _load_t2v():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

    _lazy_import_torch()
    _lazy_import_diffusers()

    _assert_model_dir(MODEL_T2V_LOCAL, "T2V")
    t0 = time.time()
    print(f"[WAN_LOAD] Loading T2V LOCAL from: {MODEL_T2V_LOCAL}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_T2V_LOCAL,
        subfolder="vae",
        torch_dtype=_torch.float32,
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
    print(f"[WAN_LOAD] T2V loaded in {time.time()-t0:.2f}s", flush=True)
    return _pipe_t2v


def _load_i2v():
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

    _lazy_import_torch()
    _lazy_import_diffusers()

    _assert_model_dir(MODEL_I2V_LOCAL, "I2V")
    t0 = time.time()
    print(f"[WAN_LOAD] Loading I2V LOCAL from: {MODEL_I2V_LOCAL}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_I2V_LOCAL,
        subfolder="vae",
        torch_dtype=_torch.float32,
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
    print(f"[WAN_LOAD] I2V loaded in {time.time()-t0:.2f}s", flush=True)
    return _pipe_i2v


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    ping = str(inp.get("ping") or "").strip().lower()

    # -------------------------
    # Basic smoke
    # -------------------------
    if ping == "smoke":
        return {
            "ok": True,
            "msg": "SMOKE_OK",
            "input": inp,
            "input_keys": list(inp.keys()),
            "gpu_info": _gpu_info(),
        }

    if ping == "gpu_sanity":
        return {
            "ok": True,
            **_gpu_info(),
        }

    # -------------------------
    # WAN load test (NO inference)
    # -------------------------
    if ping == "wan_load":
        which = str(inp.get("which") or "t2v").strip().lower()
        if which not in ("t2v", "i2v", "both"):
            which = "t2v"

        out = {
            "ok": True,
            "ping": "wan_load",
            "which": which,
            "paths": {
                "t2v": MODEL_T2V_LOCAL,
                "i2v": MODEL_I2V_LOCAL,
            },
            "gpu_info": _gpu_info(),
            "loaded": [],
            "timing": {},
            "note": "Loaded pipelines only. No inference performed.",
        }

        try:
            t0 = time.time()
            if which in ("t2v", "both"):
                _load_t2v()
                out["loaded"].append("t2v")
            out["timing"]["t2v_s"] = round(time.time() - t0, 3) if which in ("t2v", "both") else None

            t1 = time.time()
            if which in ("i2v", "both"):
                _load_i2v()
                out["loaded"].append("i2v")
            out["timing"]["i2v_s"] = round(time.time() - t1, 3) if which in ("i2v", "both") else None

            return out

        except Exception as e:
            _cuda_cleanup()
            return {
                "ok": False,
                "ping": "wan_load",
                "error": str(e),
                "paths": {
                    "t2v": MODEL_T2V_LOCAL,
                    "i2v": MODEL_I2V_LOCAL,
                },
                "gpu_info": _gpu_info(),
            }

    # Default: just echo
    return {
        "ok": True,
        "msg": "NO_PING_PROVIDED",
        "input": inp,
        "gpu_info": _gpu_info(),
    }


print("[BOOT] worker.py loaded, starting runpod serverless...", flush=True)
runpod.serverless.start({"handler": handler})
