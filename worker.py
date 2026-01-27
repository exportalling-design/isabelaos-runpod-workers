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


DEFAULT_T2V_PATH = "/workspace/models/wan22/ti2v-5b"
DEFAULT_I2V_PATH = "/workspace/models/wan22/i2v-a14b"

MODEL_T2V_LOCAL = _normalize_model_path(os.environ.get("WAN_T2V_PATH", DEFAULT_T2V_PATH))
MODEL_I2V_LOCAL = _normalize_model_path(os.environ.get("WAN_I2V_PATH", DEFAULT_I2V_PATH))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_pipe_t2v = None
_pipe_i2v = None


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


def _list_dir_safe(path: str, limit: int = 200):
    try:
        items = sorted(os.listdir(path))
        if len(items) > limit:
            return items[:limit] + [f"...(+{len(items)-limit} more)"]
        return items
    except Exception as e:
        return [f"<cannot list: {e}>"]


def _probe_model_tree(base_path: str):
    """
    Verifica estructura del modelo SIN importar diffusers.
    Útil para confirmar que el volumen trae lo necesario.
    """
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

    # Archivos comunes de diffusers
    must_files = ["model_index.json"]
    optional_files = ["config.json", "scheduler", "tokenizer", "text_encoder", "transformer", "unet", "vae"]

    for f in must_files:
        out["checks"][f] = os.path.exists(os.path.join(base_path, f))

    # Subfolders típicos
    for name in optional_files:
        p = os.path.join(base_path, name)
        out["subfolders"][name] = {
            "exists": os.path.exists(p),
            "is_dir": os.path.isdir(p),
        }

    # Si hay vae/unet/transformer, listamos 1 nivel
    for sf in ["vae", "unet", "transformer", "scheduler", "text_encoder", "tokenizer"]:
        p = os.path.join(base_path, sf)
        if os.path.isdir(p):
            out["subfolders"][sf]["items"] = _list_dir_safe(p, limit=80)

    return out


def _lazy_import_wan():
    """
    Importa Wan* SOLO cuando lo pedimos.
    Esto evita que el worker se caiga al iniciar si diffusers no trae WanPipeline.
    """
    try:
        from diffusers import WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
        return WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline, None
    except Exception as e:
        return None, None, None, str(e)


def _load_t2v():
    global _pipe_t2v
    if _pipe_t2v is not None:
        return _pipe_t2v

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


def _load_i2v():
    global _pipe_i2v
    if _pipe_i2v is not None:
        return _pipe_i2v

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
            **_diffusers_info(),
        }

    if ping == "gpu_sanity":
        return {
            "ok": True,
            **_gpu_info(),
            **_diffusers_info(),
        }

    # -------------------------
    # FASE 3: Probe modelos (SIN diffusers)
    # -------------------------
    if ping in ("probe_models", "probe_model", "models_probe"):
        return {
            "ok": True,
            "msg": "PROBE_OK",
            "paths": {
                "t2v": MODEL_T2V_LOCAL,
                "i2v": MODEL_I2V_LOCAL,
            },
            "t2v": _probe_model_tree(MODEL_T2V_LOCAL),
            "i2v": _probe_model_tree(MODEL_I2V_LOCAL),
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
            "note": "Esto no carga pipelines. Solo verifica que el volumen tenga estructura diffusers.",
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
            "paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
            "gpu_info": _gpu_info(),
            **_diffusers_info(),
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
                "paths": {"t2v": MODEL_T2V_LOCAL, "i2v": MODEL_I2V_LOCAL},
                "gpu_info": _gpu_info(),
                **_diffusers_info(),
            }

    # Default: echo
    return {
        "ok": True,
        "msg": "NO_PING_PROVIDED",
        "input": inp,
        "gpu_info": _gpu_info(),
        **_diffusers_info(),
    }


runpod.serverless.start({"handler": handler})
