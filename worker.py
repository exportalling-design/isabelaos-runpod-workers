# worker.py
import os
import json
import base64
import time
import tempfile
from typing import Any, Dict, Optional

import runpod

# ----------------------------
# Helpers
# ----------------------------

def _gpu_name() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "NO_CUDA"
        return torch.cuda.get_device_name(0)
    except Exception:
        return "UNKNOWN"

def _require_gpu_ok():
    # Para endpoint i2v en A100 SXM, puedes forzar:
    # REQUIRE_A100=1
    require_a100 = os.getenv("REQUIRE_A100", "0") == "1"
    if not require_a100:
        return
    name = _gpu_name().lower()
    # A100 SXM normalmente aparece como "A100-SXM4-80GB" o similar
    if "a100" not in name:
        raise RuntimeError(f"GPU_NOT_ALLOWED: require A100 but got: {_gpu_name()}")

def _read_image_bytes_from_input(inp: Dict[str, Any]) -> Optional[bytes]:
    """
    Soporta:
      - imageB64: base64 puro
      - image_b64: base64 puro
      - imageUrl: descarga desde URL
    """
    import requests

    b64 = inp.get("imageB64") or inp.get("image_b64")
    url = inp.get("imageUrl")
    if b64:
        return base64.b64decode(b64)
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    return None

def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ----------------------------
# Wan2.2 runner (stub)
# ----------------------------
# ⚠️ Aquí NO puedo adivinar tu implementación exacta de Wan2.2,
# así que lo dejo listo para “conectar” a tu script real.
#
# Tú ya tienes Wan2.2 funcionando en el volumen.
# Solo cambia la función run_wan22_* para llamar tu pipeline real.
# ----------------------------

def run_wan22_t2v(prompt: str, negative: str, seconds: int, fps: int, seed: int,
                 width: int, height: int, steps: int, guidance: float,
                 out_path: str) -> None:
    """
    Genera video desde prompt en out_path (mp4).
    Conecta aquí tu pipeline real de Wan2.2.
    """
    # EJEMPLO: si tienes un script en el volumen:
    # python /workspace/wan22/worker_t2v.py --prompt ... --out out_path
    #
    # Por ahora lanzamos error claro para que sepas qué debes conectar.
    raise RuntimeError("WAN22_T2V_NOT_WIRED: conecta tu pipeline Wan2.2 en run_wan22_t2v()")

def run_wan22_i2v(image_bytes: bytes, prompt: str, negative: str, seconds: int, fps: int, seed: int,
                 steps: int, guidance: float,
                 out_path: str) -> None:
    """
    Genera video desde imagen en out_path (mp4).
    Conecta aquí tu pipeline real de Wan2.2.
    """
    raise RuntimeError("WAN22_I2V_NOT_WIRED: conecta tu pipeline Wan2.2 en run_wan22_i2v()")

# ----------------------------
# RunPod handler
# ----------------------------

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job["input"] viene desde:
      POST /v2/<endpoint_id>/run
      body: { "input": { ... } }
    """
    inp = job.get("input") or {}

    # Forzar modo por endpoint (opcional)
    forced_mode = os.getenv("WORKER_MODE", "").strip().lower()  # "t2v" o "i2v"
    mode = forced_mode or (inp.get("mode") or "").strip().lower()

    if mode not in ("t2v", "i2v"):
        return {"error": "MISSING_OR_INVALID_MODE", "detail": "mode must be 't2v' or 'i2v'"}

    # Guardia de GPU si lo necesitas (solo para i2v generalmente)
    if mode == "i2v":
        _require_gpu_ok()

    prompt = inp.get("prompt", "")
    negative = inp.get("negative", "") or ""
    seconds = int(inp.get("seconds", 4))
    fps = int(inp.get("fps", 16))
    seed = int(inp.get("seed", -1))
    steps = int(inp.get("steps", 25))
    guidance = float(inp.get("guidance", 7))

    width = int(inp.get("width", 768))
    height = int(inp.get("height", 432))

    if not prompt:
        return {"error": "MISSING_PROMPT"}

    # Output mp4 a /tmp (serverless)
    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, f"wan22_{mode}_{int(time.time())}.mp4")

    # Ejecutar según modo
    if mode == "t2v":
        run_wan22_t2v(
            prompt=prompt,
            negative=negative,
            seconds=seconds,
            fps=fps,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            out_path=out_path
        )
    else:
        img_bytes = _read_image_bytes_from_input(inp)
        if not img_bytes:
            return {"error": "MISSING_IMAGE", "detail": "Provide imageUrl or imageB64/image_b64"}
        run_wan22_i2v(
            image_bytes=img_bytes,
            prompt=prompt,
            negative=negative,
            seconds=seconds,
            fps=fps,
            seed=seed,
            steps=steps,
            guidance=guidance,
            out_path=out_path
        )

    # ⚠️ Devolver MP4 como base64 es pesado.
    # Para pruebas está bien. Para producción: subir a Supabase Storage/S3 y devolver URL.
    video_b64 = _b64_file(out_path)

    return {
        "ok": True,
        "mode": mode,
        "gpu": _gpu_name(),
        "video_b64": video_b64,
        "videoUrl": None,  # luego lo reemplazas cuando subas a storage
        "meta": {
            "seconds": seconds,
            "fps": fps,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "width": width,
            "height": height,
        }
    }

# Start serverless worker
runpod.serverless.start({"handler": handler})
