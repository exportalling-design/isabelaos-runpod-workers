import runpod
import torch

def handler(job):
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

runpod.serverless.start({"handler": handler})
