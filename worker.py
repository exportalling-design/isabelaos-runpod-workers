import runpod
import torch

def handler(job):
    gpu = None
    vram_mb = None
    torch_cuda = None

    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_mb = int(props.total_memory / (1024 * 1024))
            torch_cuda = torch.version.cuda
        except Exception as e:
            return {"ok": False, "error": "CUDA_QUERY_FAILED", "detail": str(e)}

    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "vram_mb": vram_mb,
        "torch_version": torch.__version__,
        "torch_cuda": torch_cuda,
    }

runpod.serverless.start({"handler": handler})
