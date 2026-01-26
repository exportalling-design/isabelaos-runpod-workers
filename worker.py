import runpod

print("[BOOT] SMOKE WORKER v1 - isabelaos-runpod-workers", flush=True)

def handler(job):
    inp = job.get("input") or {}
    return {
        "ok": True,
        "msg": "SMOKE_OK",
        "input": inp,
        "input_keys": list(inp.keys()),
    }

runpod.serverless.start({"handler": handler})
