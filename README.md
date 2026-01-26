# IsabelaOS RunPod Workers (Serverless)

## Env vars recomendadas por endpoint
### T2V endpoint (prompt -> video)
- WORKER_MODE=t2v

### I2V endpoint (image -> video) A100
- WORKER_MODE=i2v
- REQUIRE_A100=1

## Network Volume
Montar el volume en /workspace y asegurar que:
- /workspace/wan22_env/bin/python existe (si quieres usar tu venv)
- Los modelos Wan2.2 están accesibles desde /workspace/...

## Importante
Editar worker.py:
- run_wan22_t2v()
- run_wan22_i2v()
y conectar tu pipeline real de Wan2.2 (script, diffusers, etc.)
