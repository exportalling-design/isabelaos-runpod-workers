FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# (Opcional pero recomendado) actualiza pip
RUN python3 -m pip install --upgrade pip

# ✅ 1) Instalar TORCH GPU (IGUAL a tu venv: torch 2.3.1+cu121)
# Esto fuerza el wheel correcto para CUDA 12.1
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

# ✅ 2) Instalar el resto de dependencias
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# ✅ 3) Copiar tu worker y entrypoint
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
