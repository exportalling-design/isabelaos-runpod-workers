FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Recomendado: pip actualizado
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ✅ Torch EXACTO del venv (CUDA 12.1)
# (esto evita que pip instale un torch random que no calce)
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# ✅ Instalar el resto de librerías
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# App
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
