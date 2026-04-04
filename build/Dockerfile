# ===============================
# Dockerfile finale ctx_to_lora + Slurm + GPU
# ===============================

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"

# Zero interaction e workdir
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
ENV APP_PATH=/workspace

# ===============================
# Utilities di base
# ===============================
RUN apt-get update && apt-get install -y \
    curl git wget unzip build-essential python3.10 python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Setta Python 3.10 come default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip setuptools wheel

# ===============================
# Astral UV (installazione e venv) - usa binario assoluto
# ===============================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv self update && \
    /root/.local/bin/uv venv --python 3.10 --seed

# Aggiunge uv al PATH per i RUN successivi
ENV PATH=/root/.local/bin:$PATH

# ===============================
# Copia progetto
# ===============================
COPY src /workspace/src
COPY pyproject.toml /workspace/pyproject.toml
COPY install.sh /workspace/install.sh

# PYTHONPATH per import ctx_to_lora
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"

# ===============================
# Torch + CUDA compatibili
# ===============================
RUN uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124

# ===============================
# Tokenizers + Flash Attention
# ===============================
RUN uv pip install tokenizers==0.21.0
RUN uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN uv pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6

# ===============================
# Poetry + dipendenze pyproject.toml
# ===============================
RUN python3 -m pip install poetry
RUN poetry install --no-root

# ===============================
# Volumi persistenti
# ===============================
VOLUME ["/workspace/data", "/workspace/trained_d2l"]

# Ripristina frontend
ENV DEBIAN_FRONTEND=dialog