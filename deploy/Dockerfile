ARG CUDA_VERSION=12.8
ARG TORCH_VERSION=2.8.0
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install ".[webui]" fastapi uvicorn python-multipart soundfile

VOLUME /app/checkpoints
VOLUME /app/prompts
VOLUME /app/outputs

ENV MODEL_DIR=/app/checkpoints \
    API_KEY="" \
    PORT=8300

EXPOSE 8300

CMD ["python3", "api.py"]
