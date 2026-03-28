FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_CACHE=/app/checkpoints/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install . fastapi uvicorn python-multipart soundfile "huggingface_hub[cli,hf_xet]"

# ===== 下载 IndexTTS2 主模型 =====
RUN huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /app/checkpoints

# ===== 预下载 HuggingFace 运行时依赖模型 =====
RUN python3 -c "\
from transformers import SeamlessM4TFeatureExtractor; \
SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0'); \
from huggingface_hub import hf_hub_download; \
hf_hub_download('amphion/MaskGCT', filename='semantic_codec/model.safetensors'); \
hf_hub_download('funasr/campplus', filename='campplus_cn_common.bin'); \
from indextts.s2mel.modules.bigvgan import bigvgan; \
bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x'); \
print('All models pre-downloaded successfully')"

ENV MODEL_DIR=/app/checkpoints \
    USE_FP16=true \
    PORT=8300

EXPOSE 8300

CMD ["python3", "api.py"]
