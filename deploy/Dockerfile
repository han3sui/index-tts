FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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

# 先装 CUDA 版 PyTorch（pip install . 不会读 uv 的源配置，默认装 CPU 版）
RUN pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cu121

# 再装项目本身和 API 依赖（PyTorch 已装好，pip 会跳过）
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
    API_KEY="" \
    USE_FP16=true \
    PORT=8300

EXPOSE 8300

CMD ["python3", "api.py"]
