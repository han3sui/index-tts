import io
import os
import time
import hashlib
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from pydantic import Field
from indextts.infer_v2 import IndexTTS2

API_KEY = os.environ.get("API_KEY", "")
USE_FP16 = os.environ.get("USE_FP16", "false").lower() == "true"
USE_DEEPSPEED = os.environ.get("USE_DEEPSPEED", "false").lower() == "true"
USE_CUDA_KERNEL = os.environ.get("USE_CUDA_KERNEL", "false").lower() == "true"
MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints")

app = FastAPI(title="IndexTTS API")
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY:
        return
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


tts_v2 = IndexTTS2(
    model_dir=MODEL_DIR,
    cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
    use_fp16=USE_FP16,
    use_deepspeed=USE_DEEPSPEED,
    use_cuda_kernel=USE_CUDA_KERNEL,
)

# ==== 确保目录存在 ====
PROMPTS_DIR = os.path.abspath("prompts")
OUTPUTS_DIR = os.path.abspath("outputs")
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ===== 工具函数 =====
def hash_filename(filename: str) -> str:
    """用md5对原始文件名编码，确保唯一性和兼容性"""
    ext = os.path.splitext(filename)[1] or ".wav"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{h}{ext}"


# ===== v1: 检查音频是否存在 =====
@app.get("/v1/check/audio", dependencies=[Depends(verify_api_key)])
async def check_audio(file_name: str):
    """检查参考音频是否存在 (v1)"""
    print("上传的文件名:", file_name)
    hashed_name = hash_filename(file_name)
    audio_path = os.path.join(PROMPTS_DIR, hashed_name)
    exists = os.path.isfile(audio_path)
    return JSONResponse(content={"exists": exists})


# ===== v1: 上传音频 =====
@app.post("/v1/upload_audio", dependencies=[Depends(verify_api_key)])
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...)
):
    """
    上传音频 (v1)
    - audio: 上传的音频文件
    - full_path: 客户端原始路径 (例如 C:\\Users\\xxx\\中等.wav)
    """
    content = await audio.read()
    print("上传的文件名:", full_path)

    # 存储用 hash 文件名
    encrypted_name = hash_filename(full_path)
    save_path = os.path.join(PROMPTS_DIR, encrypted_name)

    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "code": 200,
        "msg": "上传成功!",
        "original_file": full_path,
        "stored_file": save_path  # ✅ 返回服务端绝对路径
    }


# ===== v2: 请求模型 =====
class TextToSpeechRequest(BaseModel):
    text: str  # 要合成的文本
    audio_path: str  # 服务端参考音频路径
    emo_text: Optional[str] = None  # 情绪描述文本（模式3）
    emo_vector: Optional[list[float]] = Field(
        None, min_items=8, max_items=8,
        description="情绪向量，长度必须为8"
    )


# ===== v2: 合成接口 =====
@app.post("/v2/synthesize", dependencies=[Depends(verify_api_key)])
async def synthesize_speech_v2(request: TextToSpeechRequest):
    """
    V2 合成语音 API (基于 IndexTTS2)
    参数：
        - text: 要合成的文本
        - audio_path: 服务端参考音频路径 (来自 /v1/upload_audio 返回的 stored_file)
        - emo_text: 情绪描述文本 (可选)
    返回：
        - 直接输出生成的音频文件
    """
    audio_path = request.audio_path
    hashed_name = hash_filename(audio_path)
    audio_path = os.path.join(PROMPTS_DIR, hashed_name)
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=404, detail=f"音频不存在: {audio_path}")

    try:
        # 这里调用 TTS 模型推理，不写 output_path，直接返回音频数据
        if request.emo_vector:  # === 向量控制模式 ===
            vec = request.emo_vector
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=audio_path,
                text=request.text,
                output_path='',      # 不保存到文件
                emo_vector=vec,
                emo_alpha=0.6,
                use_emo_text=False
            )

        elif request.emo_text:  # === 文本情绪控制模式 ===
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=audio_path,
                text=request.text,
                output_path='',
                emo_text=request.emo_text,
                use_emo_text=True,
                emo_alpha=0.6
            )

        else:  # === 中性模式 ===
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=audio_path,
                text=request.text,
                output_path='',
                use_emo_text=False
            )

        # 写入内存 buffer
        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV")
        buf.seek(0)

        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "message": f"Synthesis failed: {str(e)}"
            },
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def info():
    return {
        "message": "IndexTTS API",
        "auth": "enabled" if API_KEY else "disabled",
        "endpoints": {
            "/v1/upload_audio": "上传音频 (保存到 prompts)",
            "/v1/check/audio": "检查音频是否存在",
            "/v2/synthesize": "v2 合成语音 (IndexTTS2)",
            "/health": "健康检查 (无需鉴权)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8300"))
    uvicorn.run(app, host="0.0.0.0", port=port)
