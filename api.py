import io
import os
import json
import tempfile
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
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


@app.post("/v2/synthesize", dependencies=[Depends(verify_api_key)])
async def synthesize(
    audio: UploadFile = File(..., description="参考音频文件"),
    text: str = Form(..., description="要合成的文本"),
    emo_text: Optional[str] = Form(None, description="情绪描述文本"),
    emo_vector: Optional[str] = Form(None, description="情绪向量 JSON, 如 [0.8,0,0,0,0,0,0,0]"),
):
    """
    无状态合成接口：参考音频随请求发送，不在服务端持久化。
    返回合成后的 WAV 音频流。
    """
    tmp_file = None
    try:
        audio_bytes = await audio.read()
        suffix = os.path.splitext(audio.filename or "ref.wav")[1] or ".wav"
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_file.write(audio_bytes)
        tmp_file.close()

        vec = None
        if emo_vector:
            vec = json.loads(emo_vector)
            if not isinstance(vec, list) or len(vec) != 8:
                raise HTTPException(status_code=400, detail="emo_vector 必须是长度为 8 的数组")

        if vec:
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=tmp_file.name,
                text=text,
                output_path='',
                emo_vector=vec,
                emo_alpha=0.6,
                use_emo_text=False,
            )
        elif emo_text:
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=tmp_file.name,
                text=text,
                output_path='',
                emo_text=emo_text,
                use_emo_text=True,
                emo_alpha=0.6,
            )
        else:
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=tmp_file.name,
                text=text,
                output_path='',
                use_emo_text=False,
            )

        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV")
        buf.seek(0)

        return Response(content=buf.read(), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 1, "message": f"Synthesis failed: {str(e)}"},
        )
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def info():
    return {
        "message": "IndexTTS API",
        "auth": "enabled" if API_KEY else "disabled",
        "endpoints": {
            "/v2/synthesize": "合成语音 (multipart: audio + text)",
            "/health": "健康检查 (无需鉴权)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8300"))
    uvicorn.run(app, host="0.0.0.0", port=port)
