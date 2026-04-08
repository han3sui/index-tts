import io
import os
import json
import tempfile
import soundfile as sf
import platform
import psutil
import torch
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
    emo_text: Optional[str] = Form(None, description="情绪描述文本（与 use_emo_text 配合使用）"),
    emo_vector: Optional[str] = Form(None, description="情绪向量 JSON, 如 [0.8,0,0,0,0,0,0,0]"),
    emo_alpha: Optional[float] = Form(None, description="情感强度 0.0-1.0，默认 0.6"),
    use_emo_text: Optional[bool] = Form(None, description="根据文本自动推断情感；若同时提供 emo_text 则用该文本推断"),
):
    """
    无状态合成接口：参考音频随请求发送，不在服务端持久化。

    情感控制优先级：emo_vector > emo_text/use_emo_text > 无情感（纯克隆）
    """
    tmp_file = None
    try:
        audio_bytes = await audio.read()
        suffix = os.path.splitext(audio.filename or "ref.wav")[1] or ".wav"
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_file.write(audio_bytes)
        tmp_file.close()

        alpha = emo_alpha if emo_alpha is not None else 0.6

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
                emo_alpha=alpha,
                use_emo_text=False,
            )
        elif use_emo_text:
            kwargs = dict(
                spk_audio_prompt=tmp_file.name,
                text=text,
                output_path='',
                use_emo_text=True,
                emo_alpha=alpha,
            )
            if emo_text:
                kwargs["emo_text"] = emo_text
            sr, wav_np = tts_v2.infer(**kwargs)
        elif emo_text:
            sr, wav_np = tts_v2.infer(
                spk_audio_prompt=tmp_file.name,
                text=text,
                output_path='',
                emo_text=emo_text,
                use_emo_text=True,
                emo_alpha=alpha,
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


def _read_file_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            val = f.read().strip()
            return int(val) if val else None
    except Exception:
        return None


def _is_in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _get_cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return name.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _get_cgroup_cpu_count() -> Optional[float]:
    """通过 cgroup 获取容器实际分配的 CPU 核心数"""
    # cgroup v2
    quota = _read_file_int("/sys/fs/cgroup/cpu.max")
    if quota is None:
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                parts = f.read().strip().split()
                if parts[0] != "max" and len(parts) == 2:
                    return int(parts[0]) / int(parts[1])
        except Exception:
            pass
    # cgroup v1
    cfs_quota = _read_file_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    cfs_period = _read_file_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if cfs_quota and cfs_quota > 0 and cfs_period and cfs_period > 0:
        return cfs_quota / cfs_period
    return None


def _get_cgroup_memory() -> Optional[dict]:
    """通过 cgroup 获取容器实际的内存限制和使用量"""
    # cgroup v2
    limit = _read_file_int("/sys/fs/cgroup/memory.max")
    usage = _read_file_int("/sys/fs/cgroup/memory.current")
    if limit and usage is not None:
        if limit > psutil.virtual_memory().total:
            return None
        return {"total": limit, "used": usage}

    # cgroup v1
    limit = _read_file_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage = _read_file_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit and usage is not None:
        if limit > psutil.virtual_memory().total:
            return None
        return {"total": limit, "used": usage}

    return None


def _get_cpu_info() -> dict:
    cpu_percent = psutil.cpu_percent(interval=0.5)
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    cgroup_cores = _get_cgroup_cpu_count() if _is_in_container() else None
    if cgroup_cores:
        logical = round(cgroup_cores)
        physical = logical

    return {
        "name": _get_cpu_model(),
        "percent": cpu_percent,
        "count_physical": physical,
        "count_logical": logical,
    }


def _get_memory_info() -> dict:
    cg_mem = _get_cgroup_memory() if _is_in_container() else None
    if cg_mem:
        total = cg_mem["total"]
        used = cg_mem["used"]
        percent = round(used / total * 100, 1) if total > 0 else 0
        return {
            "total_mb": round(total / 1024 / 1024),
            "used_mb": round(used / 1024 / 1024),
            "percent": percent,
        }

    mem = psutil.virtual_memory()
    return {
        "total_mb": round(mem.total / 1024 / 1024),
        "used_mb": round(mem.used / 1024 / 1024),
        "percent": mem.percent,
    }


def _get_gpu_info() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None

    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)

    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(dev)
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        nvmlShutdown()
    except Exception:
        gpu_util = None

    return {
        "name": torch.cuda.get_device_name(dev),
        "gpu_utilization": gpu_util,
        "memory_total_mb": round(total / 1024 / 1024),
        "memory_allocated_mb": round(allocated / 1024 / 1024),
        "memory_reserved_mb": round(reserved / 1024 / 1024),
    }


@app.get("/status")
async def status():
    return {
        "cpu": _get_cpu_info(),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_info(),
    }


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
            "/status": "系统状态 (GPU/CPU/内存负载)",
            "/health": "健康检查 (无需鉴权)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8300"))
    uvicorn.run(app, host="0.0.0.0", port=port)
