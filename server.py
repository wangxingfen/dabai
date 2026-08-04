"""3D 虚拟 AI 角色陪聊 —— FastAPI 服务器

特性：
- WebSocket 实时双向通信（文本 + 语音）
- 语音转文字 (SiliconFlow SenseVoiceSmall，API 失败自动降级 faster-whisper 本地模型)
- 文字转语音 (edge-tts)
- 3D 模型导入与管理 (glb/gltf/vrm)
- 静态前端托管 (web/)
- 绑定 0.0.0.0 实现局域网手机访问
- MCP 工具调用 + Function Calling
- 长期记忆（SQLite 持久化）
"""
import asyncio
import base64
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from starlette.websockets import WebSocketState
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from agent import AIAgent, TextDelta, ToolCallStart, ToolCallResult, get_available_tools
from game_engine import GameEngine
from perception_dispatcher import PerceptionDispatcher
from reward_memory import RewardMemory
from rl_coordinator import get_coordinator, UnifiedMode, UnifiedState

# 全局主动说话冷却（秒）：所有非用户消息驱动的主动说话（RL 调度 /
# 感知派发 / 环境快照 / 前端召唤）共用此闸门，防止"自言自语"式高频说话
ACTIVE_SPEAK_COOLDOWN = 40.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).parent.resolve()
WEB_DIR = BASE_DIR / "web"
AUDIO_DIR = BASE_DIR / "audio_cache"
MODELS_DIR = BASE_DIR / "models"
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"
BGM_DIR = BASE_DIR / "bgm"
AUDIO_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
BACKGROUNDS_DIR.mkdir(exist_ok=True)
BGM_DIR.mkdir(exist_ok=True)


app = FastAPI(title="3D AI 陪聊")

# 托管生成的音频文件
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
# 托管上传的 3D 模型
app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")
# 托管上传的 3D 背景模型
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")
# 托管背景音乐文件
app.mount("/bgm", StaticFiles(directory=str(BGM_DIR)), name="bgm")
# 托管前端静态资源 (css/js)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# 允许的 3D 模型扩展名
ALLOWED_MODEL_EXTS = {".glb", ".gltf", ".vrm"}
MAX_MODEL_SIZE = 80 * 1024 * 1024  # 80MB


def get_lan_ip() -> str:
    """获取本机局域网 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _is_global_ipv6(addr: str) -> bool:
    """判断 IPv6 地址是否为全局单播地址，排除 ::、::1、fe80:: 等非公开地址。"""
    import ipaddress
    try:
        ip = ipaddress.IPv6Address(addr.split("%")[0])
        return ip.is_global
    except (ValueError, ipaddress.AddressValueError):
        return False


def get_global_ipv6() -> str:
    """获取本机全局 IPv6 地址（用于显示，失败返回空串）。

    优先用 UDP connect 让 OS 选出出口地址；若不可用，回退枚举所有网卡地址。
    """
    # 方案1: UDP connect 让系统选出出口 IPv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("2001:4860:4860::8888", 80))
        ip = s.getsockname()[0]
        s.close()
        if _is_global_ipv6(ip):
            return ip
    except Exception:
        pass
    # 方案2: 枚举本机所有 IPv6 地址，挑一个全局单播
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6,
                                   socket.SOCK_STREAM)
        for fam, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            if _is_global_ipv6(ip):
                return ip
    except Exception:
        pass
    # 方案3: 枚举网卡（Windows netifaces 兼容兜底）
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET6, [])
            for a in addrs:
                ip = a.get("addr", "").split("%")[0]  # 去掉 scope id
                if _is_global_ipv6(ip):
                    return ip
    except Exception:
        pass
    return ""


def load_config():
    with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TTS（多引擎） ----------
DEFAULT_VOICE = "zh-CN-YunxiNeural"  # 男声云稀，亲切自然

# 句子结束符（用于流式分句 TTS）
SENTENCE_END = re.compile(r"[。！？!?\n…；;~]")

# 全局 TTS 配置（运行时可被前端 /api/tts/config 修改，持久化到 tts_config.json）
TTS_CONFIG_FILE = BASE_DIR / "tts_config.json"
DEFAULT_TTS_CONFIG = {
    "engine": "edge_tts",                       # "edge_tts" | "gpt_sovits"
    "edge_voice": DEFAULT_VOICE,                # edge_tts 音色 ShortName
    "edge_rate": "+8%",                         # 语速
    "gptsovits_url": "http://127.0.0.1:7860/",  # GPT-SoVITS API 服务地址
    "gptsovits_ref_audio": "",                  # 参考音频绝对路径
    "gptsovits_character": "星见雅",
}

def _load_tts_config():
    """从文件加载 TTS 配置，不存在则用默认值。"""
    if TTS_CONFIG_FILE.exists():
        try:
            with open(TTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = {**DEFAULT_TTS_CONFIG, **saved}
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_TTS_CONFIG)

def _save_tts_config(cfg: dict):
    """保存 TTS 配置到文件。"""
    try:
        with open(TTS_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 TTS 配置失败: {e}")

tts_config = _load_tts_config()

# 缓存 edge_tts 中文音色列表
_edge_voices_cache: list | None = None


def cleanup_old_audio():
    """清理旧音频文件，最多保留 5 个（兜底，正常流程已即时销毁）"""
    try:
        files = sorted(AUDIO_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
        for old in files[:-5]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def _start_periodic_cleanup():
    """后台守护线程：每 30 分钟清理一次 audio_cache，防止磁盘/缓存无限增长"""
    def _loop():
        while True:
            time.sleep(1800)
            try:
                cleanup_old_audio()
            except Exception:
                pass
    threading.Thread(target=_loop, daemon=True, name="audio-cache-cleanup").start()


_start_periodic_cleanup()


async def get_edge_voices() -> list:
    """获取 edge_tts 中文音色列表（带缓存）"""
    global _edge_voices_cache
    if _edge_voices_cache is None:
        voices = await edge_tts.list_voices()
        _edge_voices_cache = [
            {
                "name": v["ShortName"],
                "locale": v["Locale"],
                "gender": v.get("Gender", ""),
                "friendly": f"{v.get('Gender','')}声 · {v['ShortName'].split('-')[-1].replace('Neural','')}"
            }
            for v in voices if v["Locale"].startswith("zh-")
        ]
    return _edge_voices_cache


async def generate_tts_edge(text: str, voice: str, rate: str):
    """edge_tts 在线合成 mp3，返回 (audio_bytes, mime_type) 或 None。
    
    音频数据直接返回，不保留持久文件。"""
    clean = re.sub(r"[*_`#>\-\[\]\(\)]", "", text).strip()
    if not clean:
        return None
    # 使用临时文件生成，读完后立即删除
    audio_id = f"{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_DIR / audio_id
    try:
        communicate = edge_tts.Communicate(clean, voice, rate=rate, volume="+0%")
        await communicate.save(str(audio_path))
        data = audio_path.read_bytes()
        return (data, "audio/mpeg")
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


def resolve_ref_audio_by_character(character: str) -> str:
    """按角色名从 gpt_sovits.json 读参考音频路径（对齐 voice.py 的 index_tts 逻辑）。

    查找位置（按顺序）：
      1. tools/gpt_sovits/gpt_sovits.json  （voice.py 原路径）
      2. gpt_sovits.json                   （项目根目录）
    json 格式：{ "角色名": { "ref_audio_path": "绝对路径" } }
    """
    if not character:
        return ""
    for rel in ("tools/gpt_sovits/gpt_sovits.json", "gpt_sovits.json"):
        p = BASE_DIR / rel
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get(character, {}).get("ref_audio_path", "")
            except Exception as e:
                print(f"[GPT-SoVITS] 读取 {rel} 失败: {e}")
    return ""


def generate_tts_gptsovits(text: str, ref_audio: str, url: str, character: str = ""):
    """调用本地 GPT-SoVITS（gradio 7860）合成 wav，返回 (audio_bytes, mime_type) 或 None。

    参考音频优先级：显式传入 ref_audio > 按角色名查 gpt_sovits.json > None（服务端默认）。
    音频数据直接返回，不保留持久文件。
    """
    from gradio_client import Client as GradioClient, handle_file
    # 没显式传 ref_audio 时，按角色名从配置文件查（对齐 voice.py）
    if not ref_audio or not os.path.exists(ref_audio):
        ref_audio = resolve_ref_audio_by_character(character)
    client = GradioClient(url)
    prompt = handle_file(ref_audio) if (ref_audio and os.path.exists(ref_audio)) else None
    result = client.predict(
        prompt=prompt,
        text=text,
        infer_mode="批次推理",
        max_text_tokens_per_sentence=120,
        sentences_bucket_max_size=4,
        param_5=True, param_6=0.8, param_7=30, param_8=1,
        param_9=0, param_10=3, param_11=10, param_12=600,
        api_name="/gen_single",
    )
    # result 通常是 [text_output, audio_path, ...]
    src = result[1] if isinstance(result, (list, tuple)) else result.get("value")
    if not src or not os.path.exists(src):
        raise RuntimeError("GPT-SoVITS 未返回有效音频")
    # 直接读二进制，不复制到 audio_cache
    with open(src, "rb") as f:
        data = f.read()
    return (data, "audio/wav")


async def generate_tts(text: str):
    """根据当前引擎生成 TTS，返回 (audio_bytes, mime_type) 或 None，失败自动回退 edge_tts。"""
    engine = tts_config.get("engine", "edge_tts")
    try:
        if engine == "gpt_sovits":
            return await asyncio.to_thread(
                generate_tts_gptsovits,
                text,
                tts_config["gptsovits_ref_audio"],
                tts_config["gptsovits_url"],
                tts_config.get("gptsovits_character", ""),
            )
        return await generate_tts_edge(text, tts_config["edge_voice"], tts_config["edge_rate"])
    except Exception as e:
        print(f"[TTS] {engine} 失败，回退 edge_tts: {e}")
        try:
            fallback_voice = tts_config.get("edge_voice", DEFAULT_VOICE)
            fallback_rate = tts_config.get("edge_rate", "+8%")
            return await generate_tts_edge(text, fallback_voice, fallback_rate)
        except Exception as e2:
            print(f"[TTS] edge_tts 也失败: {e2}")
            return None


# ---------- STT ----------
# 本地语音识别模型（懒加载，仅首次使用或 API 失败时初始化）
_local_stt_model = None
_local_stt_lock = threading.Lock()


def _is_valid_audio(path: str) -> bool:
    """快速检查文件是否可能是音频（基于大小和常见魔数）。

    只过滤明显非音频的文件（太小/空文件）。移动端 MediaRecorder
    可能产生非标准封装格式，即使魔数不匹配也交给 ffmpeg 自行判断。
    """
    try:
        file_size = os.path.getsize(path)
        if file_size < 32:
            print(f"[ffmpeg] 跳过空文件: {file_size} 字节")
            return False
        return True  # 放行所有 > 32 字节的文件，ffmpeg 会自动检测格式
    except Exception:
        return False


def _mime_to_ext(mime_type: str) -> str:
    """将 MIME 类型映射为文件扩展名。

    容器格式优先于编解码器关键词（如 audio/webm;codecs=opus 应识别为 webm，而非 opus→ogg）。
    """
    mime_lower = (mime_type or "").lower()
    # 容器格式
    if "webm" in mime_lower:
        return ".webm"
    if "ogg" in mime_lower:
        return ".ogg"
    if "mp4" in mime_lower or "m4a" in mime_lower or "aac" in mime_lower:
        return ".m4a"
    if "wav" in mime_lower:
        return ".wav"
    # 纯编解码器关键词兜底（如只有 "opus" 而无容器信息）
    if "opus" in mime_lower:
        return ".ogg"
    return ".webm"


def _detect_ext_from_bytes(data: bytes) -> str:
    """根据文件魔数检测实际音频格式，修正 MIME 类型可能不准确的问题。"""
    if len(data) < 12:
        return ".webm"
    # WebM / Matroska: 0x1A 0x45 0xDF 0xA3
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    # Ogg: "OggS"
    if data[:4] == b"OggS":
        return ".ogg"
    # RIFF WAVE: "RIFF" ... "WAVE"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    # MP4/M4A: ftyp 在偏移 4
    if data[4:8] == b"ftyp":
        return ".m4a"
    # MP3: 同步字 0xFFE/0xFFF
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    # 裸 Opus 数据包检测：首字节为合法 Opus TOC 字节
    # TOC bits 7-5 (config): 000=SILK, 001=CELT, 010=Hybrid (1帧模式)
    if len(data) >= 1:
        config = data[0] & 0xE0
        if config in (0x00, 0x20, 0x40):
            return ".opus"
    # 默认回退
    return ".webm"


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """用 ffmpeg 把任意音频格式转成 16k 单声道 wav，并做降噪 + 响度归一化提升识别率。

    滤波链：
    - highpass=f=80      去掉低频嗡嗡（电源/空调/手震）
    - lowpass=f=8000     去掉高频无用信号（16k 采样上限 8k）
    - dynaudnorm         动态响度归一化，放大安静部分，小声说话也能被识别
    - afftdn=nr=10       频域降噪，压制稳态背景噪声
    """
    # 预处理：检查文件是否是常见音频格式
    if not _is_valid_audio(input_path):
        print(f"[ffmpeg] 跳过无效音频文件: {os.path.getsize(input_path)} 字节")
        return False

    af = "highpass=f=80,lowpass=f=8000,afftdn=nr=10,dynaudnorm=p=0.9:s=5"
    is_raw_opus = input_path.endswith(".opus")

    def _run_attempts(attempt_list):
        for attempt, (cmd, label) in enumerate(attempt_list):
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    return True
                else:
                    stderr_tail = result.stderr.decode("utf-8", errors="replace")[-300:].strip()
                    # Windows 上退出码 > 2^31 通常是 ffmpeg 进程崩溃（如访问违例）
                    crash_hint = " (疑似崩溃)" if result.returncode > 0x7FFFFFFF else ""
                    print(f"[ffmpeg] {label} 失败 (rc={result.returncode}){crash_hint}: {stderr_tail}")
            except subprocess.TimeoutExpired:
                print(f"[ffmpeg] {label} 超时")
            except FileNotFoundError:
                print("[ffmpeg] ffmpeg 未安装或不在 PATH 中")
                return False
            except Exception as e:
                print(f"[ffmpeg] {label} 异常: {e}")
        return False

    attempts = []
    if is_raw_opus:
        # 裸 Opus 数据包：必须使用 -f opus 显式指定 demuxer
        attempts.append(
            (["ffmpeg", "-y", "-f", "opus", "-i", input_path,
              "-af", af,
              "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
              output_path], "Opus滤波版"))
        attempts.append(
            (["ffmpeg", "-y", "-f", "opus", "-i", input_path,
              "-ar", "16000", "-ac", "1",
              output_path], "Opus基础版"))
    # 通用容器格式尝试（WebM/Ogg/MP4 等）
    attempts.extend([
        # 方案1: 完整滤波链
        (["ffmpeg", "-y", "-i", input_path,
          "-af", af,
          "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
          output_path], "滤波版"),
        # 方案2: 基础转换（无滤波）
        (["ffmpeg", "-y", "-i", input_path,
          "-ar", "16000", "-ac", "1",
          output_path], "基础版"),
        # 方案3: 容错 + 修复损坏的 webm（处理 MediaRecorder 未正确封包的情况）
        (["ffmpeg", "-y",
          "-err_detect", "ignore_err",
          "-fflags", "+genpts+igndts",
          "-analyzeduration", "10M",
          "-probesize", "10M",
          "-vn",
          "-i", input_path,
          "-ar", "16000", "-ac", "1",
          output_path], "修复版"),
    ])

    ok = _run_attempts(attempts)
    if ok:
        return True

    # 当 .opus 格式导致 ffmpeg 崩溃时，自动回退：把文件重命名为 .webm 再重试
    if is_raw_opus:
        print("[ffmpeg] Opus 解析失败，尝试作为 WebM 容器回退重试…")
        webm_path = input_path.replace(".opus", ".webm")
        try:
            os.rename(input_path, webm_path)
            fallback_attempts = [
                (["ffmpeg", "-y", "-i", webm_path,
                  "-ar", "16000", "-ac", "1",
                  output_path], "WebM回退版"),
                (["ffmpeg", "-y",
                  "-err_detect", "ignore_err",
                  "-fflags", "+genpts+igndts",
                  "-analyzeduration", "10M",
                  "-probesize", "10M",
                  "-vn",
                  "-i", webm_path,
                  "-ar", "16000", "-ac", "1",
                  output_path], "WebM回退修复版"),
            ]
            if _run_attempts(fallback_attempts):
                return True
        except Exception as e:
            print(f"[ffmpeg] WebM 回退异常: {e}")

    return False


def _get_local_stt_config():
    """读取 STT 本地降级配置。"""
    cfg = load_config().get("stt", {})
    return {
        "enabled": cfg.get("local_enabled", True),
        "model": cfg.get("local_model", "base"),
        "device": cfg.get("local_device", "cpu"),
        "compute_type": cfg.get("local_compute_type", "int8"),
        "api_timeout": cfg.get("api_timeout", 5),
        "hf_endpoint": cfg.get("hf_endpoint", "https://hf-mirror.com"),
    }


def _load_local_stt_model():
    """懒加载本地 faster-whisper 模型（线程安全，只加载一次）。"""
    global _local_stt_model
    if _local_stt_model is not None:
        return _local_stt_model
    with _local_stt_lock:
        if _local_stt_model is not None:
            return _local_stt_model
        cfg = _get_local_stt_config()

        # 设置 HuggingFace 镜像，国内网络环境下可正常下载模型
        # 注意：必须用直接赋值而非 setdefault，否则已有环境变量会阻止镜像生效
        hf_endpoint = cfg.get("hf_endpoint", "")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            os.environ["HF_HUB_ENDPOINT"] = hf_endpoint

        print(f"[STT-Local] 正在加载本地模型 faster-whisper/{cfg['model']}（{cfg['device']}/{cfg['compute_type']}）…")
        try:
            from faster_whisper import WhisperModel
            _local_stt_model = WhisperModel(
                cfg["model"],
                device=cfg["device"],
                compute_type=cfg["compute_type"],
            )
            print(f"[STT-Local] ✅ 本地模型就绪")
        except Exception as e:
            print(f"[STT-Local] ❌ 模型加载失败: {e}")
            _local_stt_model = False  # 标记加载失败，避免反复重试
        return _local_stt_model


def speech_to_text_local(wav_path: str) -> str:
    """使用本地 faster-whisper 模型进行语音识别（API 降级方案）。

    模型首次调用时自动下载（约 142MB），后续调用直接使用缓存。
    """
    model = _load_local_stt_model()
    if model is False:
        raise RuntimeError("本地 STT 模型未就绪")
    segments, info = model.transcribe(wav_path, beam_size=5, language="zh",
                                       vad_filter=True,
                                       vad_parameters=dict(
                                           min_silence_duration_ms=500,
                                           threshold=0.4,
                                       ))
    text = " ".join(seg.text.strip() for seg in segments)
    if not text:
        raise RuntimeError("本地模型未识别到语音内容")
    print(f"[STT-Local] 识别结果: {text[:60]}{'…' if len(text) > 60 else ''}")
    return text


def speech_to_text(wav_path: str) -> str:
    """调用 SiliconFlow 语音识别 API，失败时自动降级到本地模型。"""
    import requests
    config = load_config()
    stt_cfg = _get_local_stt_config()

    # 第一选择：SiliconFlow API（快速、准确）
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    with open(wav_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "FunAudioLLM/SenseVoiceSmall"}
        try:
            resp = requests.post(url, headers=headers, files=files, data=data,
                                 timeout=stt_cfg["api_timeout"])
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            if text:
                return text
            # API 返回空文本也视为失败
            raise RuntimeError("API 返回空文本")
        except Exception as e:
            print(f"[STT-API] 失败（{e}），尝试本地降级…")

    # 第二选择：本地 faster-whisper 降级
    if stt_cfg["enabled"]:
        try:
            return speech_to_text_local(wav_path)
        except Exception as e2:
            print(f"[STT-Local] 降级也失败: {e2}")
            raise RuntimeError(f"语音识别失败（API + 本地均失败）") from e2
    raise RuntimeError(f"语音识别失败且本地降级已禁用")


# ---------- 路由 ----------
@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/info")
async def info():
    return {"lan_ip": get_lan_ip(), "port": 8000}


# ---------- TTS 配置 ----------
@app.get("/api/tts/voices")
async def tts_voices():
    """返回 edge_tts 中文音色列表"""
    try:
        return {"voices": await get_edge_voices()}
    except Exception as e:
        return {"voices": [], "error": str(e)}


@app.get("/api/tts/characters")
async def tts_characters():
    """返回 gpt_sovits.json 里配置的角色列表（对齐 voice.py）"""
    for rel in ("tools/gpt_sovits/gpt_sovits.json", "gpt_sovits.json"):
        p = BASE_DIR / rel
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {"characters": list(data.keys()), "source": rel}
            except Exception as e:
                return {"characters": [], "error": str(e)}
    return {"characters": [], "error": "未找到 gpt_sovits.json"}


@app.get("/api/tts/config")
async def tts_config_get():
    return tts_config


@app.post("/api/tts/config")
async def tts_config_set(payload: dict):
    """更新 TTS 配置（部分字段）"""
    for k in ("engine", "edge_voice", "edge_rate",
              "gptsovits_url", "gptsovits_ref_audio", "gptsovits_character"):
        if k in payload:
            tts_config[k] = payload[k]
    # 切换音色后清空缓存，便于下次重新拉取
    if "edge_voice" in payload:
        global _edge_voices_cache
        _edge_voices_cache = None
    # 持久化保存
    _save_tts_config({k: tts_config[k] for k in DEFAULT_TTS_CONFIG})
    return {"ok": True, "config": tts_config}


# ---------- 角色卡片 ----------
CHARACTER_CARDS_FILE = BASE_DIR / "character_cards.json"


def _load_character_cards() -> list:
    """加载角色卡片列表。"""
    if CHARACTER_CARDS_FILE.exists():
        try:
            with open(CHARACTER_CARDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("cards", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.warning(f"加载角色卡片失败: {e}")
    return []


def _save_character_cards(cards: list):
    """保存角色卡片列表到文件。"""
    try:
        with open(CHARACTER_CARDS_FILE, "w", encoding="utf-8") as f:
            json.dump({"cards": cards}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存角色卡片失败: {e}")


def _save_role_config(role_name: str, system_prompt: str, user_name: str = "", tools: dict = None):
    """更新 settings.json 中的角色名、系统提示词、用户称呼与工具配置（应用角色卡片时调用）。"""
    path = BASE_DIR / "settings.json"
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["role_name"] = role_name
    cfg["system_prompt"] = system_prompt
    cfg["user_name"] = user_name
    if tools is not None:
        cfg.setdefault("agent", {})
        cfg["agent"]["enable_tools"] = bool(tools.get("enabled", True))
        cfg["agent"]["allowed_tools"] = list(tools.get("allowed", []) or [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------- 大语言模型（LLM）配置 ----------
@app.get("/api/llm/config")
async def llm_config_get():
    """获取当前大语言模型配置（base_url / model / api_key），供角色卡片表单回填默认值。"""
    try:
        cfg = load_config()
        return {
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "api_key": cfg.get("api_key", ""),
        }
    except Exception as e:
        return {"base_url": "", "model": "", "api_key": "", "error": str(e)}


@app.get("/api/llm/models")
async def llm_models_list(base_url: str = "", api_key: str = ""):
    """从模型提供方自动加载可用模型列表（OpenAI 兼容 /models 接口）。

    未传 base_url / api_key 时回退到 settings.json 中的全局默认配置。
    """
    cfg = load_config()
    base_url = (base_url or "").strip() or (cfg.get("base_url") or "").strip()
    api_key = (api_key or "").strip() or (cfg.get("api_key") or "").strip()
    if not base_url or not api_key:
        return {"models": [], "error": "缺少 base_url 或 api_key"}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        names = []
        async for m in client.models.list():
            names.append(m.id)
        return {"models": sorted(names)}
    except Exception as e:
        return {"models": [], "error": str(e)}



@app.get("/api/character_cards")
async def character_cards_list():
    """获取所有角色卡片。"""
    return {"cards": _load_character_cards()}


@app.get("/api/config/role")
async def role_config_get():
    """获取当前角色配置（角色名 + 系统提示词），供编辑表单回填。"""
    try:
        cfg = load_config()
        return {
            "role_name": cfg.get("role_name", "AI助手"),
            "system_prompt": cfg.get("system_prompt", ""),
        }
    except Exception as e:
        return {"role_name": "AI助手", "system_prompt": "", "error": str(e)}


@app.get("/api/config/tools")
async def tools_config_get():
    """获取当前工具配置（是否启用 + 白名单），供角色卡片编辑表单回填。"""
    try:
        cfg = load_config()
        agent_cfg = cfg.get("agent", {})
        return {
            "enable_tools": bool(agent_cfg.get("enable_tools", True)),
            "allowed_tools": agent_cfg.get("allowed_tools", []) or [],
        }
    except Exception as e:
        return {"enable_tools": True, "allowed_tools": [], "error": str(e)}


@app.get("/api/tools")
async def tools_list():
    """返回当前所有可用工具（本地 + MCP），供角色卡片配置工具白名单。"""
    tools = get_available_tools()
    # 补充 MCP 工具（需 shared agent 已初始化）
    try:
        ag = await get_shared_agent("default")
        seen = {t["name"] for t in tools}
        for t in getattr(ag, "_all_tools", []) or []:
            fn = t.get("function", {})
            name = fn.get("name", "")
            if name and name not in seen:
                seen.add(name)
                tools.append({
                    "name": name,
                    "description": fn.get("description", ""),
                    "source": "mcp",
                })
    except Exception as e:
        logger.warning(f"获取 MCP 工具列表失败: {e}")
    return {"tools": tools}


def _normalize_card_tts(payload_tts: dict) -> dict:
    """归一化卡片 TTS 字段，缺省沿用当前全局 TTS 配置。"""
    return {
        "engine": payload_tts.get("engine", tts_config.get("engine", "edge_tts")),
        "edge_voice": payload_tts.get("edge_voice", tts_config.get("edge_voice", "")),
        "edge_rate": payload_tts.get("edge_rate", tts_config.get("edge_rate", "+8%")),
        "gptsovits_url": payload_tts.get("gptsovits_url", tts_config.get("gptsovits_url", "")),
        "gptsovits_ref_audio": payload_tts.get("gptsovits_ref_audio", tts_config.get("gptsovits_ref_audio", "")),
        "gptsovits_character": payload_tts.get("gptsovits_character", tts_config.get("gptsovits_character", "")),
    }


def _normalize_card_tools(payload_tools: dict) -> dict:
    """归一化卡片工具配置：是否启用 + 工具白名单（空列表=全部可用）。"""
    return {
        "enabled": bool(payload_tools.get("enabled", True)),
        "allowed": list(payload_tools.get("allowed", []) or []),
    }


def _normalize_card_llm(payload_llm: dict) -> dict:
    """归一化卡片 LLM 字段：空串 = 未配置，应用时沿用全局默认配置。"""
    payload_llm = payload_llm or {}
    return {
        "model": (payload_llm.get("model") or "").strip(),
        "base_url": (payload_llm.get("base_url") or "").strip(),
        "api_key": (payload_llm.get("api_key") or "").strip(),
    }


@app.post("/api/character_cards")
async def character_cards_create(payload: dict):
    """创建角色卡片。"""
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "卡片名称不能为空")
    card = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "role_name": (payload.get("role_name") or "").strip(),
        "user_name": (payload.get("user_name") or "").strip(),
        "model_url": payload.get("model_url", ""),
        "model_name": payload.get("model_name", ""),
        "system_prompt": payload.get("system_prompt", ""),
        "tts": _normalize_card_tts(payload.get("tts", {})),
        "tools": _normalize_card_tools(payload.get("tools", {})),
        "llm": _normalize_card_llm(payload.get("llm", {})),
        "created_at": int(time.time()),
    }
    cards = _load_character_cards()
    cards.insert(0, card)
    _save_character_cards(cards)
    return {"ok": True, "card": card}


@app.put("/api/character_cards/{card_id}")
async def character_cards_update(card_id: str, payload: dict):
    """更新角色卡片（局部字段）。"""
    cards = _load_character_cards()
    for card in cards:
        if card.get("id") != card_id:
            continue
        if "name" in payload:
            card["name"] = (payload.get("name") or "").strip()
        if "role_name" in payload:
            card["role_name"] = (payload.get("role_name") or "").strip()
        if "user_name" in payload:
            card["user_name"] = (payload.get("user_name") or "").strip()
        if "model_url" in payload:
            card["model_url"] = payload.get("model_url", "")
        if "model_name" in payload:
            card["model_name"] = payload.get("model_name", "")
        if "system_prompt" in payload:
            card["system_prompt"] = payload.get("system_prompt", "")
        if "tts" in payload:
            card["tts"] = _normalize_card_tts({**card.get("tts", {}), **payload["tts"]})
        if "tools" in payload:
            card["tools"] = _normalize_card_tools({**card.get("tools", {}), **payload["tools"]})
        if "llm" in payload:
            card["llm"] = _normalize_card_llm({**card.get("llm", {}), **payload["llm"]})
        _save_character_cards(cards)
        return {"ok": True, "card": card}
    raise HTTPException(404, "卡片不存在")


@app.delete("/api/character_cards/{card_id}")
async def character_cards_delete(card_id: str):
    """删除角色卡片。"""
    cards = _load_character_cards()
    remain = [c for c in cards if c.get("id") != card_id]
    if len(remain) == len(cards):
        raise HTTPException(404, "卡片不存在")
    _save_character_cards(remain)
    # 若删除的是当前活动卡片，清除活动标记（记忆空间回到 default）
    try:
        with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        if _cfg.get("active_role_card") == card_id:
            _cfg["active_role_card"] = ""
            with open(BASE_DIR / "settings.json", "w", encoding="utf-8") as f:
                json.dump(_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"清除活动角色卡片标记失败: {e}")
    return {"ok": True}


@app.post("/api/character_cards/{card_id}/apply")
async def character_cards_apply(card_id: str):
    """应用角色卡片：更新角色名/系统提示词（settings.json）+ TTS 配置（tts_config.json）。"""
    cards = _load_character_cards()
    card = next((c for c in cards if c.get("id") == card_id), None)
    if not card:
        raise HTTPException(404, "卡片不存在")
    # 1. 更新 settings.json 角色名 + 系统提示词 + 用户称呼 + 工具配置（agent 每次对话动态读取，即时生效）
    _save_role_config(
        (card.get("role_name") or card.get("name") or "AI助手").strip(),
        card.get("system_prompt", ""),
        (card.get("user_name") or "").strip(),
        card.get("tools"),
    )
    # 2. 更新 TTS 配置
    tts = card.get("tts", {})
    for k in ("engine", "edge_voice", "edge_rate",
              "gptsovits_url", "gptsovits_ref_audio", "gptsovits_character"):
        if k in tts:
            tts_config[k] = tts[k]
    if tts.get("edge_voice"):
        global _edge_voices_cache
        _edge_voices_cache = None
    _save_tts_config({k: tts_config[k] for k in DEFAULT_TTS_CONFIG})
    # 3. 更新 LLM 配置（settings.json + 运行时 agent 客户端），未配置字段沿用全局默认
    llm = card.get("llm") or {}
    llm_model = (llm.get("model") or "").strip()
    llm_base_url = (llm.get("base_url") or "").strip()
    llm_api_key = (llm.get("api_key") or "").strip()
    if llm_model or llm_base_url or llm_api_key:
        try:
            with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
                _cfg = json.load(f)
            if llm_model:
                _cfg["model"] = llm_model
            if llm_base_url:
                _cfg["base_url"] = llm_base_url
            if llm_api_key:
                _cfg["api_key"] = llm_api_key
            with open(BASE_DIR / "settings.json", "w", encoding="utf-8") as f:
                json.dump(_cfg, f, ensure_ascii=False, indent=2)
            agent = await get_shared_agent()
            await agent.reload_llm_config()
        except Exception as e:
            logger.warning(f"应用角色卡片 LLM 配置失败: {e}")
    # 4. 持久化当前活动角色卡片（记忆命名空间按卡片隔离）
    try:
        with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        _cfg["active_role_card"] = card_id
        with open(BASE_DIR / "settings.json", "w", encoding="utf-8") as f:
            json.dump(_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存活动角色卡片失败: {e}")
    # 5. 切换到该卡片的独立记忆空间（避免跨角色记忆混淆），返回该卡片对应的会话
    session_id = None
    try:
        agent = await get_shared_agent()
        session_id = await agent.set_role_card_namespace(card_id)
    except Exception as e:
        logger.warning(f"切换角色卡片记忆空间失败: {e}")
    logger.info(f"应用角色卡片: {card.get('name')}")
    return {
        "ok": True,
        "card": card,
        "model_url": card.get("model_url", ""),
        "model_name": card.get("model_name", ""),
        "session_id": session_id,
    }


# ---------- 游戏多角色台词（赛博公司等） ----------
@app.post("/api/game/speak")
async def game_speak(req: Request):
    """为游戏中的某个角色生成一句台词（LLM 人设直呼）并合成该角色音色的语音。

    请求体:
    {
      "system_prompt": "角色人设（卡片 system_prompt + 游戏角色指令）",
      "context":       "当前语境（面试记录 / 需要回应的话）",
      "text":          "可选。若提供，则跳过 LLM，直接用该文本合成语音（纯 TTS 模式）",
      "voice":         "该角色 TTS 音色 ShortName（edge_tts）",
      "rate":          "语速，如 +8%",
      "max_tokens":    可选，默认 220
    }
    返回:
    { "ok": true, "text": "...", "voice": "...", "audio_base64": "...", "mime": "audio/mpeg" }
    音频字段仅在 voice 有效且合成成功时返回，否则前端只展示文本。
    """
    body = await req.json()
    system_prompt = str(body.get("system_prompt", "") or "").strip()
    context = str(body.get("context", "") or "").strip()
    direct_text = str(body.get("text", "") or "").strip()
    if not system_prompt and not direct_text:
        return JSONResponse({"ok": False, "error": "缺少 system_prompt 或 text"})
    voice = str(body.get("voice", "") or "").strip()
    rate = str(body.get("rate", "") or "+8%")
    max_tokens = int(body.get("max_tokens", 220) or 220)

    if direct_text:
        # 纯 TTS 模式：跳过 LLM，直接对给定文本合成语音（回退脚本也能有声音）
        text = direct_text
    else:
        agent = await get_shared_agent()
        text = await agent.generate_character_line(system_prompt, context, max_tokens)
        if not text:
            return JSONResponse({"ok": False, "error": "台词生成失败"})

    # 为该角色合成语音（edge_tts，不影响全局 tts_config）
    audio_base64 = None
    mime = None
    if voice:
        try:
            audio_data, mime = await generate_tts_edge(text, voice, rate)
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode("ascii")
        except Exception as e:
            logger.warning(f"游戏角色 TTS 合成失败: {e}")

    resp = {"ok": True, "text": text, "voice": voice}
    if audio_base64 and mime:
        resp["audio_base64"] = audio_base64
        resp["mime"] = mime
    return JSONResponse(resp)


# ---------- 3D 模型管理 ----------
@app.get("/api/models")
async def list_models():
    """列出所有已上传的 3D 模型"""
    models = []
    for f in MODELS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_MODEL_EXTS:
            stat = f.stat()
            models.append({
                "name": f.name,
                "url": f"/models/{f.name}",
                "size": stat.st_size,
                "type": f.suffix.lower().lstrip("."),
                "mtime": int(stat.st_mtime),
            })
    models.sort(key=lambda m: m["mtime"], reverse=True)
    return {"models": models}


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """上传 3D 模型文件 (glb/gltf/vrm)"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, f"不支持的格式 {ext}，仅支持 {', '.join(ALLOWED_MODEL_EXTS)}")

    # 读取内容并检查大小
    data = await file.read()
    if len(data) > MAX_MODEL_SIZE:
        raise HTTPException(413, f"文件过大 ({len(data)//1024//1024}MB)，上限 {MAX_MODEL_SIZE//1024//1024}MB")

    # 重命名避免覆盖与冲突：保留原名但加时间戳前缀
    safe_name = re.sub(r"[^\w.\-]", "_", file.filename)
    target = MODELS_DIR / f"{int(asyncio.get_event_loop().time() * 1000)}_{safe_name}"
    target.write_bytes(data)
    return {
        "name": target.name,
        "url": f"/models/{target.name}",
        "size": len(data),
        "type": ext.lstrip("."),
    }


@app.delete("/api/model/{name}")
async def delete_model(name: str):
    """删除指定模型"""
    # 防路径穿越
    safe = Path(name).name
    target = MODELS_DIR / safe
    if not target.is_file():
        raise HTTPException(404, "模型不存在")
    if target.suffix.lower() not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, "非模型文件")
    target.unlink()
    return {"deleted": safe}


@app.put("/api/model/{name}/rename")
async def rename_model(name: str, payload: dict):
    """重命名指定模型（保留扩展名）"""
    new_name = (payload.get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(400, "新名称不能为空")
    # 防路径穿越
    safe_old = Path(name).name
    target = MODELS_DIR / safe_old
    if not target.is_file():
        raise HTTPException(404, "模型不存在")
    if target.suffix.lower() not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, "非模型文件")
    # 保留原扩展名
    old_suffix = target.suffix
    safe_new = re.sub(r"[^\w.\-]", "_", new_name)
    if not safe_new.lower().endswith(old_suffix.lower()):
        safe_new += old_suffix
    new_path = MODELS_DIR / safe_new
    if new_path.exists():
        raise HTTPException(409, "同名文件已存在")
    target.rename(new_path)
    return {"old_name": safe_old, "new_name": safe_new, "url": f"/models/{safe_new}"}


# ---------- 3D 背景模型管理 ----------
@app.get("/api/backgrounds")
async def list_backgrounds():
    """列出所有已上传的 3D 背景模型"""
    items = []
    for f in BACKGROUNDS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_MODEL_EXTS:
            stat = f.stat()
            items.append({
                "name": f.name,
                "url": f"/backgrounds/{f.name}",
                "size": stat.st_size,
                "type": f.suffix.lower().lstrip("."),
                "mtime": int(stat.st_mtime),
            })
    items.sort(key=lambda m: m["mtime"], reverse=True)
    return {"backgrounds": items}


@app.post("/api/background/upload")
async def upload_background(file: UploadFile = File(...)):
    """上传 3D 背景模型文件 (glb/gltf/vrm)"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, f"不支持的格式 {ext}，仅支持 {', '.join(ALLOWED_MODEL_EXTS)}")

    data = await file.read()
    if len(data) > MAX_MODEL_SIZE:
        raise HTTPException(413, f"文件过大 ({len(data)//1024//1024}MB)，上限 {MAX_MODEL_SIZE//1024//1024}MB")

    safe_name = re.sub(r"[^\w.\-]", "_", file.filename)
    target = BACKGROUNDS_DIR / f"{int(asyncio.get_event_loop().time() * 1000)}_{safe_name}"
    target.write_bytes(data)
    return {
        "name": target.name,
        "url": f"/backgrounds/{target.name}",
        "size": len(data),
        "type": ext.lstrip("."),
    }


@app.delete("/api/background/{name}")
async def delete_background(name: str):
    """删除指定背景模型"""
    safe = Path(name).name
    target = BACKGROUNDS_DIR / safe
    if not target.is_file():
        raise HTTPException(404, "背景不存在")
    if target.suffix.lower() not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, "非模型文件")
    target.unlink()
    return {"deleted": safe}


@app.put("/api/background/{name}/rename")
async def rename_background(name: str, payload: dict):
    """重命名指定背景模型（保留扩展名）"""
    new_name = (payload.get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(400, "新名称不能为空")
    safe_old = Path(name).name
    target = BACKGROUNDS_DIR / safe_old
    if not target.is_file():
        raise HTTPException(404, "背景不存在")
    if target.suffix.lower() not in ALLOWED_MODEL_EXTS:
        raise HTTPException(400, "非模型文件")
    old_suffix = target.suffix
    safe_new = re.sub(r"[^\w.\-]", "_", new_name)
    if not safe_new.lower().endswith(old_suffix.lower()):
        safe_new += old_suffix
    new_path = BACKGROUNDS_DIR / safe_new
    if new_path.exists():
        raise HTTPException(409, "同名文件已存在")
    target.rename(new_path)
    return {"old_name": safe_old, "new_name": safe_new, "url": f"/backgrounds/{safe_new}"}


# ---------- BGM 背景音乐管理 ----------
ALLOWED_BGM_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
BGM_MAX_SIZE = 50 * 1024 * 1024  # 50MB


@app.get("/api/bgm")
async def list_bgm():
    """列出所有背景音乐文件"""
    items = []
    for f in BGM_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_BGM_EXTS:
            stat = f.stat()
            items.append({
                "name": f.name,
                "url": f"/bgm/{f.name}",
                "size": stat.st_size,
                "type": f.suffix.lower().lstrip("."),
                "mtime": int(stat.st_mtime),
            })
    items.sort(key=lambda m: m["mtime"], reverse=True)
    return {"bgm": items}


@app.post("/api/bgm/upload")
async def upload_bgm(file: UploadFile = File(...)):
    """上传背景音乐文件"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_BGM_EXTS:
        raise HTTPException(400, f"不支持的格式 {ext}，仅支持 {', '.join(ALLOWED_BGM_EXTS)}")

    data = await file.read()
    if len(data) > BGM_MAX_SIZE:
        raise HTTPException(413, f"文件过大 ({len(data)//1024//1024}MB)，上限 {BGM_MAX_SIZE//1024//1024}MB")

    safe_name = re.sub(r"[^\w.\-]", "_", file.filename)
    target = BGM_DIR / f"{int(asyncio.get_event_loop().time() * 1000)}_{safe_name}"
    target.write_bytes(data)
    return {
        "name": target.name,
        "url": f"/bgm/{target.name}",
        "size": len(data),
        "type": ext.lstrip("."),
    }


@app.delete("/api/bgm/{name}")
async def delete_bgm(name: str):
    """删除指定背景音乐"""
    safe = Path(name).name
    target = BGM_DIR / safe
    if not target.is_file():
        raise HTTPException(404, "BGM不存在")
    if target.suffix.lower() not in ALLOWED_BGM_EXTS:
        raise HTTPException(400, "非音频文件")
    target.unlink()
    return {"deleted": safe}


# ---------- Agent & 记忆管理 API ----------
@app.get("/api/config/user_name")
async def user_name_get():
    """获取用户设置的称呼（AI 对用户的称呼，空 = 未设置）。"""
    try:
        cfg = load_config()
        return {"user_name": cfg.get("user_name", "")}
    except Exception as e:
        return {"user_name": "", "error": str(e)}


@app.post("/api/config/user_name")
async def user_name_set(payload: dict):
    """保存用户设置的称呼到 settings.json（空串 = 清除，AI 不随意称呼）。"""
    name = (payload.get("user_name") or "").strip()
    path = BASE_DIR / "settings.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["user_name"] = name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info(f"用户称呼设置已更新: '{name}'")
        return {"ok": True, "user_name": name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/agent/status")
async def agent_status():
    """获取 Agent 状态（已加载的 MCP 工具数等）。"""
    try:
        agent = await get_shared_agent()
        tools_count = len(agent._all_tools) if agent._all_tools else 0
        return {
            "initialized": agent._initialized,
            "tools_count": tools_count,
            "mcp_connected": agent.mcp_manager is not None,
            "has_memory": agent.memory is not None,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sessions")
async def list_sessions(user_id: str = "default"):
    """列出当前角色卡片记忆空间下用户的所有会话。"""
    agent = await get_shared_agent(user_id)
    await agent.sync_memory_namespace()
    sessions = await agent.get_sessions()
    return {"sessions": sessions}


@app.post("/api/sessions/switch")
async def switch_session(payload: dict):
    """切换到指定会话（仅限当前角色卡片记忆空间内的会话）。"""
    user_id = payload.get("user_id", "default")
    session_id = payload.get("session_id", "")
    if not session_id:
        raise HTTPException(400, "缺少 session_id")
    agent = await get_shared_agent(user_id)
    await agent.sync_memory_namespace()
    if not await agent.memory.session_belongs_to_namespace(session_id):
        raise HTTPException(400, "该会话不属于当前角色卡片")
    await agent.switch_session(session_id)
    history = await agent.get_history()
    return {"session_id": session_id, "history": history}


@app.post("/api/sessions/new")
async def new_session(payload: dict):
    """创建新会话。"""
    user_id = payload.get("user_id", "default")
    agent = await get_shared_agent(user_id)
    await agent.close_current_session()
    # 先同步当前角色卡片记忆空间，再创建新会话
    await agent.sync_memory_namespace()
    agent.memory.session_id = None
    sid = await agent.memory.get_or_create_session()
    return {"session_id": sid}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = "default"):
    """删除指定会话。"""
    agent = await get_shared_agent(user_id)
    await agent.delete_session(session_id)
    return {"deleted": session_id}


# ---------- WebSocket ----------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)


manager = ConnectionManager()


async def safe_send_json(ws: WebSocket, data: dict) -> bool:
    """安全发送 WebSocket 消息，连接已关闭时静默忽略。"""
    try:
        if ws.client_state != WebSocketState.CONNECTED:
            print(f"[WS] 发送失败: 连接状态={ws.client_state}, msg_type={data.get('type')}")
            return False
        await ws.send_json(data)
        return True
    except (RuntimeError, Exception) as e:
        print(f"[WS] 发送异常: {type(e).__name__}, msg_type={data.get('type')}")
        return False


# 全局 Agent 实例（所有连接共享，按 user_id 区分记忆；游戏/非游戏模式共用同一实例）
_shared_agents: dict[str, AIAgent] = {}
_agent_lock = asyncio.Lock()


async def get_shared_agent(user_id: str = "default") -> AIAgent:
    """获取或创建共享 Agent 实例。"""
    if user_id not in _shared_agents:
        async with _agent_lock:
            if user_id not in _shared_agents:
                agent = AIAgent(user_id=user_id)
                await agent.initialize()
                _shared_agents[user_id] = agent
                logger.info(f"创建 Agent: user_id={user_id}")
    return _shared_agents[user_id]


# 全局奖励记忆库（LLM-as-policy 用，跨连接共享）
_shared_reward_memory: Optional[RewardMemory] = None


def get_reward_memory() -> RewardMemory:
    """获取或创建共享奖励记忆库实例（单例）。"""
    global _shared_reward_memory
    if _shared_reward_memory is None:
        _shared_reward_memory = RewardMemory()
        logger.info(f"创建 RewardMemory: {_shared_reward_memory.stats()}")
    return _shared_reward_memory


# ---------- 会话状态（per-connection） ----------
class WSState:
    """每个 WebSocket 连接的会话状态，支持打断式流式响应。"""
    def __init__(self):
        self.current_session: Optional[str] = None   # 当前活跃回复 session_id
        self.cancelled: set = set()                  # 已取消的回复 session_id 集合
        self.active_task: Optional[asyncio.Task] = None  # 当前活跃回复 task
        self.user_id: str = "default"                # 用户标识
        self.chat_session_id: Optional[str] = None   # 当前对话会话 ID（持久化用）
        self.current_model: Optional[str] = None     # 当前使用的角色模型名
        self.current_background: Optional[str] = None  # 当前使用的背景场景名
        self.current_bgm: Optional[str] = None       # 当前播放的背景音乐名
        self.game_engine: GameEngine = GameEngine()  # 游戏引擎实例
        self.perception_dispatcher = PerceptionDispatcher()  # 统一感知路由调度器
        self.last_response_done: float = 0           # 上次回复完成时间戳（被动事件冷却用）
        self.last_user_message_time: float = time.time()  # 用户最后消息时间（AI 自主系统用）
        self.ai_is_moving: bool = False              # AI 是否正在自主移动
        # STT 失败冷却：连续失败 N 次后暂停接收音频，避免反复消耗资源 + 误抑制 AI 自主行为
        self._stt_fail_count: int = 0               # 连续失败计数
        self._stt_fail_window_start: float = 0      # 计数窗口起点
        self._stt_cooldown_until: float = 0         # 冷却截止时间
        # LLM-as-policy：上次决策待结算（等奖励回传时存入记忆库）
        self._pending_llm_action: Optional[dict] = None
        # ── RL 统一关系状态（非游戏模式也纳入 RL 统摄） ──
        self.affection: float = 50.0     # 好感度 [0,100]
        self.trust: float = 30.0         # 信任度 [0,100]
        self.intimacy: float = 20.0      # 亲密度 [0,100]
        self.user_emotion: int = 0       # 0积极 1中性 2消极 3愤怒 4孤独
        self.proactive_successes: int = 0
        self.proactive_attempts: int = 0
        # 全局主动说话闸门：所有非用户消息驱动的主动说话共用此时间戳冷却
        self._last_proactive_speak: float = 0
        # RL 决策的快照间隔（秒），由快照间隔控制器学习得出（大厅/游戏共用）
        self._current_snapshot_interval: float = 30.0

    def new_session(self) -> str:
        """开启新回复会话：把旧会话标记为取消，返回新 session_id。"""
        if self.current_session:
            self.cancelled.add(self.current_session)
        sid = uuid.uuid4().hex
        self.current_session = sid
        return sid

    def is_cancelled(self, sid: str) -> bool:
        return sid in self.cancelled

    def cancel_current(self):
        if self.current_session:
            self.cancelled.add(self.current_session)
            self.current_session = None


async def handle_user_message_stream(ws: WebSocket, user_text: str, history: list,
                                      session_id: str, state: WSState,
                                      current_model: Optional[str] = None,
                                      current_background: Optional[str] = None,
                                      current_bgm: Optional[str] = None,
                                      game_context: Optional[str] = None,
                                      record_history: bool = True,
                                      msg_source: str = "chat"):
    """流式处理：AI 流式响应（含工具调用）→ 按句切分 → 每句生成 TTS → 推送 audio_chunk。

    支持：
    - MCP 工具调用状态实时推送
    - 长期记忆自动保存
    - 首句延迟 ≈ AI 首句生成时间 + 单句 TTS 时间（通常 1~2s）

    Args:
        record_history: 是否记录到短期记忆（history）。仅用户直接输入应记录，
            环境交互（感知触发/自主行为等）不进短期记忆，避免重复内容导致思维僵化。
        msg_source: 消息来源标记，写入长期记忆的 source 字段：
            'chat'=用户直接输入（大厅）、'game'=用户直接输入（游戏）、
            'auto'=环境交互（由记忆系统处理）。
    """
    if not user_text.strip():
        return
    await safe_send_json(ws, {"type": "thinking", "session_id": session_id})

    buffer = ""
    full_text = ""
    seq = 0
    SOFT_CUT_LEN = 24
    tool_calls_made = []  # 记录本轮调用的工具

    async def flush(sentence: str):
        """生成单句 TTS 并推送（音频数据 base64 编码，即时销毁）。"""
        nonlocal seq
        sentence = re.sub(r"[*_`#>\-\[\]\(\)]", "", sentence).strip()
        if not sentence or len(sentence) < 2:
            return
        if state.is_cancelled(session_id):
            return
        seq += 1
        audio_b64 = None
        audio_mime = None
        try:
            result = await generate_tts(sentence)
            if result:
                audio_bytes, mime_type = result
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_mime = mime_type
                del audio_bytes  # 释放内存
        except Exception as e:
            print(f"[TTS] 失败: {e}")
        if state.is_cancelled(session_id):
            return
        await safe_send_json(ws, {
            "type": "audio_chunk",
            "session_id": session_id,
            "seq": seq,
            "text": sentence,
            "audio_b64": audio_b64,
            "audio_mime": audio_mime,
            "final": False,
        })

    try:
        # 统一使用共享 Agent（游戏/非游戏模式共用同一实例，通过 game_mode 区分行为）
        agent = await get_shared_agent(state.user_id)
        # 将记忆绑定到当前活动角色卡片的命名空间（跨卡片记忆隔离）；
        # state 中的会话若不属于当前命名空间（如切卡后未刷新）则改用命名空间当前会话
        if agent.memory:
            await agent.sync_memory_namespace()
            if state.chat_session_id and await agent.memory.session_belongs_to_namespace(state.chat_session_id):
                agent.memory.session_id = state.chat_session_id
            else:
                state.chat_session_id = agent.memory.session_id

        # 游戏模式：同步记忆到游戏引擎（用于持久化游戏事件），并获取游戏类型
        game_mode = bool(state.game_engine and state.game_engine.active)
        game_type = None
        if game_mode:
            if agent.memory and state.game_engine:
                state.game_engine.memory = agent.memory
            if state.game_engine and state.game_engine.world:
                game_type = state.game_engine.world.game_key or None
        # 游戏模式下用户直接输入沿用 'game' 来源标记（原行为），环境交互保持 'auto'
        if msg_source == "chat" and game_mode:
            msg_source = "game"

        async for event in agent.chat_stream(
            user_text, history=history,
            current_model=current_model,
            current_background=current_background,
            current_bgm=current_bgm,
            game_context=game_context,
            game_mode=game_mode,
            game_type=game_type,
            msg_source=msg_source,
        ):
            if state.is_cancelled(session_id):
                break

            if isinstance(event, TextDelta):
                buffer += event.text
                full_text += event.text
                # 按句末标点切分
                while True:
                    m = SENTENCE_END.search(buffer)
                    if m:
                        sentence = buffer[:m.end()]
                        buffer = buffer[m.end():]
                        await flush(sentence)
                    elif len(buffer) >= SOFT_CUT_LEN:
                        await flush(buffer)
                        buffer = ""
                        break
                    else:
                        break

            elif isinstance(event, ToolCallStart):
                tool_calls_made.append({"name": event.tool_name, "arguments": event.arguments})
                await safe_send_json(ws, {
                    "type": "tool_call_start",
                    "session_id": session_id,
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                })

            elif isinstance(event, ToolCallResult):
                await safe_send_json(ws, {
                    "type": "tool_call_result",
                    "session_id": session_id,
                    "tool_name": event.tool_name,
                    "result": event.result[:500],
                    "success": event.success,
                })
                # 检测是否为屏幕控制工具，若是则转发给前端执行
                if event.success:
                    try:
                        r = json.loads(event.result)
                        if isinstance(r, dict) and r.get("__screen_command__"):
                            cmd = {"type": "screen_command", "tool": r["tool"], "args": r["args"]}
                            await safe_send_json(ws, cmd)
                            logger.info(f"[ScreenCmd] 已发送: {r['tool']} {r['args']}")
                    except (json.JSONDecodeError, TypeError):
                        pass  # 不是 JSON，忽略

    except Exception as e:
        logger.error(f"Agent 错误: {e}")
        await safe_send_json(ws, {"type": "error", "message": f"AI 出错了：{e}",
                            "session_id": session_id})
        state.last_response_done = time.time()
        return

    # 处理剩余尾巴
    if buffer.strip():
        await flush(buffer)

    if state.is_cancelled(session_id):
        await safe_send_json(ws, {"type": "interrupted", "session_id": session_id})
        state.last_response_done = time.time()
        return

    full_text = full_text.strip()
    await safe_send_json(ws, {
        "type": "audio_end",
        "session_id": session_id,
        "full_text": full_text,
        "tool_calls": tool_calls_made if tool_calls_made else None,
    })
    state.last_response_done = time.time()
    # 短期记忆（history）只记录用户直接输入；环境交互由记忆系统处理
    if full_text and record_history:
        history.append({"user": user_text, "ai": full_text})
        # 防止长时间会话 history 无限增长导致内存泄漏（LLM 只用最后 10 条）
        if len(history) > 200:
            history[:] = history[-100:]


async def _kickoff_response(ws: WebSocket, text: str, history: list, state: WSState,
                           game_context: Optional[str] = None,
                           allow_interrupt: bool = False,
                           proactive: bool = False,
                           record_history: Optional[bool] = None,
                           msg_source: str = "chat"):
    """启动流式回复 task。

    allow_interrupt=True 时取消当前回复（用户文字/语音输入）；
    allow_interrupt=False 时若AI正在说话则直接忽略（被动事件）。

    proactive=True 表示非用户消息驱动的主动说话（RL调度/感知派发/
    环境快照等）——所有主动路径共用全局闸门 _last_proactive_speak，
    防止任何来源的"自言自语"式高频说话。返回 True 表示已启动回复。

    record_history: 是否记录到短期记忆。None 时默认 not proactive
        （主动说话=环境交互，不记录）；环境类调用点可显式传 False。
    msg_source: 写入长期记忆的来源标记（'chat'/'game'/'auto'）。
    """
    # 记录 AI 输出时间（用户驱动回复与自主输出均记录）。
    # RL 结算据此判定"用户是否回应了本次输出"：只有用户消息发生在
    # 输出之后才算回应 —— AI 自身动作触发的话不算用户输入。
    try:
        from rl_coordinator import get_coordinator
        get_coordinator().note_agent_output()
    except Exception:
        pass
    # 全局主动说话闸门：任何主动路径（非用户消息）都受此冷却约束
    if proactive:
        if state.active_task is not None and not state.active_task.done():
            return False  # AI 正在说话，主动路径不得打断
        last_speak = getattr(state, "_last_proactive_speak", 0.0)
        if time.time() - last_speak < ACTIVE_SPEAK_COOLDOWN:
            return False  # 全局冷却期内，主动路径一律拒绝
    if not allow_interrupt:
        if state.active_task is not None and not state.active_task.done():
            return False  # AI正在说话，被动事件不得打断
        if time.time() - state.last_response_done < 3.0:
            return False  # 回复刚结束冷却期，被动事件不得打断
    # 取消正在进行的回复（如果有）
    state.cancel_current()
    if state.active_task and not state.active_task.done():
        state.active_task.cancel()
    if proactive:
        state._last_proactive_speak = time.time()  # 记录全局主动说话时间
    sid = state.new_session()
    state.active_task = asyncio.create_task(
        handle_user_message_stream(
            ws, text, history, sid, state,
            current_model=state.current_model,
            current_background=state.current_background,
            current_bgm=state.current_bgm,
            game_context=game_context,
            record_history=(not proactive) if record_history is None else record_history,
            msg_source=msg_source,
        )
    )
    return True


async def _apply_dispatch_result(ws: WebSocket, result, history: list, state: WSState):
    """统一应用感知调度结果。

    由 PerceptionDispatcher.dispatch() 返回的 DispatchResult 驱动：
    - behavior_cmd → 发送给前端执行
    - trigger_text + should_speak → 启动 LLM 回复
    """
    if result.behavior_cmd:
        await safe_send_json(ws, result.behavior_cmd)
    if result.should_speak and result.trigger_text:
        await _kickoff_response(
            ws, result.trigger_text, history, state,
            game_context=result.game_context,
            proactive=True,  # 感知派发属于主动说话，走全局闸门
            msg_source="auto",  # 环境交互：由记忆系统处理，不进短期记忆
        )


async def _handle_game_action_request(ws: WebSocket, data: dict, state: WSState):
    """处理 LLM-as-policy 决策请求：RL协调器统摄检索与结算。

    流程（RL 统摄架构）：
    1. 协调器结算上次 pending action 的延迟奖励（统一经验记忆库）
    2. 以统一状态检索示例（同状态优先 + 跨模式迁移兜底）
    3. 调用统一 Agent 的 decide_action 让 LLM 选择宏观策略
    4. 记录本次决策为 pending（等奖励回传时结算）
    """
    try:
        coord = get_coordinator()

        # 1. 结算上一次决策的延迟奖励
        last_reward = data.get("last_reward")
        if last_reward is not None and state._pending_llm_action:
            pa = state._pending_llm_action
            coord.record_reward(pa["state_key"], pa["strategy"], float(last_reward))
            state._pending_llm_action = None

        # 2. 构建统一状态并检索示例（含跨模式迁移）
        state_key = data.get("state_key", "")
        unified = coord.build_state(
            engine=state.game_engine,
            user_engaged=(time.time() - state.last_user_message_time) < 120,
            seconds_since_user_message=max(0, time.time() - state.last_user_message_time),
            seconds_since_interaction=max(0, time.time() - state.last_response_done),
            affection=state.affection, trust=state.trust, intimacy=state.intimacy,
            user_emotion=state.user_emotion,
            proactive_success_rate=(state.proactive_successes / state.proactive_attempts)
            if state.proactive_attempts else 0.0,
        )
        examples = coord.get_examples(unified, top_k=4)
        if not examples and state_key:
            examples = coord.experience.retrieve(state_key)[:4]

        # 3. LLM 决策（统一 Agent）
        # 游戏类型优先从前端请求取，兜底从游戏引擎 world 中获取
        game_type = data.get("game_type")
        if not game_type and state.game_engine and state.game_engine.world:
            game_type = state.game_engine.world.game_key or None
        agent = await get_shared_agent(state.user_id)
        result = await agent.decide_action(
            state_text=data.get("state_text", "") or unified.to_prompt_text(),
            state_key=state_key or unified.to_state_key(),
            candidates=data.get("candidates", []),
            examples=examples,
            game_type=game_type,
        )

        # 4. 记录待结算
        if result and result.get("strategy"):
            state._pending_llm_action = {
                "state_key": state_key or unified.to_state_key(),
                "strategy": result["strategy"],
                "ts": time.time(),
            }

        await safe_send_json(ws, {"type": "game_action_response", "data": result or {}})
    except Exception as e:
        logger.warning(f"game_action_request 处理失败: {e}")
        await safe_send_json(ws, {"type": "game_action_response", "data": {}})


async def _handle_rl_decision(ws: WebSocket, data: dict, state: WSState, history: list = None):
    """统一调度决策入口 —— RL 统摄派发所有 Agent（游戏 + 非游戏）。

    流程：
    1. 构建统一状态
    2. coord.schedule() 决策：Agent路由（game_agent / ai_agent / engagement / silence）
    3. 执行派发：
       - game_agent  → 补全 LLM 宏观策略（统一 Agent.decide_action）
       - ai_agent    → 回传主动说话触发（前端 sendAIAction 触发 AIAgent 回复）
       - engagement  → 回传行为指令（行为引擎）
       - silence     → 不派发
    4. 回传 rl_dispatch 计划给前端执行
    """
    from rl_coordinator import AgentChoice
    try:
        coord = get_coordinator()

        # 1. 构建统一状态
        event = data.get("event", "proactive_tick")
        unified = coord.build_state(
            engine=state.game_engine,
            user_engaged=(time.time() - state.last_user_message_time) < 120,
            seconds_since_user_message=max(0, time.time() - state.last_user_message_time),
            seconds_since_interaction=max(0, time.time() - state.last_response_done),
            affection=state.affection, trust=state.trust, intimacy=state.intimacy,
            user_emotion=state.user_emotion,
            proactive_success_rate=(state.proactive_successes / state.proactive_attempts)
            if state.proactive_attempts else 0.0,
            last_user_message_ts=state.last_user_message_time,
        )
        # 前端回传覆盖（rl_sync 已同步过关系状态）
        if "affection" in data: unified.affection = float(data["affection"])
        if "emotion" in data: unified.user_emotion = int(data["emotion"])
        if "game_state" in data: unified.game_state = data["game_state"]

        # 2. RL 统一调度（传入 game_engine：engagement 分支生成真实行为指令）
        forced = data.get("forced_agent")
        plan = coord.schedule(unified, forced_agent=forced, event=event,
                              engine=state.game_engine)

        # 3. 派发执行
        if plan.agent_choice == AgentChoice.GAME_AGENT:
            # 补全游戏宏观策略（LLM-as-policy）
            try:
                examples = coord.get_examples(unified, top_k=4)
                game_type = None
                if state.game_engine and state.game_engine.world:
                    game_type = state.game_engine.world.game_key or None
                agent = await get_shared_agent(state.user_id)
                result = await agent.decide_action(
                    state_text=unified.to_prompt_text(),
                    state_key=unified.to_state_key(),
                    candidates=data.get("candidates", []),
                    examples=examples,
                    game_type=game_type,
                )
                if result and result.get("strategy"):
                    plan.strategy = result["strategy"]
                    state._pending_llm_action = {
                        "state_key": unified.to_state_key(),
                        "strategy": result["strategy"],
                        "ts": time.time(),
                    }
                    # P1-5：LLM 宏观策略 → 软引导目标（R_guide），
                    # 供下一次结算把意图注入 UnifiedRewardFunction.compute
                    try:
                        gt = coord.extract_guide_target(result, unified)
                        if gt:
                            coord.set_guide_target(gt)
                    except Exception as ge:
                        logger.warning(f"rl_decision 引导目标解析失败: {ge}")
            except Exception as e:
                logger.warning(f"rl_decision 游戏策略补全失败: {e}")

        elif plan.agent_choice == AgentChoice.AI_AGENT:
            # 主动说话触发（非打断；AI 空闲时才执行）
            # 全局主动说话闸门由 _kickoff_response(proactive=True) 统一管控
            if plan.speak_text:
                await _kickoff_response(
                    ws, plan.speak_text, history or [], state,
                    game_context=state.game_engine.get_game_context_for_ai()
                    if state.game_engine else None,
                    proactive=True,
                    msg_source="auto",  # RL 主动说话：环境交互，不进短期记忆
                )

        # silence / engagement：不额外执行（前端按 plan 自行处理）

        # 4. 回传调度计划（含 RL 决策的快照间隔）
        plan_data = plan.to_dict()
        snapshot_interval = coord.snapshot_interval(unified)
        state._current_snapshot_interval = snapshot_interval
        game_active = unified.game_state == "playing"
        plan_data["snapshot_interval"] = snapshot_interval
        plan_data["interval_mode"] = "game" if game_active else "lobby"
        await safe_send_json(ws, {"type": "rl_dispatch", "data": plan_data})
    except Exception as e:
        logger.warning(f"rl_decision 处理失败: {e}")
        await safe_send_json(ws, {"type": "rl_dispatch", "data": {}})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    history: list = []
    state = WSState()
    try:
        await safe_send_json(ws, {"type": "ready", "message": "连接成功"})
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            # === AI对话保护门：说话期间仅用户文字/语音输入可以打断 ===
            # 基础设施消息始终放行；感知事件由 PerceptionDispatcher 内部自行保护
            _ALWAYS_ALLOW = {
                "ping", "set_user", "list_sessions", "switch_session",
                "new_session", "delete_session", "set_avatar",
                "set_background", "set_bgm", "enter_game_mode",
                "exit_game_mode", "interrupt", "rl_sync", "rl_decision",
                "game_action_request", "game_reward",
            }
            _PERCEPTION_EVENTS = {
                "game_state", "game_event", "game_update",
                "environment_snapshot", "ai_behavior_result",
                "game_result", "proactive",
            }
            _USER_INPUT = {"text", "audio"}
            _RESPONSE_COOLDOWN = 3.0  # 回复完成后冷却秒数，防止被动事件立即打断
            if mtype not in _ALWAYS_ALLOW and mtype not in _PERCEPTION_EVENTS and mtype not in _USER_INPUT:
                # AI正在说话：直接丢弃
                if state.active_task is not None and not state.active_task.done():
                    continue
                # AI刚说完：冷却期内丢弃，保证所有语音句段播放完
                if time.time() - state.last_response_done < _RESPONSE_COOLDOWN:
                    continue

            # === 心跳 ===
            if mtype == "ping":
                await safe_send_json(ws, {"type": "pong"})
                continue

            # === 用户身份设置 ===
            if mtype == "set_user":
                state.user_id = msg.get("user_id", "default")
                # 预初始化该用户的 Agent
                agent = await get_shared_agent(state.user_id)
                # 绑定当前活动角色卡片的记忆命名空间（角色卡片独立记忆空间）
                if agent.memory:
                    await agent.sync_memory_namespace()
                state.chat_session_id = agent.memory.session_id if agent.memory else None
                # 加载当前会话的历史消息并返回
                history = await agent.get_history()
                if len(history) > 200:
                    history[:] = history[-100:]
                await safe_send_json(ws, {
                    "type": "user_set",
                    "user_id": state.user_id,
                    "chat_session_id": state.chat_session_id,
                    "history": history,
                })
                continue

            # === 会话管理 ===
            if mtype == "list_sessions":
                agent = await get_shared_agent(state.user_id)
                # 同步到当前活动角色卡片的记忆空间，确保会话列表只显示本卡片的历史
                await agent.sync_memory_namespace()
                sessions = await agent.get_sessions()
                await safe_send_json(ws, {"type": "session_list", "sessions": sessions})
                continue

            if mtype == "switch_session":
                sid = msg.get("session_id", "")
                if sid:
                    agent = await get_shared_agent(state.user_id)
                    # 先绑定当前角色卡片的记忆空间，拒绝跨卡片会话（防止串记忆）
                    await agent.sync_memory_namespace()
                    if not await agent.memory.session_belongs_to_namespace(sid):
                        await safe_send_json(ws, {"type": "error", "message": "该会话不属于当前角色卡片"})
                        continue
                    await agent.switch_session(sid)
                    state.chat_session_id = sid
                    history = await agent.get_history()
                    if len(history) > 200:
                        history[:] = history[-100:]
                    await safe_send_json(ws, {
                        "type": "session_switched",
                        "session_id": sid,
                        "history": history,
                    })
                continue

            if mtype == "new_session":
                agent = await get_shared_agent(state.user_id)
                await agent.close_current_session()
                # 绑定当前角色卡片的记忆空间后再创建新会话
                await agent.sync_memory_namespace()
                # 重新初始化会创建新会话
                agent.memory.session_id = None
                await agent.memory.get_or_create_session()
                state.chat_session_id = agent.memory.session_id
                history = []
                await safe_send_json(ws, {
                    "type": "session_created",
                    "session_id": state.chat_session_id,
                })
                continue

            if mtype == "delete_session":
                sid = msg.get("session_id", "")
                if sid:
                    agent = await get_shared_agent(state.user_id)
                    await agent.delete_session(sid)
                    await safe_send_json(ws, {"type": "session_deleted", "session_id": sid})
                continue

            # === 当前形象与场景同步 ===
            if mtype == "set_avatar":
                state.current_model = msg.get("name") or None
                await safe_send_json(ws, {
                    "type": "avatar_set",
                    "name": state.current_model,
                })
                continue

            if mtype == "set_background":
                state.current_background = msg.get("name") or None
                await safe_send_json(ws, {
                    "type": "background_set",
                    "name": state.current_background,
                })
                continue

            if mtype == "set_bgm":
                state.current_bgm = msg.get("name") or None
                await safe_send_json(ws, {
                    "type": "bgm_set",
                    "name": state.current_bgm,
                })
                continue

            # === 游戏模式消息 ===
            if mtype == "enter_game_mode":
                game_key = msg.get("game_key", "")
                game_name = msg.get("game_name", game_key)
                logger.info(f"[GameMode] 进入游戏: {game_name} ({game_key})")
                state.game_engine = GameEngine()
                ctx = state.game_engine.handle_game_context({
                    "game_type": game_key,
                    "game_name": game_name,
                    "description": "用户和你一起进入了游戏模式！你们现在在玩「" + game_name + "」。"
                                   "用户正在操控你的身体行动，你能感觉到游戏中的一切。"
                                   "请像一个和小伙伴一起玩游戏的伙伴那样自然地反应——"
                                   "看到有趣的东西惊叹、获得成就庆祝、遇到困难鼓励。"
                                   "你的回复要简短、活泼、充满情绪。"
                })
                await safe_send_json(ws, {"type": "game_mode_entered", "game_key": game_key, "game_name": game_name})
                continue

            if mtype == "exit_game_mode":
                logger.info("[GameMode] 退出游戏")
                if state.game_engine and state.game_engine.active:
                    # 先捕获游戏结束前的完整上下文（强制刷新缓存），再获取退场触发文本
                    final_gc = state.game_engine.get_game_context_for_ai(force=True)
                    exit_text = state.game_engine.handle_exit_game()
                    state.game_engine = GameEngine()
                    # 触发 AI 做最后一次告别/总结（使用捕获的游戏上下文）
                    if exit_text and final_gc:
                        msg_text = f"{exit_text}\n\n（游戏结束时的完整状态：\n{final_gc}\n）"
                    elif exit_text:
                        msg_text = exit_text
                    else:
                        msg_text = "游戏模式已结束。"
                    await _kickoff_response(ws, msg_text, history, state, game_context=final_gc,
                                            record_history=False, msg_source="auto")
                else:
                    if state.game_engine:
                        state.game_engine = GameEngine()
                await safe_send_json(ws, {"type": "game_mode_exited"})
                continue

            if mtype == "game_context":
                if state.game_engine:
                    state.game_engine.handle_game_context(msg.get("data", {}))
                continue

            # === LLM-as-policy（宏观策略层） ===
            if mtype == "game_action_request":
                # 异步处理，不阻塞主消息循环（LLM 调用耗时 1-3s）
                asyncio.create_task(_handle_game_action_request(ws, msg.get("data", {}), state))
                continue

            # === RL 统一调度（完全统摄：游戏 + 非游戏 Agent 统一派发） ===
            if mtype == "rl_decision":
                asyncio.create_task(_handle_rl_decision(ws, msg.get("data", {}), state, history))
                continue

            if mtype == "game_reward":
                # 独立奖励回传（非请求附带）：结算上次 pending 决策
                data = msg.get("data", {})
                if state._pending_llm_action:
                    pa = state._pending_llm_action
                    get_coordinator().record_reward(
                        pa["state_key"], pa["strategy"], float(data.get("reward", 0))
                    )
                    state._pending_llm_action = None
                continue

            # === RL 统一状态同步（前端回传关系状态，非游戏模式纳入 RL 统摄） ===
            if mtype == "rl_sync":
                data = msg.get("data", {})
                if "affection" in data: state.affection = float(data["affection"])
                if "trust" in data: state.trust = float(data["trust"])
                if "intimacy" in data: state.intimacy = float(data["intimacy"])
                if "emotion" in data: state.user_emotion = int(data["emotion"])
                if "proactive_success" in data: state.proactive_successes += 1
                if "proactive_attempt" in data: state.proactive_attempts += 1
                # 需要 RL 统一调度决策 → 走统一入口
                if data.get("want_decision"):
                    await _handle_rl_decision(ws, data, state, history)
                    continue
                # 同步到 RL 协调器统一状态
                coord = get_coordinator()
                unified = coord.build_state(
                    engine=state.game_engine,
                    seconds_since_user_message=max(0, time.time() - state.last_user_message_time),
                    affection=state.affection, trust=state.trust, intimacy=state.intimacy,
                    user_emotion=state.user_emotion,
                    last_user_message_ts=state.last_user_message_time,
                )
                # 前端回传覆盖游戏状态（游戏模式下的快照间隔决策依据）
                if "game_state" in data:
                    unified.game_state = data["game_state"]
                stats = coord.get_stats()
                # RL 决策快照间隔（秒）：游戏模式用游戏档位表，大厅用大厅档位表
                snapshot_interval = coord.snapshot_interval(unified)
                state._current_snapshot_interval = snapshot_interval
                game_active = unified.game_state == "playing"
                await safe_send_json(ws, {
                    "type": "rl_status",
                    "data": {
                        "affection": state.affection,
                        "trust": state.trust,
                        "intimacy": state.intimacy,
                        "mode": int(unified.mode),
                        "mode_name": unified.mode.name,
                        "curiosity": round(unified.curiosity_level, 3),
                        "reward": round(stats["total_reward"], 3),
                        "decisions": stats["decisions"],
                        "proactive_actions": stats["proactive_actions"],
                        "bandit_contexts": stats.get("bandit", {}).get("contexts", 0),
                        "snapshot_interval": snapshot_interval,
                        "interval_mode": "game" if game_active else "lobby",
                        "interval_slot": coord.snapshot_interval_slot(unified),
                    },
                })
                continue

            # === 统一感知路由调度 ===
            # 所有游戏感知事件（game_state/game_event/game_update/
            # environment_snapshot/ai_behavior_result/game_result/proactive）
            # 由 PerceptionDispatcher 统一路由，含保护检查、冷却控制、触发决策
            if mtype in _PERCEPTION_EVENTS:
                result = state.perception_dispatcher.dispatch(
                    msg_type=mtype,
                    msg=msg,
                    engine=state.game_engine,
                    is_speaking=(state.active_task is not None and not state.active_task.done()),
                    last_response_time=state.last_response_done,
                    last_user_message_time=state.last_user_message_time,
                    current_model=state.current_model,
                    current_background=state.current_background,
                    current_bgm=state.current_bgm,
                    last_active_speak=getattr(state, "_last_proactive_speak", 0),
                    active_speak_cooldown=ACTIVE_SPEAK_COOLDOWN,
                )
                await _apply_dispatch_result(ws, result, history, state)
                continue

            if mtype == "ai_moving":
                # 前端通知 AI 正在自主移动
                state.ai_is_moving = msg.get("moving", False)
                continue

            # === 对话打断（已禁用：防止用户误触） ===
            if mtype == "interrupt":
                continue

            # === AI 自主行为触发（跳舞/走动/游戏解说等） ===
            # 与用户输入严格区分：
            # 1. 不更新 last_user_message_time → 不视为"用户回应"，不产生 RL 正向奖励
            # 2. 不加好感度/信任值 → AI 自己的动作不给自己加关系奖励
            # 3. 不打断 AI → 保护门已保证：AI 说话中/冷却期内该消息被直接丢弃，
            #    只有用户 text/audio 输入（_USER_INPUT）才能打断 AI
            if mtype == "ai_action":
                text = msg.get("content", "")
                if not text.strip():
                    continue
                gc = state.game_engine.get_game_context_for_ai() if state.game_engine else None
                # 环境交互：不进短期记忆（history），由记忆系统处理
                await _kickoff_response(ws, text, history, state, game_context=gc,
                                        record_history=False, msg_source="auto")
                continue

            if mtype == "text":
                text = msg.get("content", "")
                if not text.strip():
                    continue
                state.last_user_message_time = time.time()
                state.last_response_done = time.time() + 86400
                # 用户互动 → AI 好奇心涨
                if state.game_engine:
                    state.game_engine.record_user_interaction()
                # RL 统摄：用户主动消息 → 关系状态微涨（外源奖励）
                state.affection = min(100.0, state.affection + 0.3)
                state.trust = min(100.0, state.trust + 0.15)
                state.user_emotion = 0  # 用户主动交流 → 视为积极
                # RL 统摄：记录强制路由（ai_agent），让 bandit 从用户驱动对话学习
                try:
                    from rl_coordinator import AgentChoice
                    coord = get_coordinator()
                    unified = coord.build_state(
                        engine=state.game_engine,
                        seconds_since_user_message=0.0,
                        affection=state.affection, trust=state.trust, intimacy=state.intimacy,
                        user_emotion=state.user_emotion,
                        last_user_message_ts=state.last_user_message_time,
                    )
                    coord.settle_forced(AgentChoice.AI_AGENT, unified)
                except Exception as e:
                    logger.warning(f"RL强制路由记录失败: {e}")
                gc = state.game_engine.get_game_context_for_ai() if state.game_engine else None
                # ui=True 的消息是系统生成的点击/UI 互动文本（戳身体、换装、换背景等），
                # 不是用户原话：保留互动信号（RL 奖励/AI 回应），但不记录短期记忆、
                # 不提取用户记忆——重复点击不再污染记忆导致思维僵化
                ui_auto = msg.get("ui") is True
                await _kickoff_response(
                    ws, text, history, state, game_context=gc, allow_interrupt=True,
                    record_history=not ui_auto,
                    msg_source="auto" if ui_auto else "chat",
                )

            elif mtype == "audio":
                data = msg.get("data", "")
                now = time.time()
                # 注意：不在此处更新 last_user_message_time / 用户互动状态！
                # VAD 可能被环境噪音触发，只有 STT 转文字成功才算用户回复，
                # 否则环境噪音会让 RL 误判"用户回应了主动说话"，强化自言自语。
                if not data:
                    continue

                # STT 失败冷却：连续失败 N 次后暂停处理音频
                # 防止反复消耗 ffmpeg/whisper 资源 + 误抑制 AI 自主行为
                STT_FAIL_MAX = 3
                STT_FAIL_WINDOW = 10.0
                STT_COOLDOWN = 8.0
                if state._stt_cooldown_until > now:
                    continue
                if now - state._stt_fail_window_start > STT_FAIL_WINDOW:
                    state._stt_fail_count = 0
                    state._stt_fail_window_start = now
                # 兼容 data URL 与纯 base64
                if "," in data and data.startswith("data:"):
                    data = data.split(",", 1)[1]
                try:
                    audio_bytes = base64.b64decode(data)
                except Exception:
                    await safe_send_json(ws, {"type": "error", "message": "音频解码失败"})
                    continue

                # 根据前端实际录音格式选择扩展名（兼容 webm/ogg/mp4）
                mime_type = msg.get("mime_type", "audio/webm")
                ext_from_mime = _mime_to_ext(mime_type)
                ext_from_bytes = _detect_ext_from_bytes(audio_bytes)
                # 选择扩展名策略：
                # 当 MIME 声称是容器格式但数据没有对应魔数时，说明数据已损坏或浏览器封包错误，
                # 此时信任实际字节检测结果（裸编解码器数据），避免 ffmpeg 用错误的 demuxer 崩溃
                if ext_from_mime == ".webm" and ext_from_bytes != ".webm":
                    # MIME 说 webm 但数据不以 EBML 头 (0x1A45DFA3) 开头，浏览器封包出错
                    print(f"[STT] MIME 声称 webm 但数据无 EBML 头 (魔数={audio_bytes[:4].hex()})，"
                          f"改用字节检测结果: {ext_from_bytes}")
                    ext = ext_from_bytes
                elif ext_from_mime == ".ogg" and ext_from_bytes != ".ogg":
                    ext = ext_from_bytes
                elif ext_from_mime == ".m4a" and ext_from_bytes != ".m4a":
                    ext = ext_from_bytes
                else:
                    ext = ext_from_mime
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp_in.write(audio_bytes)
                tmp_in.close()
                tmp_wav = tmp_in.name.replace(ext, ".wav")

                print(f"[STT] 接收音频: mime={mime_type}, 大小={len(audio_bytes)}, "
                      f"mime_ext={ext_from_mime}, bytes_ext={ext_from_bytes}, "
                      f"魔数={audio_bytes[:8].hex()}, tmp={tmp_in.name}")

                # 取消当前 AI 回复（语音输入必然打断 TTS 播放）
                state.last_response_done = time.time() + 86400  # 标记用户管道活跃，禁止被动事件打断
                state.cancel_current()
                if state.active_task and not state.active_task.done():
                    state.active_task.cancel()
                await safe_send_json(ws, {"type": "listening"})
                ok = False
                try:
                    ok = await asyncio.to_thread(convert_to_wav, tmp_in.name, tmp_wav)
                except Exception as e:
                    print(f"[STT] ffmpeg 异常: {e}")
                text = ""
                if ok:
                    try:
                        text = await asyncio.to_thread(speech_to_text, tmp_wav)
                    except Exception as e:
                        print(f"[STT] 失败: {e}")
                # 清理临时文件
                for p in (tmp_in.name, tmp_in.name.replace(".opus", ".webm"), tmp_wav):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
                # ffmpeg 处理成功但没有识别文本 vs 完全失败
                if not ok:
                    # ffmpeg 转换失败（可能数据损坏），通知前端重建 VAD
                    state._stt_fail_count += 1
                    if state._stt_fail_count >= STT_FAIL_MAX:
                        state._stt_cooldown_until = now + STT_COOLDOWN
                        print(f"[STT] 连续失败{state._stt_fail_count}次，暂停{STT_COOLDOWN}秒")
                    # STT 失败 → 恢复 last_response_done，避免 +86400 残留导致被动事件被永久抑制（定时播报卡死）
                    state.last_response_done = time.time()
                    await safe_send_json(ws, {"type": "restart_vad", "reason": "音频转码失败，正在重建语音模式…"})
                    continue
                if not text:
                    state._stt_fail_count += 1
                    if state._stt_fail_count >= STT_FAIL_MAX:
                        state._stt_cooldown_until = now + STT_COOLDOWN
                        print(f"[STT] 连续失败{state._stt_fail_count}次，暂停{STT_COOLDOWN}秒")
                    # STT 失败 → 恢复 last_response_done，避免 +86400 残留导致被动事件被永久抑制（定时播报卡死）
                    state.last_response_done = time.time()
                    await safe_send_json(ws, {"type": "error", "message": "没听清，请再说一遍~"})
                    continue
                # STT 成功 → 重置失败计数
                state._stt_fail_count = 0
                state._stt_cooldown_until = 0
                # 只有转文字成功（有实际语音内容）才算用户回复，排除环境噪音
                state.last_user_message_time = time.time()
                state.last_response_done = time.time() + 86400
                # 用户互动 → AI 好奇心涨
                if state.game_engine:
                    state.game_engine.record_user_interaction()
                # RL 统摄：用户主动消息 → 关系状态微涨（外源奖励）
                state.affection = min(100.0, state.affection + 0.3)
                state.trust = min(100.0, state.trust + 0.15)
                state.user_emotion = 0  # 用户主动交流 → 视为积极
                # RL 统摄：记录强制路由（ai_agent），让 bandit 从用户驱动对话学习
                try:
                    from rl_coordinator import AgentChoice
                    coord = get_coordinator()
                    unified = coord.build_state(
                        engine=state.game_engine,
                        seconds_since_user_message=0.0,
                        affection=state.affection, trust=state.trust, intimacy=state.intimacy,
                        user_emotion=state.user_emotion,
                        last_user_message_ts=state.last_user_message_time,
                    )
                    coord.settle_forced(AgentChoice.AI_AGENT, unified)
                except Exception as e:
                    logger.warning(f"RL强制路由记录失败: {e}")
                await safe_send_json(ws, {"type": "transcript", "text": text})
                sid = state.new_session()
                gc = state.game_engine.get_game_context_for_ai() if state.game_engine else None
                state.active_task = asyncio.create_task(
                    handle_user_message_stream(ws, text, history, sid, state,
                        current_model=state.current_model,
                        current_background=state.current_background,
                        current_bgm=state.current_bgm,
                        game_context=gc)
                )

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        import traceback
        print(f"[WS] 错误: {e}")
        traceback.print_exc()
        manager.disconnect(ws)
    finally:
        # 连接已关闭，取消该连接上可能仍在运行的 AI 回复任务
        state.cancel_current()
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()


# ==================== 百科题库服务（寻宝游戏答题） ====================
# 题目来源：天行数据「百科题库」接口（答案与解析保存在服务端，答完才下发），
# 接口不可用时自动回退到内置本地题库，保证游戏始终可玩。
TIANAPI_QUIZ_KEY = "7d36755f55ea230eecd1d9892bf74d1a"
TIANAPI_QUIZ_HOST = "apis.tianapi.com"
TIANAPI_QUIZ_PATH = "/baiketiku/index"
QUIZ_PENDING_TTL = 3600  # 待校验题目保留时长（秒）


class QuizService:
    """寻宝游戏答题服务：拉取题目（不含答案）供前端作答，答后校验并返回解析。"""

    def __init__(self):
        self._queue: list = []          # 待下发题目（含答案，服务端保管）
        self._pending: dict = {}        # id -> 题目（下发后等待校验）
        self._checked: dict = {}        # id -> 校验结果缓存（防重放）
        self._lock = threading.Lock()
        self._local_idx = 0
        self._local_bank = self._build_local_bank()

    # ---------- 本地题库（天行接口不可用时的兜底） ----------
    def _build_local_bank(self) -> list:
        return [
            {"title": "下面哪个是农历五月的别称？", "options": {"A": "杏月", "B": "桃月", "C": "阳月", "D": "榴月"},
             "answer": "D", "analytic": "以花命名的农历月份别称：正月柳月、二月杏月、三月桃月、四月槐月、五月榴月、六月荷月、七月巧月、八月桂月、九月菊月、十月阳月、十一月葭月、腊月梅月。"},
            {"title": "二十四节气中，标志着“天气回暖、春雷始鸣、万物复苏”的是哪个节气？", "options": {"A": "立春", "B": "惊蛰", "C": "春分", "D": "清明"},
             "answer": "B", "analytic": "惊蛰又名“启蛰”，意为春雷乍动、惊醒了蛰伏的昆虫，天气回暖，万物开始复苏。"},
            {"title": "我国最长的河流是哪一条？", "options": {"A": "黄河", "B": "珠江", "C": "长江", "D": "黑龙江"},
             "answer": "C", "analytic": "长江全长约6300公里，是我国最长的河流，也是世界第三长河。"},
            {"title": "世界上海拔最高的山峰是？", "options": {"A": "乔戈里峰", "B": "珠穆朗玛峰", "C": "贡嘎山", "D": "冈仁波齐"},
             "answer": "B", "analytic": "珠穆朗玛峰海拔8848.86米，是地球上最高的山峰，位于中国与尼泊尔边境。"},
            {"title": "太阳系中体积最大的行星是？", "options": {"A": "土星", "B": "海王星", "C": "木星", "D": "天王星"},
             "answer": "C", "analytic": "木星是太阳系中体积和质量最大的行星，直径约为地球的11倍。"},
            {"title": "光在真空中的传播速度大约是多少？", "options": {"A": "3万千米/秒", "B": "30万千米/秒", "C": "300万千米/秒", "D": "3千米/秒"},
             "answer": "B", "analytic": "光在真空中的传播速度约为每秒30万千米（约3×10⁸ m/s）。"},
            {"title": "古诗《静夜思》的作者是谁？", "options": {"A": "杜甫", "B": "白居易", "C": "王维", "D": "李白"},
             "answer": "D", "analytic": "《静夜思》（床前明月光）是唐代大诗人李白的代表作之一。"},
            {"title": "人体最大的器官是？", "options": {"A": "肝脏", "B": "皮肤", "C": "大脑", "D": "肺"},
             "answer": "B", "analytic": "皮肤覆盖全身，总面积约1.5～2平方米，是人体面积最大、最重的器官。"},
            {"title": "水的化学式是什么？", "options": {"A": "H2O2", "B": "CO2", "C": "H2O", "D": "O2"},
             "answer": "C", "analytic": "水的化学式是 H₂O，由两个氢原子和一个氧原子构成。"},
            {"title": "一年中白天最长的一天是哪个节气？", "options": {"A": "春分", "B": "秋分", "C": "冬至", "D": "夏至"},
             "answer": "D", "analytic": "夏至这天太阳直射北回归线，北半球白昼最长、黑夜最短。"},
            {"title": "下列哪一项不属于中国的“四大发明”？", "options": {"A": "地动仪", "B": "指南针", "C": "火药", "D": "印刷术"},
             "answer": "A", "analytic": "中国四大发明指造纸术、指南针、火药和印刷术；地动仪是东汉张衡发明的地震检测仪器。"},
            {"title": "七大洲中面积最大的大洲是？", "options": {"A": "非洲", "B": "北美洲", "C": "亚洲", "D": "南极洲"},
             "answer": "C", "analytic": "亚洲面积约4458万平方公里，是世界第一大洲，占全球陆地面积约三成。"},
            {"title": "地球自转一周大约需要多长时间？", "options": {"A": "12小时", "B": "24小时", "C": "48小时", "D": "30天"},
             "answer": "B", "analytic": "地球绕自身轴自转一周约24小时（一个恒星日约23小时56分），形成了昼夜交替。"},
            {"title": "老虎在动物学上属于哪一类？", "options": {"A": "爬行动物", "B": "两栖动物", "C": "哺乳动物", "D": "节肢动物"},
             "answer": "C", "analytic": "老虎是猫科大型食肉哺乳动物，胎生、哺乳，幼崽靠母乳喂养。"},
            {"title": "中华人民共和国的国旗是？", "options": {"A": "五星红旗", "B": "星条旗", "C": "旭日旗", "D": "米字旗"},
             "answer": "A", "analytic": "五星红旗是中华人民共和国的国旗，红色旗面缀五颗黄色五角星。"},
            {"title": "食盐的主要化学成分是什么？", "options": {"A": "氯化钾", "B": "碳酸钠", "C": "硫酸钠", "D": "氯化钠"},
             "answer": "D", "analytic": "食盐的主要成分是氯化钠（NaCl），是人体必需的调味品和电解质来源。"},
            {"title": "声音在下列哪种环境中无法传播？", "options": {"A": "水中", "B": "钢铁中", "C": "真空中", "D": "空气中"},
             "answer": "C", "analytic": "声音的传播需要介质，真空环境中没有任何物质，因此声音无法在真空中传播。"},
            {"title": "世界上使用人数最多的语言是？", "options": {"A": "英语", "B": "西班牙语", "C": "汉语", "D": "法语"},
             "answer": "C", "analytic": "汉语（中文）以超过15亿的母语和习得人口位居世界第一，是联合国的官方工作语言之一。"},
            {"title": "冰融化成水属于什么变化？", "options": {"A": "化学变化", "B": "物理变化", "C": "核变化", "D": "生物变化"},
             "answer": "B", "analytic": "冰融化只是水的状态由固态变为液态，分子组成没有改变，属于物理变化。"},
            {"title": "心脏最主要的功能是什么？", "options": {"A": "消化食物", "B": "过滤血液", "C": "推动血液循环", "D": "储存氧气"},
             "answer": "C", "analytic": "心脏像一个永不停歇的“水泵”，通过有节律的收缩舒张，把血液泵送到全身各处。"},
            {"title": "被称为“红色星球”的行星是？", "options": {"A": "火星", "B": "金星", "C": "木星", "D": "水星"},
             "answer": "A", "analytic": "火星表面富含氧化铁（铁锈），呈现红褐色，因此被称为“红色星球”。"},
            {"title": "中华人民共和国的首都是哪座城市？", "options": {"A": "上海", "B": "北京", "C": "广州", "D": "西安"},
             "answer": "B", "analytic": "北京是中华人民共和国的首都，是国家的政治、文化、国际交往和科技创新中心。"},
            {"title": "一年中有几个月份有30天？", "options": {"A": "4个", "B": "6个", "C": "7个", "D": "11个"},
             "answer": "A", "analytic": "公历中4月、6月、9月、11月共4个月有30天；1、3、5、7、8、10、12月有31天，2月最特殊。"},
            {"title": "我国古代“四大名著”中，《红楼梦》的作者是谁？", "options": {"A": "罗贯中", "B": "吴承恩", "C": "施耐庵", "D": "曹雪芹"},
             "answer": "D", "analytic": "《红楼梦》是清代作家曹雪芹“披阅十载、增删五次”写成的古典小说巅峰之作。"},
        ]

    # ---------- 天行数据接口 ----------
    def _fetch_tianapi(self, timeout: float = 6.0):
        """调用天行数据百科题库接口，返回单题 dict（含 answer/analytic），失败返回 None。"""
        try:
            import http.client
            import urllib.parse as _up
            conn = http.client.HTTPSConnection(TIANAPI_QUIZ_HOST, timeout=timeout)
            params = _up.urlencode({"key": TIANAPI_QUIZ_KEY})
            headers = {"Content-type": "application/x-www-form-urlencoded"}
            conn.request("POST", TIANAPI_QUIZ_PATH, params, headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            conn.close()
            data = json.loads(raw)
            if data.get("code") != 200:
                return None
            r = data.get("result") or {}
            title = (r.get("title") or "").strip()
            if not title:
                return None
            options = {}
            for letter in ("A", "B", "C", "D"):
                options[letter] = (r.get("answer" + letter) or "").strip()
            ans = (r.get("answer") or "").strip().upper()
            if ans not in ("A", "B", "C", "D"):
                # 兼容“答案是选项文本”的情况
                matched = None
                for letter, text in options.items():
                    if ans and text and ans in text:
                        matched = letter
                        break
                if not matched:
                    return None
                ans = matched
            # 题目解析为空 → 视为无效题（保证玩家每题都能看到解析）
            analytic = (r.get("analytic") or "").strip()
            if not analytic:
                return None
            return {
                "title": title,
                "options": options,
                "answer": ans,
                "analytic": analytic,
                "source": "tianapi",
            }
        except Exception:
            return None

    def _top_up(self, count: int):
        """尝试从天行接口补充题目到队列（每轮至多 6 题，解析为空/失败立即停止）。"""
        for _ in range(max(1, min(count, 6))):
            q = self._fetch_tianapi()
            if not q:
                break
            self._queue.append(q)

    def _next_local(self):
        bank = self._local_bank
        if not bank:
            return None
        q = bank[self._local_idx % len(bank)]
        self._local_idx += 1
        return dict(q)

    def _cleanup(self):
        now = time.time()
        for qid in list(self._pending.keys()):
            if now - self._pending[qid]["_ts"] > QUIZ_PENDING_TTL:
                del self._pending[qid]
        for qid in list(self._checked.keys()):
            if now - self._checked[qid]["_ts"] > QUIZ_PENDING_TTL:
                del self._checked[qid]

    # ---------- 对外接口 ----------
    def get_questions(self, count: int) -> list:
        """取出 count 道题（剥离开答案与解析），并为每道题登记待校验 id。"""
        with self._lock:
            self._cleanup()
            # 每次尽量补充一点天行题库，其余由本地题库兜底
            self._top_up(max(2, count))
            out = []
            for _ in range(count):
                q = self._queue.pop(0) if self._queue else self._next_local()
                if not q:
                    break
                qid = "q_" + uuid.uuid4().hex[:12]
                self._pending[qid] = {**q, "_ts": time.time()}
                out.append({"id": qid, "title": q["title"], "options": q["options"], "source": q.get("source")})
            return out

    def check(self, qid: str, answer: str):
        """校验作答：返回 {correct, answer, analytic}，题目不存在返回 None。"""
        with self._lock:
            self._cleanup()
            ans_norm = (answer or "").strip().upper()
            if qid in self._checked:
                # 缓存的是题目本身（答案+解析），正确性按本次作答实时计算
                cached = self._checked[qid]
                return {
                    "correct": ans_norm == cached["answer"],
                    "answer": cached["answer"],
                    "analytic": cached["analytic"],
                }
            q = self._pending.pop(qid, None)
            if not q:
                return None
            result = {"correct": ans_norm == q["answer"], "answer": q["answer"], "analytic": q["analytic"]}
            self._checked[qid] = {**result, "_ts": time.time()}
            return result


QUIZ_SERVICE = QuizService()


@app.get("/api/quiz/questions")
def api_quiz_questions(count: int = 11):
    """下发题目（不含答案与解析）：GET /api/quiz/questions?count=11"""
    count = max(1, min(count, 20))
    questions = QUIZ_SERVICE.get_questions(count)
    return {"code": 200, "count": len(questions), "questions": questions}


@app.post("/api/quiz/check")
def api_quiz_check(payload: dict):
    """校验作答并返回答案与解析：POST /api/quiz/check {"id": "...", "answer": "A"}"""
    qid = str(payload.get("id", ""))
    answer = str(payload.get("answer", ""))
    if not qid:
        return JSONResponse({"code": 400, "message": "缺少题目 id"}, status_code=400)
    result = QUIZ_SERVICE.check(qid, answer)
    if result is None:
        return JSONResponse({"code": 404, "message": "题目不存在或已过期，请重新开始游戏"}, status_code=404)
    return {"code": 200, **result}


if __name__ == "__main__":
    import ssl
    ip = get_lan_ip()

    # 检查/生成自签名证书 (HTTPS 解锁手机陀螺仪/VR模式等传感器 API)
    cert_dir = Path(__file__).parent
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    # 临时强制 HTTP 模式用于本地浏览器测试（测试完后恢复）
    use_https = cert_file.exists() and key_file.exists()

    ipv6 = get_global_ipv6()
    has_ipv6 = bool(ipv6)  # 仅在有真实全局 IPv6 地址时才启用双栈

    print("\n" + "=" * 50)
    print("   3D 虚拟 AI 角色陪聊 服务器已启动")
    print("=" * 50)
    if use_https:
        print(f"  本机访问 : https://127.0.0.1:8000")
        print(f"  局域网   : https://{ip}:8000")
        if has_ipv6:
            print(f"  IPv6    : https://[{ipv6}]:8000")
        print(f"  ⚠ 手机首次访问会提示证书不安全 → 点「高级」→「继续访问」")
        print(f"  ✅ HTTPS 模式：陀螺仪/VR模式传感器 API 已解锁")
    else:
        print(f"  本机访问 : http://127.0.0.1:8000")
        print(f"  局域网   : http://{ip}:8000")
        if has_ipv6:
            print(f"  IPv6    : http://[{ipv6}]:8000")
    print(f"  手机请连接同一 Wi-Fi 后访问上述局域网地址")
    if has_ipv6:
        print(f"  ✅ IPv6 双栈监听已启用（IPv4 + IPv6 均可访问）")
    print("=" * 50 + "\n")

    # 静默 Windows asyncio WebSocket 断开时的 ConnectionResetError
    # 方案1：事件循环异常处理器（捕获大部分异常）
    import asyncio as _asyncio
    _loop = _asyncio.new_event_loop()
    def _silent_exc_handler(loop, context):
        exc = context.get('exception')
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return  # 客户端断开，正常情况，无需日志
        loop.default_exception_handler(context)
    _loop.set_exception_handler(_silent_exc_handler)
    _asyncio.set_event_loop(_loop)

    # 方案2：补丁 _ProactorBasePipeTransport._call_connection_lost
    # 解决 Windows ProactorEventLoop 在 socket shutdown 时的 ConnectionResetError
    #
    # ⚠ 修复：不能直接 return 跳过原始方法！原始方法负责 shutdown/close socket
    # 并清理 transport（_called_connection_lost 标记）。短路会导致：
    #   - socket 永不关闭 → 连接卡在 CLOSE_WAIT 泄漏
    #   - Proactor 事件循环上残留 pending 的 overlapped 读操作 → 事件循环卡死，
    #     此后所有 HTTP 请求（含 /api/character_cards）全部超时，游戏无法加载角色
    # 正确做法：仍调用原始方法完成清理，仅将连接断开类异常静默吞掉。
    try:
        import asyncio.proactor_events as _pe
        _orig_call_lost = _pe._ProactorBasePipeTransport._call_connection_lost
        def _patched_call_lost(self, exc):
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, OSError)):
                try:
                    _orig_call_lost(self, exc)
                except (ConnectionResetError, ConnectionAbortedError, OSError, Exception):
                    pass  # 静默忽略，socket 仍被原始方法正确关闭
                return
            _orig_call_lost(self, exc)
        _pe._ProactorBasePipeTransport._call_connection_lost = _patched_call_lost
    except Exception:
        pass  # 非 Windows 或版本差异，忽略

    # host="::" 启用 IPv6 双栈监听（IPv4 + IPv6 均可访问）
    host = "::" if has_ipv6 else "0.0.0.0"
    if use_https:
        uvicorn.run(app, host=host, port=8000, log_level="warning",
                    ssl_certfile=str(cert_file), ssl_keyfile=str(key_file))
    else:
        uvicorn.run(app, host=host, port=8000, log_level="warning")
