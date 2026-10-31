import os
import io
import re
import json
import wave
import time
import socket
import base64
import asyncio
import tempfile
import httpx
import aiohttp
import requests
from pathlib import Path
from xml.sax.saxutils import escape
from dotenv import load_dotenv
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse
from .speech_cleanup import clean_transcript

load_dotenv()

router = APIRouter()

DEFAULT_OPENAI_KEY = ""

EDGE_TTS_VOICES = [
    {"id": "uz-UZ-SardorNeural", "name": "Sardor — O'zbek (Erkak, Tabiiy Neural)", "lang": "uz-UZ"},
    {"id": "uz-UZ-MadinaNeural", "name": "Madina — O'zbek (Ayol, Tabiiy Neural)", "lang": "uz-UZ"},
    {"id": "ru-RU-SvetlanaNeural", "name": "Svetlana — Русский (Женский Neural)", "lang": "ru-RU"},
    {"id": "ru-RU-DmitryNeural", "name": "Dmitry — Русский (Мужской Neural)", "lang": "ru-RU"},
    {"id": "en-US-AvaMultilingualNeural", "name": "Ava — English / Multilingual (Female)", "lang": "en-US"},
    {"id": "en-US-AndrewMultilingualNeural", "name": "Andrew — English / Multilingual (Male)", "lang": "en-US"},
    {"id": "tr-TR-EmelNeural", "name": "Emel — Türkçe (Kadın Neural)", "lang": "tr-TR"},
]

FREE_AI_MODELS = [
    {"id": "gpt-5.4-nano", "name": "GPT-5.4-Nano (0.2s Super-Turbo ⚡)"},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1-Mini (Ultra-Tezkor ⚡)"},
    {"id": "gpt-4o-mini", "name": "GPT-4o-Mini (Tezkor)"},
]

# Persistent warm IPv4 HTTP client for zero-handshake latency to OpenAI
_HTTP_CLIENT: httpx.AsyncClient | None = None
_SYNC_SESSION = requests.Session()
_TTS_CACHE: dict[tuple[str, str, str], bytes] = {}
_PROMPT_CACHE: dict[str, str] = {}

CALL_CONNECT_GREETING = "Assalomu alaykum! Eshitaman, sizga qanday yordam bera olaman?"
FALLBACK_ERROR_REPLY = "Uzr, aloqada qisqa uzilish bo'ldi, savolingizni qaytarib yuborasizmi?"

SAHIY_FAQ_ITEMS: list[dict] = []

VOICE_BRIDGES = {
    "uz-UZ": {
        "q_opener": "Albatta,",
        "s_opener": "Tushunarli,",
    },
    "ru-RU": {
        "q_opener": "Конечно,",
        "s_opener": "Понятно,",
    },
    "en-US": {
        "q_opener": "Sure,",
        "s_opener": "Got it,",
    },
}


class PersistentEdgeTTS:
    """
    Maintains a persistent, pre-handshaked IPv4 WebSocket connection to Microsoft Edge-TTS.
    Eliminates the 850ms-11,000ms TCP+TLS+WebSocket+IPv6 handshake overhead on every turn.
    """

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._lock = asyncio.Lock()

    async def _ensure_ws(self) -> aiohttp.ClientWebSocketResponse:
        from edge_tts.communicate import (
            WSS_URL,
            SEC_MS_GEC_VERSION,
            WSS_HEADERS,
            _SSL_CTX,
            connect_id,
            date_to_string,
        )
        from edge_tts.drm import DRM

        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ttl_dns_cache=600)
            self._session = aiohttp.ClientSession(connector=connector)

        if self._ws is None or self._ws.closed:
            self._ws = await self._session.ws_connect(
                f"{WSS_URL}&ConnectionId={connect_id()}"
                f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
                compress=15,
                headers=DRM.headers_with_muid(WSS_HEADERS),
                ssl=_SSL_CTX,
                heartbeat=25.0,
            )
            await self._ws.send_str(
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":{'
                '"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"'
                '},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n'
            )
        return self._ws

    async def synthesize(self, text: str, voice: str, rate_str: str) -> bytes:
        from edge_tts.communicate import (
            connect_id,
            date_to_string,
            ssml_headers_plus_data,
            mkssml,
            get_headers_and_data,
            remove_incompatible_characters,
        )
        from edge_tts.data_classes import TTSConfig

        escaped = escape(remove_incompatible_characters(text)).encode("utf-8")
        cfg = TTSConfig(voice, rate_str, "+0%", "+0Hz", "SentenceBoundary")

        async with self._lock:
            for attempt in range(2):
                try:
                    ws = await self._ensure_ws()
                    await ws.send_str(
                        ssml_headers_plus_data(
                            connect_id(),
                            date_to_string(),
                            mkssml(cfg, escaped),
                        )
                    )
                    buf = bytearray()
                    turn_finished = False
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            if len(msg.data) >= 2:
                                hlen = int.from_bytes(msg.data[:2], "big")
                                params, data = get_headers_and_data(msg.data, hlen)
                                if params.get(b"Path") == b"audio" and data:
                                    buf.extend(data)
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            if "Path:turn.end" in msg.data:
                                turn_finished = True
                                break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self._ws = None
                            break
                    if buf and turn_finished:
                        return bytes(buf)
                    await self._discard_socket()
                except asyncio.CancelledError:
                    # A cancelled request leaves audio/turn.end frames unread.
                    # Detach before awaiting close so no later call can consume
                    # that audio or cache it under a different customer's text.
                    await self._discard_socket()
                    raise
                except Exception:
                    await self._discard_socket()
                    if attempt == 1:
                        raise
        return b""

    async def _discard_socket(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=2)
            except Exception:
                pass


_EDGE_POOL = PersistentEdgeTTS()


def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
        _HTTP_CLIENT = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(12.0, connect=2.5),
            limits=httpx.Limits(max_keepalive_connections=15, max_connections=30, keepalive_expiry=600.0),
        )
    return _HTTP_CLIENT


def get_active_openai_key() -> str:
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key and env_key not in {"your_api_key_here", "replace_me"}:
        return env_key
    return DEFAULT_OPENAI_KEY


def clean_text_for_tts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"[#`_~>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_duplicate_opener(text: str) -> str:
    """Remove leading filler words if the browser already played a voice bridge opener."""
    cleaned = re.sub(
        r"^(ha\s+albatta|albatta|tushunarli|hoziroq|marhamat|ha|конечно|понятно|да|sure|got\s+it|yes)[\,\!\.\;\s]+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if cleaned:
        return cleaned[0].upper() + cleaned[1:]
    return text


def get_character_system_prompt(character: str) -> str:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    char_file = os.path.join(project_dir, "characters", character, f"{character}.txt")
    if os.path.exists(char_file):
        with open(char_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        _PROMPT_CACHE[character] = prompt
        return prompt
    return (
        "Siz aqlli, samimiy va universal ovozli AI yordamchisiz. "
        "Har qanday mavzudagi savollarga cheklovlarsiz, aniq va tabiiy javob bering."
    )


def mp3_bytes_to_wav_file(mp3_bytes: bytes, wav_path: str):
    """Convert MP3 bytes to a standard 16-bit PCM WAV file using PyAV."""
    import av

    container = av.open(io.BytesIO(mp3_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
    pcm_chunks = []

    for frame in container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        if not isinstance(resampled_frames, list):
            resampled_frames = [resampled_frames] if resampled_frames is not None else []
        for rframe in resampled_frames:
            pcm_chunks.append(rframe.to_ndarray().tobytes())

    flushed = resampler.resample(None)
    if flushed:
        if not isinstance(flushed, list):
            flushed = [flushed]
        for rframe in flushed:
            pcm_chunks.append(rframe.to_ndarray().tobytes())

    container.close()

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"".join(pcm_chunks))


async def generate_edge_tts_audio(
    text: str,
    voice: str = "uz-UZ-SardorNeural",
    speed: float = 1.15,
    output_path: str | None = None,
) -> bytes:
    """Generate natural neural speech via persistent IPv4 Edge-TTS WebSocket + RAM LRU cache."""
    import edge_tts

    cleaned = clean_text_for_tts(text)
    if not cleaned:
        cleaned = "..."

    try:
        spd = float(speed)
    except (ValueError, TypeError):
        spd = 1.15
    pct = int(round((spd - 1.0) * 100))
    rate_str = f"{pct:+d}%"

    cache_key = (cleaned, voice, rate_str)
    if cache_key in _TTS_CACHE:
        mp3_bytes = _TTS_CACHE[cache_key]
    else:
        mp3_bytes = b""
        try:
            mp3_bytes = await _EDGE_POOL.synthesize(cleaned, voice=voice, rate_str=rate_str)
        except Exception:
            mp3_bytes = b""

        if not mp3_bytes:
            connector = aiohttp.TCPConnector(family=socket.AF_INET)
            communicate = edge_tts.Communicate(cleaned, voice=voice, rate=rate_str, connector=connector)
            mp3_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])
            mp3_bytes = mp3_buffer.getvalue()

        if mp3_bytes and len(_TTS_CACHE) < 512 and len(cleaned) < 350:
            _TTS_CACHE[cache_key] = mp3_bytes

    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext == ".wav":
            await asyncio.to_thread(mp3_bytes_to_wav_file, mp3_bytes, output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(mp3_bytes)

    return mp3_bytes


def free_ai_chat_sync(messages: list[dict], model: str = "gpt-4.1-mini", max_tokens: int = 60) -> str:
    """Synchronous fast LLM call (used by classic Dashboard)."""
    api_key = get_active_openai_key()
    fast_model = "gpt-4.1-mini" if model in {"openai", "openai-fast", "llama", "mistral", "gpt-4.1-nano"} else model
    try:
        resp = _SYNC_SESSION.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": fast_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                return content
    except Exception as e:
        print(f"Sync OpenAI warning: {e}")

    return "Eshitaman, marhamat."


async def fast_llm_reply_async(
    messages: list[dict],
    model: str = "gpt-5.4-nano",
    provider: str = "free_ai",
    max_tokens: int = 160,
) -> str:
    """Ultra-low-latency async LLM call using persistent IPv4 connection pool & gpt-5.4-nano."""
    client = get_http_client()

    api_key = get_active_openai_key()
    fast_model = (
        "gpt-5.4-nano"
        if model in {"openai", "openai-fast", "llama", "mistral", "llama3.2", "gpt-4.1-nano", ""}
        else model
    )

    payload: dict = {
        "model": fast_model,
        "messages": messages,
    }
    if fast_model.startswith("gpt-5"):
        payload["reasoning_effort"] = "none"
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = 0.2
        payload["max_tokens"] = max_tokens

    try:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code == 200:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                return content
        elif fast_model != "gpt-4.1-mini":
            resp2 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4.1-mini",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
            )
            if resp2.status_code == 200:
                content = (
                    resp2.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if content:
                    return content
    except Exception as e:
        print(f"Async fast LLM error: {e}")

    return FALLBACK_ERROR_REPLY


async def warmup_connections():
    """Pre-warm OpenAI IPv4 TLS, Persistent Edge-TTS WebSocket, and universal call greeting in RAM."""
    try:
        client = get_http_client()
        api_key = get_active_openai_key()
        await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        for v in ("uz-UZ-SardorNeural", "uz-UZ-MadinaNeural"):
            await generate_edge_tts_audio(CALL_CONNECT_GREETING, voice=v, speed=1.15)
        print("Universal Voice AI audio cache & connections pre-warmed!")
    except Exception as e:
        print(f"Warmup info: {e}")


@router.on_event("startup")
async def _on_startup_warmup():
    asyncio.create_task(warmup_connections())


@router.get("/edge_tts_voices")
async def get_edge_tts_voices():
    return {"voices": EDGE_TTS_VOICES}


@router.get("/free_ai_models")
async def get_free_ai_models():
    return {"models": FREE_AI_MODELS}


@router.get("/api/call/faq")
async def get_sahiy_faq():
    return {
        "company": "Sahiy",
        "faq": [
            {"id": item["id"], "question": item["question"], "answer": item["answer"]}
            for item in SAHIY_FAQ_ITEMS
            if item["id"] not in {"greeting", "how_are_you", "thanks", "goodbye"}
        ],
    }


@router.get("/api/call/network_info")
async def get_network_info():
    local_ip = "192.168.68.106"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    return {
        "local_ip": local_ip,
        "wifi_url": f"http://{local_ip}:8000/call",
        "https_url": os.getenv(
            "PUBLIC_HTTPS_URL",
            "https://scoop-shakespeare-tier-legs.trycloudflare.com/call",
        ),
    }


@router.get("/api/call/reflex_pack")
async def api_call_reflex_pack(voice: str = "uz-UZ-SardorNeural", speed: float = 1.15, language: str = "uz-UZ"):
    bridges = VOICE_BRIDGES.get(language, VOICE_BRIDGES["uz-UZ"])
    q_bytes = await generate_edge_tts_audio(bridges["q_opener"], voice=voice, speed=speed)
    s_bytes = await generate_edge_tts_audio(bridges["s_opener"], voice=voice, speed=speed)
    return {
        "voice": voice,
        "q_text": bridges["q_opener"],
        "q_audio_b64": base64.b64encode(q_bytes).decode("ascii"),
        "s_text": bridges["s_opener"],
        "s_audio_b64": base64.b64encode(s_bytes).decode("ascii"),
    }


def _build_call_messages(user_text: str, character: str, history: list, language: str) -> list[dict]:
    system_prompt = get_character_system_prompt(character)
    voice_style_rule = (
        "USLUB: Hech qanday mavzu cheklovi yo'q — foydalanuvchining istalgan savoliga erkin, to'liq va aniq javob bering. "
        "Jonli ovozli suhbat bo'lgani uchun ravon, samimiy va tushunarli gapiring (odatda 1-3 gap). "
        "Har safar gapni 'Albatta' deb boshlamang."
    )

    language_code = (language or "auto").lower().split("-")[0]
    language_rule = {
        "uz": "O'zbek tilida, lotin yozuvida javob bering.",
        "ru": "Отвечайте на русском языке.",
        "en": "Reply in English.",
        "tr": "Türkçe yanıt verin.",
    }.get(language_code, "Reply in the language the customer is using; do not translate their words unnecessarily.")
    recognition_rule = (
        "The user's words may come from speech recognition. Preserve names, numbers, order IDs, "
        "quantities and negation. Never silently rewrite an uncertain phrase into an assumed order "
        "or action. Ask one short clarifying question when meaning is unclear."
    )
    messages = [{"role": "system", "content": f"{system_prompt}\n\n{voice_style_rule}\n{language_rule}\n{recognition_rule}"}]
    for item in history[-8:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def _match_sahiy_faq(user_text: str) -> str | None:
    """Instant greeting only for automatic call-connect handshake; no topic restrictions."""
    if user_text.strip() == "__call_connect__":
        return CALL_CONNECT_GREETING
    return None


async def _process_text_turn(
    user_text: str,
    character: str,
    history: list,
    language: str,
    provider: str,
    model: str,
    voice: str,
    speed: float,
    used_bridge: bool,
    needs_confirmation: bool = False,
) -> dict:
    t0 = time.perf_counter()

    faq_reply = _match_sahiy_faq(user_text)
    if faq_reply:
        reply_text = strip_duplicate_opener(faq_reply) if used_bridge else faq_reply
        t_llm = time.perf_counter()
        mp3_bytes = await generate_edge_tts_audio(text=reply_text, voice=voice, speed=speed)
        t_tts = time.perf_counter()
        return {
            "user_text": user_text,
            "reply": reply_text,
            "audio_b64": base64.b64encode(mp3_bytes).decode("ascii"),
            "llm_ms": int((t_llm - t0) * 1000),
            "tts_ms": int((t_tts - t_llm) * 1000),
            "latency_ms": int((t_tts - t0) * 1000),
            "instant": True,
            "character": character,
        }

    messages = _build_call_messages(user_text, character, history, language)
    if needs_confirmation:
        messages[0]["content"] += (
            "\nThe audio decoder flagged this utterance as uncertain. Ask the customer to confirm "
            "or repeat the unclear part before interpreting an order, amount, number or instruction."
        )
    reply_text = await fast_llm_reply_async(messages, model=model, provider=provider, max_tokens=160)
    reply_text = clean_text_for_tts(reply_text)
    if used_bridge:
        reply_text = strip_duplicate_opener(reply_text)
    t_llm = time.perf_counter()

    mp3_bytes = await generate_edge_tts_audio(text=reply_text, voice=voice, speed=speed)
    t_tts = time.perf_counter()

    return {
        "user_text": user_text,
        "reply": reply_text,
        "audio_b64": base64.b64encode(mp3_bytes).decode("ascii"),
        "llm_ms": int((t_llm - t0) * 1000),
        "tts_ms": int((t_tts - t_llm) * 1000),
        "latency_ms": int((t_tts - t0) * 1000),
        "instant": False,
        "character": character,
    }


@router.post("/api/call/voice_turn")
async def api_call_voice_turn(request: Request):
    data = await request.json()
    user_text = (data.get("message") or "").strip()
    character = data.get("character") or "ai_call_center"
    history = data.get("history") or []
    language = data.get("language") or "uz-UZ"
    provider = data.get("provider") or "free_ai"
    model = data.get("model") or "gpt-5.4-nano"
    voice = data.get("voice") or "uz-UZ-SardorNeural"
    speed = float(data.get("speed") or 1.15)
    used_bridge = bool(data.get("used_bridge", False))
    needs_confirmation = data.get("needs_confirmation") is True

    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    work = _process_text_turn(
        user_text=user_text,
        character=character,
        history=history,
        language=language,
        provider=provider,
        model=model,
        voice=voice,
        speed=speed,
        used_bridge=used_bridge,
        needs_confirmation=needs_confirmation,
    )
    return await _until_client_disconnect(request, work)


async def _until_client_disconnect(request: Request, work):
    """Abort obsolete LLM/TTS work when the browser barges in or hangs up."""
    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return
    operation = asyncio.create_task(work)
    watcher = asyncio.create_task(disconnected())
    try:
        async with asyncio.timeout(60):
            done, _ = await asyncio.wait({operation, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                return Response(status_code=499)
            return operation.result()
    finally:
        for task in (operation, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation, watcher, return_exceptions=True)


def _is_hallucinated_transcript(text: str) -> bool:
    # Text alone cannot prove hallucination. "ha", "7", a phone number, a URL
    # and "thank you" can all be valid customer answers. Reject only no content;
    # speech evidence belongs to VAD/decoder confidence, not a word blacklist.
    return not any(ch.isalnum() for ch in (text or ""))


async def transcribe_audio_fast(raw_bytes: bytes, content_type: str = "audio/webm", language: str = "auto") -> str:
    """Legacy explicit file-upload ASR; live calls use the WebSocket pipeline."""
    import av
    import numpy as np

    wav_io = io.BytesIO()
    try:
        container = av.open(io.BytesIO(raw_bytes))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm_chunks = []
        for frame in container.decode(audio=0):
            rframes = resampler.resample(frame)
            if not isinstance(rframes, list):
                rframes = [rframes] if rframes is not None else []
            for rf in rframes:
                pcm_chunks.append(rf.to_ndarray().tobytes())
        for rf in resampler.resample(None):
            pcm_chunks.append(rf.to_ndarray().tobytes())
        container.close()

        pcm_data = b"".join(pcm_chunks)
        if len(pcm_data) < 3200:  # 100 ms; short confirmations are meaningful.
            return ""
        # Do not reject quiet speech using a hard amplitude threshold. The live
        # path has trained VAD; this explicit batch endpoint rejects digital zero.
        if not np.any(np.frombuffer(pcm_data, dtype="<i2")):
            return ""

        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_data)
        upload_bytes = wav_io.getvalue()
        upload_name = "speech.wav"
        upload_mime = "audio/wav"
    except Exception:
        return ""

    client = get_http_client()
    api_key = get_active_openai_key()

    try:
        stt_data = {"model": "gpt-4o-mini-transcribe"}
        language_code = (language or "auto").lower().split("-")[0]
        if language_code in {"ru", "en", "tr"}:
            stt_data["language"] = language_code
        vocab_hints = (os.getenv("STT_VOCABULARY_HINTS") or "").strip()
        if vocab_hints:
            stt_data["prompt"] = vocab_hints[:300]

        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (upload_name, upload_bytes, upload_mime)},
            data=stt_data,
        )
        if resp.status_code == 200:
            raw_text = (resp.json().get("text") or "").strip()
            text = clean_transcript(raw_text)
            if _is_hallucinated_transcript(text):
                return ""
            return text
    except Exception as e:
        print(f"Fast STT error: {e}")

    return ""


@router.post("/api/call/voice_turn_audio")
async def api_call_voice_turn_audio(
    file: UploadFile = File(...),
    character: str = Form("ai_call_center"),
    language: str = Form("uz-UZ"),
    voice: str = Form("uz-UZ-SardorNeural"),
    speed: float = Form(1.15),
    used_bridge: bool = Form(False),
    history_json: str = Form("[]"),
):
    """Legacy file-upload Audio -> LLM -> TTS endpoint."""
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) < 1200:
        return JSONResponse({"user_text": "", "reply": "", "audio_b64": ""})

    user_text = await transcribe_audio_fast(raw_bytes, file.content_type or "audio/webm", language)
    stt_ms = int((time.perf_counter() - t0) * 1000)

    cleaned_user = user_text.strip()
    if _is_hallucinated_transcript(cleaned_user):
        return JSONResponse({"user_text": "", "reply": "", "audio_b64": ""})

    try:
        history = json.loads(history_json) if history_json else []
    except Exception:
        history = []

    res = await _process_text_turn(
        user_text=cleaned_user,
        character=character,
        history=history,
        language=language,
        provider="free_ai",
        model="gpt-5.4-nano",
        voice=voice,
        speed=speed,
        used_bridge=used_bridge,
    )
    res["stt_ms"] = stt_ms
    return res


@router.post("/api/call/chat")
async def api_call_chat(request: Request):
    data = await request.json()
    user_text = (data.get("message") or "").strip()
    character = data.get("character") or "ai_call_center"
    history = data.get("history") or []
    language = data.get("language") or "uz-UZ"
    provider = data.get("provider") or "free_ai"
    model = data.get("model") or "gpt-5.4-nano"

    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    messages = _build_call_messages(user_text, character, history, language)
    t0 = time.perf_counter()
    reply_text = await fast_llm_reply_async(messages, model=model, provider=provider, max_tokens=160)
    reply_text = clean_text_for_tts(reply_text)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "reply": reply_text,
        "latency_ms": latency_ms,
        "character": character,
    }


@router.post("/api/call/tts")
async def api_call_tts(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or os.getenv("EDGE_TTS_VOICE", "uz-UZ-SardorNeural")
    speed = float(data.get("speed") or os.getenv("VOICE_SPEED", "1.15"))

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    mp3_bytes = await generate_edge_tts_audio(text=text, voice=voice, speed=speed)
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@router.post("/api/call/transcribe")
async def api_call_transcribe(
    file: UploadFile = File(...),
    language: str = Form("uz"),
):
    raw_bytes = await file.read()
    text = await transcribe_audio_fast(raw_bytes, file.content_type or "audio/webm", language)
    return {"text": text}
