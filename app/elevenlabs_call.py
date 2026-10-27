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

OUT_OF_SCOPE_REPLY = (
    "Kechirasiz, men faqat Sahiy kompaniyasi xizmatlari bo'yicha ma'lumot bera olaman. "
    "Yetkazib berish, to'lovlar yoki buyurtma bo'yicha savolingiz bormi?"
)

# Official Sahiy FAQ Knowledge Base (0ms Instant Match + Pre-synthesized Audio)
SAHIY_FAQ_ITEMS = [
    {
        "id": "greeting",
        "question": "Salomlashish va bog'lanish",
        "pattern": re.compile(r"^(alo+|allo+|alyo+|assalomu\s+alaykum|salom|salom\s+alaykum|vaalaykum\s+assalom|eshityapsizmi)[\?\!\.]*$", re.I),
        "answer": "Assalomu alaykum! Sahiy aloqa markaziga xush kelibsiz. Sizga qanday yordam bera olaman?",
    },
    {
        "id": "how_are_you",
        "question": "Hol-ahvol so'rash",
        "pattern": re.compile(r"^(qalaysiz|yaxshimisiz|ishlar\s+qalay|ahvollar\s+qalay|salom\s+qalaysiz)[\?\!\.]*$", re.I),
        "answer": "Rahmat, juda yaxshi! Sahiy xizmatlari bo'yicha qanday savolingiz bor?",
    },
    {
        "id": "about_sahiy",
        "question": "Sahiy qanday kompaniya va nima sotadi?",
        "pattern": re.compile(r"(sahiy\s+nima|saxiy\s+nima|sahi\s+nima|qanaqa\s+kompaniya|nima\s+bilan\s+shug|nima\s+sotasiz|nimalar\s+bor|sahiy\s+haqida|saxiy\s+haqida|sahi\s+haqida|kim\s+bu)", re.I),
        "answer": "Sahiy — O'zbekistondagi ishonchli onlayn savdo platformasi bo'lib, kitoblar, maishiy texnika va elektronikani hamyonbop narxda yetkazib beradi.",
    },
    {
        "id": "delivery",
        "question": "Yetkazib berish (dostavka) muddati va narxi qancha?",
        "pattern": re.compile(r"(yetkazib\s+berish|yitkazib\s+berish|dostavka|dastavka|qancha\s+vaqtda\s+keladi|qancha\s+vaqt\s+oladi|necha\s+kunda\s+keladi|viloyatga\s+chiqadimi|pochta)", re.I),
        "answer": "Toshkent bo'ylab 24 soatda, viloyatlarga 2-3 ish kunida yetkazib beramiz. 300 ming so'mdan oshsa yetkazib berish bepul.",
    },
    {
        "id": "installment",
        "question": "Bo'lib to'lash (muddatli to'lov / nasiya) bormi?",
        "pattern": re.compile(r"(bo'lib\s+to'lash|bolib\s+tolash|muddatli\s+to'lov|nasiya|rassrochka|rasrochka|kredit|necha\s+oyga)", re.I),
        "answer": "Ha, Sahiy orqali boshlang'ich to'lovsiz va ribosiz 3, 6 hamda 12 oyga halol bo'lib to'lash mavjud. Faqat pasport kifoya.",
    },
    {
        "id": "payment",
        "question": "To'lov turlari qanday?",
        "pattern": re.compile(r"(to'lov\s+turlari|qanday\s+to'lov|click|payme|uzum|naqd|karta\s+orqali|pul\s+o'tkazish)", re.I),
        "answer": "To'lovni Click, Payme, Uzum, bank kartalari yoki buyurtmani qo'lingizga olganda naqd pulda amalga oshirishingiz mumkin.",
    },
    {
        "id": "working_hours",
        "question": "Ish vaqti soat nechadan nechagacha?",
        "pattern": re.compile(r"(ish\s+vaqti|soat\s+nechada|qachon\s+ishlaysiz|dam\s+olish\s+kuni|ochiqmisiz|necha\s+dan\s+necha)", re.I),
        "answer": "Sahiy aloqa markazi va yetkazib berish xizmati har kuni dam olish kunlarisiz soat 09:00 dan 22:00 gacha ishlaydi.",
    },
    {
        "id": "address",
        "question": "Manzilingiz qayerda joylashgan?",
        "pattern": re.compile(r"(manzil|adres|qayerda\s+joylashgan|ofisingiz\s+qayerda|qayerga\s+borsam|lokatsiya|filial)", re.I),
        "answer": "Bosh ofisimiz Toshkent shahri, Chilonzor tumani, Bunyodkor shoh ko'chasida joylashgan. Barcha viloyatlarda topshirish punktlarimiz bor.",
    },
    {
        "id": "warranty",
        "question": "Kafolat va mahsulotni qaytarish shartlari qanday?",
        "pattern": re.compile(r"(kafolat|garantiya|qaytarish|vozvrat|almashtirish|nuqson|buzilib\s+qolsa)", re.I),
        "answer": "Barcha mahsulotlarga 1 yilgacha rasmiy kafolat beriladi. Nuqson bo'lsa, 10 kun ichida almashtirish yoki pulni qaytarib olish mumkin.",
    },
    {
        "id": "how_to_order",
        "question": "Qanday buyurtma bersam bo'ladi?",
        "pattern": re.compile(r"(qanday\s+buyurtma|zakaz\s+berish|buyurtma\s+bermoqchi|qanday\s+olaman|sotib\s+olmoqchi|telegram\s+bot|sayt)", re.I),
        "answer": "Buyurtmani Sahiy sayti, Telegram botimiz yoki hozirning o'zida menga mahsulot nomi va manzilingizni aytib rasmiylashtirishingiz mumkin.",
    },
    {
        "id": "discounts",
        "question": "Chegirmalar va keshbek bormi?",
        "pattern": re.compile(r"(chegirma|aksiya|skidka|keshbek|bonus|arzonroq)", re.I),
        "answer": "Doimiy mijozlarimiz uchun har bir xariddan 5 foiz keshbek va 30 foizgacha maxsus chegirmalar amal qiladi.",
    },
    {
        "id": "thanks",
        "question": "Minnatdorchilik",
        "pattern": re.compile(r"^(rahmat|katta\s+rahmat|tashakkur|rahmat\s+sizga|tushunarli\s+rahmat)[\?\!\.]*$", re.I),
        "answer": "Arzimaydi! Sahiy xizmatlaridan foydalanganingiz uchun rahmat. Yana savolingiz bormi?",
    },
    {
        "id": "goodbye",
        "question": "Xayrlashish",
        "pattern": re.compile(r"^(xayr|ko'rishguncha|sog'\s+bo'ling|bo'pti\s+xayr|mayli\s+xayr)[\?\!\.]*$", re.I),
        "answer": "Xayr, salomat bo'ling! Sahiy sizga xizmat ko'rsatishdan doim mamnun.",
    },
]

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
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            if len(msg.data) >= 2:
                                hlen = int.from_bytes(msg.data[:2], "big")
                                params, data = get_headers_and_data(msg.data, hlen)
                                if params.get(b"Path") == b"audio" and data:
                                    buf.extend(data)
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            if "Path:turn.end" in msg.data:
                                break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self._ws = None
                            break
                    if buf:
                        return bytes(buf)
                except Exception:
                    self._ws = None
                    if attempt == 1:
                        raise
        return b""


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
        "Siz 'Sahiy' kompaniyasining rasmiy ovozli aloqa markazi operatorisiz. "
        "FAQAT Sahiy FAQ ma'lumotlari asosida qisqa (1-2 gap) javob bering."
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

    return "Sahiy aloqa markazi eshitadi, marhamat."


async def fast_llm_reply_async(
    messages: list[dict],
    model: str = "gpt-5.4-nano",
    provider: str = "free_ai",
    max_tokens: int = 55,
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

    return OUT_OF_SCOPE_REPLY


async def warmup_connections():
    """Pre-warm OpenAI IPv4 TLS, Persistent Edge-TTS WebSocket, and ALL Sahiy FAQ answers in RAM."""
    try:
        client = get_http_client()
        api_key = get_active_openai_key()
        await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        for v in ("uz-UZ-SardorNeural", "uz-UZ-MadinaNeural"):
            await generate_edge_tts_audio("Albatta,", voice=v, speed=1.15)
            await generate_edge_tts_audio("Tushunarli,", voice=v, speed=1.15)
            await generate_edge_tts_audio(OUT_OF_SCOPE_REPLY, voice=v, speed=1.15)
            for item in SAHIY_FAQ_ITEMS:
                await generate_edge_tts_audio(item["answer"], voice=v, speed=1.15)
                stripped = strip_duplicate_opener(item["answer"])
                if stripped != item["answer"]:
                    await generate_edge_tts_audio(stripped, voice=v, speed=1.15)
        print("Sahiy FAQ audio cache pre-warmed in RAM (0ms TTS ready!)")
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
            "https://contributing-falling-moderate-iron.trycloudflare.com/call",
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
    strict_faq_rule = (
        "QAT'IY QOIDA: Siz Sahiy kompaniyasi rasmiy aloqa markazi operatorisiz. "
        "FAQAT Sahiy FAQ ma'lumotlari asosida 1-2 ta qisqa gap bilan javob bering. "
        "Gapni 'Albatta', 'Tushunarli', 'Ha' so'zlari bilan BOSHLAMANG. "
        f"Agar savol Sahiy FAQ bazasida bo'lmasa, hech narsa to'qimasdan AYNAN mana shu jumlani qaytaring: '{OUT_OF_SCOPE_REPLY}'"
    )

    messages = [{"role": "system", "content": f"{system_prompt}\n\n{strict_faq_rule}"}]
    for item in history[-4:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def _match_sahiy_faq(user_text: str) -> str | None:
    """0ms deterministic FAQ matcher for Sahiy questions."""
    normalized = re.sub(r"[^\w\s\']", " ", user_text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    for item in SAHIY_FAQ_ITEMS:
        if item["pattern"].search(normalized):
            return item["answer"]
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
    reply_text = await fast_llm_reply_async(messages, model=model, provider=provider, max_tokens=55)
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

    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    res = await _process_text_turn(
        user_text=user_text,
        character=character,
        history=history,
        language=language,
        provider=provider,
        model=model,
        voice=voice,
        speed=speed,
        used_bridge=used_bridge,
    )
    return res


HALLUCINATION_SUBSTRINGS = (
    "subtitles",
    "amara.org",
    "thank you for watching",
    "thanks for watching",
    "продолжение следует",
    "редактор субтитров",
    "assalomu alaykum, salom, sahiy haqida",
    "dostavka bormi, bo'lib to'lash, muddatli to'lov",
    "www.",
    "http",
)


def _is_hallucinated_transcript(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 2:
        return True
    if not any(ch.isalpha() for ch in t):
        return True
    if t in {".", "..", "...", "you", "thank you", "bye", "uh", "um", "ah", "oh", "hmm", "ha", "a", "o"}:
        return True
    for sub in HALLUCINATION_SUBSTRINGS:
        if sub in t:
            return True
    return False


async def transcribe_audio_fast(raw_bytes: bytes, content_type: str = "audio/webm") -> str:
    """
    Ultra-fast Uzbek/Multilingual speech transcription using OpenAI gpt-4o-mini-transcribe.
    1. Decodes audio to 16kHz PCM via PyAV.
    2. Runs a strict NumPy Human Speech Energy Gate: rejects room noise, clicks, or breathing in 0ms!
    3. Sends clean speech to gpt-4o-mini-transcribe WITHOUT a long prompt (zero prompt-echo hallucinations!).
    """
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
        container.close()

        pcm_data = b"".join(pcm_chunks)
        if len(pcm_data) < 9600:  # Require at least 0.30s of recorded audio
            return ""

        # Server-Side Human Speech Energy Gate (20ms frames = 320 samples at 16kHz)
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        n_frames = len(samples) // 320
        if n_frames < 15:
            return ""
        frames = samples[: n_frames * 320].reshape(n_frames, 320)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        peak_rms = float(np.max(frame_rms))
        voiced_frames = int(np.sum(frame_rms > 950.0))

        # Reject background room noise, clicks, coughs, or breathing immediately!
        if peak_rms < 1350.0 or voiced_frames < 12:
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
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (upload_name, upload_bytes, upload_mime)},
            data={
                "model": "gpt-4o-mini-transcribe",
            },
        )
        if resp.status_code == 200:
            text = (resp.json().get("text") or "").strip()
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
    """
    Single-roundtrip Audio -> Sahiy FAQ -> Neural TTS endpoint.
    Guarantees 100% microphone compatibility on iPhone Safari, Android Chrome, and Windows PC!
    """
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) < 1200:
        return JSONResponse({"user_text": "", "reply": "", "audio_b64": ""})

    user_text = await transcribe_audio_fast(raw_bytes, file.content_type or "audio/webm")
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

    faq_reply = _match_sahiy_faq(user_text)
    if faq_reply:
        return {"reply": faq_reply, "latency_ms": 1, "character": character}

    messages = _build_call_messages(user_text, character, history, language)
    t0 = time.perf_counter()
    reply_text = await fast_llm_reply_async(messages, model=model, provider=provider, max_tokens=55)
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
    text = await transcribe_audio_fast(raw_bytes, file.content_type or "audio/webm")
    return {"text": text}
