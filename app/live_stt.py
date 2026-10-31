"""Browser PCM -> revisable captions -> exactly one finalized customer turn."""
import asyncio
import json
import logging
import os
from collections import OrderedDict, deque
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .live_agent import LiveAgent
from .openai_live_stt import OpenAITranscriptionSession, transcription_session

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)
LANGUAGES = {"uz", "ru", "en", "tr", "auto"}
_active_sessions = 0


def normalize_language(value: str | None) -> str:
    language = (value or os.getenv("LIVE_STT_LANGUAGE", "uz")).strip().lower().split("-")[0]
    if language not in LANGUAGES:
        raise ValueError("Til uz, ru, en, tr yoki auto bo'lishi kerak.")
    return language


def normalize_silence(value: str | int | None) -> int:
    try:
        result = int(value if value is not None else os.getenv("LIVE_STT_SILENCE_MS", "600"))
    except (ValueError, TypeError) as exc:
        raise ValueError("Sukut muddati 300–1500 ms oralig'ida bo'lishi kerak.") from exc
    if not 300 <= result <= 1500:
        raise ValueError("Sukut muddati 300–1500 ms oralig'ida bo'lishi kerak.")
    return result


def live_config(language=None, silence_ms=None) -> dict:
    provider = os.getenv("LIVE_STT_PROVIDER", "local").strip().lower()
    if provider not in {"local", "openai"}:
        raise ValueError("LIVE_STT_PROVIDER local yoki openai bo'lishi kerak.")
    return {
        "provider": provider,
        "sample_rate": 16000 if provider == "local" else 24000,
        "language": normalize_language(language),
        "silence_ms": normalize_silence(silence_ms),
    }


@router.get("/api/live-stt/config")
async def live_stt_config():
    """Configuration, not model readiness. No model load/download on a GET."""
    try:
        return live_config()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/live-stt", response_class=HTMLResponse)
async def live_stt_page():
    return (Path(__file__).parent / "templates" / "live_stt.html").read_text(encoding="utf-8")


class TranscriptRelay:
    """Coalesce bounded acoustic windows without answering mid-sentence."""

    def __init__(self, emit, agent=None):
        self.emit, self.agent = emit, agent
        self.latest_turn = None
        self.segments, self.raw_segments, self.uncertain = {}, {}, {}
        self.completed = OrderedDict()
        self.order = deque()
        self.finals = {}
        self.lock = asyncio.Lock()
        self.unsent = []
        self.unsent_uncertain = False
        self.stopping = False

    async def __call__(self, event):
        async with self.lock:
            await self._handle(event)

    async def _handle(self, event):
        event = dict(event)
        kind = event.get("type")
        turn_id = event.get("turn_id") or event.get("item_id")
        if turn_id:
            event["item_id"] = turn_id
        if kind == "input_audio_buffer.speech_started":
            self.latest_turn = turn_id
            if turn_id and turn_id not in self.order and turn_id not in self.completed:
                self.order.append(turn_id)
                if len(self.order) > 32:
                    raise RuntimeError("Transkripsiya navbati to'ldi. Qayta ulang.")
            if self.agent:
                await self.agent.interrupt()
        elif kind == "transcript.partial":
            event["transcript"] = " ".join(
                self.segments.get(turn_id, []) + [event.get("transcript", "")]
            ).strip()
        elif kind == "conversation.item.input_audio_transcription.completed":
            if not turn_id or turn_id in self.completed:
                return
            text = str(event.get("transcript") or "").strip()
            raw = str(event.get("raw_transcript", text) or "").strip()
            self.segments.setdefault(turn_id, []).append(text)
            self.raw_segments.setdefault(turn_id, []).append(raw)
            self.uncertain[turn_id] = self.uncertain.get(turn_id, False) or bool(event.get("needs_confirmation"))
            if event.get("end_of_turn") is False:
                await self.emit({"type": "transcript.partial", "item_id": turn_id,
                                 "transcript": " ".join(self.segments[turn_id]).strip()})
                return
            event["transcript"] = " ".join(self.segments.pop(turn_id)).strip()
            event["raw_transcript"] = " ".join(self.raw_segments.pop(turn_id)).strip()
            event["needs_confirmation"] = self.uncertain.pop(turn_id)
            self.completed[turn_id] = True
            if len(self.completed) > 256:
                self.completed.popitem(last=False)
            if turn_id not in self.order:
                self.order.append(turn_id)
            self.finals[turn_id] = event
            # Hosted completion order is not guaranteed. Deliver final customer
            # turns in speech order, without losing the earlier question.
            while self.order and self.order[0] in self.finals:
                final_id = self.order.popleft()
                final = self.finals.pop(final_id)
                superseded = self.latest_turn is not None and final_id != self.latest_turn
                final["superseded"] = superseded
                if final["transcript"]:
                    self.unsent.append(final["transcript"])
                    self.unsent_uncertain |= final["needs_confirmation"]
                if sum(map(len, self.unsent)) > 24000:
                    raise ValueError("Transkripsiya navbati to'ldi. Qo'ng'iroqni qayta boshlang.")
                if not superseded:
                    final["agent_text"] = " ".join(self.unsent)
                    final["needs_confirmation"] |= self.unsent_uncertain
                    self.unsent.clear()
                    self.unsent_uncertain = False
                await self.emit(final)
                if self.agent and not superseded and not self.stopping and final.get("agent_text"):
                    await self.agent.submit(final_id, final["agent_text"], final["needs_confirmation"], language=final.get("language"))
            return
        await self.emit(event)


def create_transcriber(config, emit):
    if config["provider"] == "openai":
        return OpenAITranscriptionSession(language=config["language"], silence_ms=config["silence_ms"], emit=emit)
    from .local_stt import LocalSTTSession
    return LocalSTTSession(language=config["language"], silence_ms=config["silence_ms"], emit=emit)


async def start_transcriber(client, transcriber, timeout=80):
    """Release abandoned cold-start sessions without cancelling native inference.

    The bounded native worker can finish a model download in the background;
    the disconnected browser no longer occupies a WebSocket admission slot.
    """
    initializing = asyncio.create_task(transcriber.start())
    receiving = asyncio.create_task(client.receive())
    try:
        done, _ = await asyncio.wait({initializing, receiving}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if receiving in done:
            message = receiving.result()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            raise ValueError("Mikrofonni model tayyor bo'lgandan keyin yuboring.")
        if initializing in done:
            return initializing.result()
        raise RuntimeError("Nutq modeli hali tayyor emas. Modelni oldindan yuklang va qayta ulang.")
    finally:
        for task in (initializing, receiving):
            if not task.done():
                task.cancel()
        await asyncio.gather(initializing, receiving, return_exceptions=True)


@router.websocket("/ws/live-stt")
async def live_stt_socket(client: WebSocket):
    global _active_sessions
    origin = client.headers.get("origin")
    if origin and urlsplit(origin).netloc.lower() != client.headers.get("host", "").lower():
        await client.close(code=1008)
        return
    await client.accept()
    transcriber = agent = None
    admitted = False
    send_lock = asyncio.Lock()

    async def emit(event):
        async with send_lock:
            await client.send_json(event)

    try:
        config = live_config(client.query_params.get("language"), client.query_params.get("silence_ms"))
        max_sessions = max(1, min(32, int(os.getenv("LIVE_STT_MAX_SESSIONS", "2"))))
        if _active_sessions >= max_sessions:
            await emit({"type": "error", "message": "Ovoz serveri band. Birozdan keyin qayta ulang."})
            await client.close(code=1013)
            return
        _active_sessions += 1
        admitted = True
        if client.query_params.get("agent", "false").lower() == "true":
            agent = LiveAgent(emit, language=config["language"])
        relay = TranscriptRelay(emit, agent)
        transcriber = create_transcriber(config, relay)
        await emit({"type": "loading", "message": "Nutq modeli tayyorlanmoqda…"})
        ready = await start_transcriber(client, transcriber)
        await emit({"type": "ready", **config, **ready})
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                break
            packet = message.get("bytes")
            if packet is not None:
                if not packet:
                    continue
                if len(packet) % 2 or len(packet) > config["sample_rate"] * 2:
                    raise ValueError("Audio mono PCM16 bo'lishi va paket 1 soniyadan oshmasligi kerak.")
                await transcriber.feed(packet)
            elif message.get("text"):
                if len(message["text"]) > 4096:
                    raise ValueError("Boshqaruv xabari juda katta.")
                try:
                    command = json.loads(message["text"])
                except ValueError as exc:
                    raise ValueError("Boshqaruv xabari JSON bo'lishi kerak.") from exc
                if not isinstance(command, dict):
                    raise ValueError("Boshqaruv xabari obyekt bo'lishi kerak.")
                if command.get("type") == "stop":
                    relay.stopping = True
                    if agent:
                        await agent.interrupt()
                    await asyncio.wait_for(transcriber.flush(), timeout=30)
                    await emit({"type": "stopped"})
                    break
                if command.get("type") == "interrupt" and agent:
                    await agent.interrupt()
                if command.get("type") == "agent_playback_done" and agent:
                    await agent.acknowledge(str(command.get("turn_id") or "")[:120])
                # Only server finals enter the agent; ignore legacy client echoes.
    except WebSocketDisconnect:
        pass
    except (ValueError, RuntimeError, TimeoutError) as exc:
        logger.warning("Live STT session failed: %s", type(exc).__name__)
        try:
            await emit({"type": "error", "message": str(exc) or "Nutq serveri javob bermadi."})
        except (RuntimeError, WebSocketDisconnect):
            pass
    except Exception:
        logger.warning("Live STT transport failed", exc_info=False)
        try:
            await emit({"type": "error", "message": "Jonli transkripsiyada uzilish. Qayta ulang."})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        cleanup = []
        if agent:
            cleanup.append(agent.close())
        if transcriber:
            cleanup.append(transcriber.close())
        await asyncio.gather(*cleanup, return_exceptions=True)
        if admitted:
            _active_sessions -= 1
        try:
            await client.close()
        except RuntimeError:
            pass
