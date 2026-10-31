"""Explicit hosted adapter retaining the existing server-VAD ASR model."""
import asyncio
import base64
import json
import os

import websockets

from .speech_cleanup import clean_transcript

# Local Whisper supports uz explicitly. Retain the working hosted API language
# subset and pass Uzbek as context; don't assume every provider accepts its code.
_SUPPORTED_REALTIME_LANGS = {"ru", "en", "tr"}
_CONTEXT = {
    "uz": "O'zbekcha suhbat. buyurtma, buyurtmam, yetkazib berish, to'lov, manzil, qaytarish, Sahiy, kuryer, postamat.",
    "ru": "Разговор с клиентом. заказ, доставка, оплата, адрес, возврат, курьер, Sahiy.",
    "en": "Customer conversation. order, delivery, payment, address, return, courier, Sahiy.",
    "auto": "Customer conversation in Uzbek, Russian or English. buyurtma, yetkazib berish, заказ, доставка, order, delivery, Sahiy.",
}


def transcription_session(language=None, silence_ms=None):
    language = (language or os.getenv("LIVE_STT_LANGUAGE", "uz")).split("-")[0].lower()
    model = os.getenv("LIVE_STT_MODEL", "gpt-4o-transcribe")
    if model in {"gpt-live-transcribe", "gpt-realtime-whisper"}:
        raise ValueError("Bu hosted adapter server_vad talab qiladi; LIVE_STT_MODEL=gpt-4o-transcribe ishlating.")
    transcription = {"model": model}
    if language in _SUPPORTED_REALTIME_LANGS:
        transcription["language"] = language
    prompt = os.getenv("LIVE_STT_VOCABULARY_HINTS", _CONTEXT.get(language, _CONTEXT["auto"])).strip()
    if prompt:
        transcription["prompt"] = prompt[:1000]
    return {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": transcription,
        "noise_reduction": {"type": "near_field"},
        "turn_detection": {"type": "server_vad", "threshold": 0.35, "prefix_padding_ms": 500,
                           "silence_duration_ms": silence_ms or int(os.getenv("LIVE_STT_SILENCE_MS", "600"))},
    }}}


class OpenAITranscriptionSession:
    def __init__(self, language, silence_ms, emit):
        self.language, self.silence_ms, self.emit = language, silence_ms, emit
        self.socket = self.reader = None
        self.updated, self.changed = asyncio.Event(), asyncio.Event()
        self.pending = set()
        self.speaking = False
        self.failure = None

    async def start(self):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("Hosted transkripsiya uchun OPENAI_API_KEY sozlanmagan.")
        self.socket = await websockets.connect(
            "wss://api.openai.com/v1/realtime?intent=transcription",
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=2**22, max_queue=16, open_timeout=15,
        )
        self.reader = asyncio.create_task(self._receive())
        await self.socket.send(json.dumps({"type": "session.update", "session": transcription_session(self.language, self.silence_ms)}))
        await asyncio.wait_for(self.updated.wait(), 15)
        self._raise_failure()
        return {"provider": "openai", "sample_rate": 24000, "language": self.language, "silence_ms": self.silence_ms}

    def _raise_failure(self):
        if self.failure:
            raise RuntimeError(self.failure)

    async def feed(self, pcm):
        self._raise_failure()
        await self.socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}))

    async def flush(self):
        self._raise_failure()
        # Ack is a barrier after prior appends. Commit active speech explicitly:
        # stopping the microphone alone sends no silence to trigger server VAD.
        self.updated.clear()
        await self.socket.send(json.dumps({"type": "session.update", "session": {
            "type": "transcription", "audio": {"input": {"turn_detection": None}},
        }}))
        await asyncio.wait_for(self.updated.wait(), 15)
        self._raise_failure()
        if self.speaking:
            await self.socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        async with asyncio.timeout(20):
            while self.pending:
                self.changed.clear()
                self._raise_failure()
                await self.changed.wait()
        self._raise_failure()

    async def _receive(self):
        try:
            async for raw in self.socket:
                event = json.loads(raw)
                kind, item_id = event.get("type", ""), event.get("item_id")
                if kind in {"session.updated", "transcription_session.updated"}:
                    self.updated.set()
                elif kind == "input_audio_buffer.speech_started":
                    self.speaking = True
                    self.pending.add(item_id)
                    await self.emit({"type": kind, "item_id": item_id})
                elif kind == "input_audio_buffer.speech_stopped":
                    self.speaking = False
                    await self.emit({"type": kind, "item_id": item_id})
                elif kind == "input_audio_buffer.committed":
                    self.speaking = False
                    self.pending.add(item_id)
                elif kind == "conversation.item.input_audio_transcription.delta":
                    await self.emit({"type": kind, "item_id": item_id, "delta": event.get("delta", "")})
                elif kind == "conversation.item.input_audio_transcription.completed":
                    self.pending.discard(item_id)
                    text = str(event.get("transcript") or "")
                    await self.emit({"type": kind, "item_id": item_id, "transcript": clean_transcript(text),
                                     "raw_transcript": text, "language": self.language, "needs_confirmation": False})
                    self.changed.set()
                elif kind == "conversation.item.input_audio_transcription.failed":
                    self.pending.discard(item_id)
                    self.failure = "Hosted transkripsiya gapni yakunlay olmadi. Qayta ulang."
                    self.changed.set()
                    self.updated.set()
                    await self.emit({"type": "error", "message": self.failure})
                elif kind == "error":
                    self.failure = "Hosted transkripsiya sozlamasi yoki API ulanishida xato."
                    self.changed.set()
                    self.updated.set()
                    await self.emit({"type": "error", "message": self.failure})
                    return
            self.failure = self.failure or "Hosted transkripsiya ulanishi yopildi."
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failure = "Hosted transkripsiya ulanishi uzildi."
        finally:
            self.changed.set()
            self.updated.set()

    async def close(self):
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.socket:
            await self.socket.close()
