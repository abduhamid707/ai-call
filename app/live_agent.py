"""Cancelable streaming replies for the standalone live transcription page."""
import asyncio
import base64
import json
import os
import re

import httpx

from .elevenlabs_call import _build_call_messages, generate_edge_tts_audio, get_active_openai_key

_VOICES = {"uz": "uz-UZ-SardorNeural", "ru": "ru-RU-DmitryNeural", "en": "en-US-AndrewMultilingualNeural", "tr": "tr-TR-AhmetNeural"}


class LiveAgent:
    def __init__(self, emit, language="uz"):
        self.emit, self.language = emit, language
        self.current_language = language
        self.history = []
        self.task = self.active_turn = None
        self.pending_answer = None
        self.control_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15))

    async def interrupt(self):
        async with self.control_lock:
            await self._interrupt_locked()

    async def _interrupt_locked(self):
        task, turn_id = self.task, self.active_turn
        self.task = self.active_turn = self.pending_answer = None
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if turn_id:
            await self.emit({"type": "agent_cancelled", "turn_id": turn_id})

    async def close(self):
        try:
            await self.interrupt()
        finally:
            await self.http.aclose()

    async def submit(self, turn_id, question, needs_confirmation=False, language=None):
        async with self.control_lock:
            await self._interrupt_locked()
            self.current_language = language if self.language == "auto" and language in _VOICES else self.language
            self.active_turn = turn_id
            self.task = asyncio.create_task(self._reply(turn_id, question, needs_confirmation))

    async def acknowledge(self, turn_id):
        """Only fully played speech becomes an assistant history message."""
        async with self.control_lock:
            if self.pending_answer and self.active_turn == turn_id and self.pending_answer[0] == turn_id:
                self.history.append({"role": "assistant", "content": self.pending_answer[1]})
                self.history[:] = self.history[-14:]
                self.pending_answer = None

    async def _reply(self, turn_id, question, needs_confirmation):
        synthesis = []
        reply_language = self.current_language
        try:
            key = get_active_openai_key()
            if not key:
                await self.emit({"type": "agent_error", "turn_id": turn_id, "message": "Matn tayyor. AI javobi uchun OPENAI_API_KEY sozlang yoki faqat transkripsiyani tanlang."})
                return
            messages = _build_call_messages(question, "ai_call_center", self.history, reply_language)
            if os.getenv("LIVE_AGENT_PROMPT"):
                messages[0]["content"] += "\n" + os.environ["LIVE_AGENT_PROMPT"]
            if needs_confirmation:
                messages[0]["content"] += "\nAudio recognition is uncertain. Ask a brief clarifying question; do not assume the order, number, or intended action."
            self.history.append({"role": "user", "content": question})
            self.history[:] = self.history[-14:]
            await self.emit({"type": "agent_thinking", "turn_id": turn_id})
            text_parts, pending = [], ""
            async with self.http.stream(
                "POST", "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": os.getenv("LIVE_AGENT_MODEL", "gpt-4o-mini"), "messages": messages,
                      "stream": True, "max_completion_tokens": 240},
            ) as response:
                response.raise_for_status()
                # Retain incomplete SSE lines across network chunks.
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:].strip() == "[DONE]":
                        break
                    payload = json.loads(line[6:])
                    choices = payload.get("choices") or []
                    delta = choices[0].get("delta", {}).get("content", "") if choices else ""
                    if not delta:
                        continue
                    text_parts.append(delta)
                    pending += delta
                    await self.emit({"type": "agent_text_delta", "turn_id": turn_id, "delta": delta})
                    boundary = re.search(r"[.!?](?:\s|$)", pending)
                    if boundary and len(pending) >= 24 and len(synthesis) < 6:
                        piece, pending = pending[:boundary.end()], pending[boundary.end():]
                        synthesis.append(asyncio.create_task(self._synthesize(turn_id, len(synthesis), piece, key, reply_language)))
            if pending.strip():
                synthesis.append(asyncio.create_task(self._synthesize(turn_id, len(synthesis), pending, key, reply_language)))
            answer = "".join(text_parts).strip()
            if answer:
                self.pending_answer = (turn_id, answer)
            if synthesis:
                await asyncio.gather(*synthesis)
            await self.emit({"type": "agent_done", "turn_id": turn_id, "chunks": len(synthesis)})
        except asyncio.CancelledError:
            raise
        except Exception:
            self.pending_answer = None
            await self.emit({"type": "agent_error", "turn_id": turn_id, "message": "AI javobi yoki ovozida uzilish. Qayta gapiring."})
            await self.emit({"type": "agent_done", "turn_id": turn_id, "chunks": 0})
        finally:
            for task in synthesis:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*synthesis, return_exceptions=True)

    async def _synthesize(self, turn_id, index, text, key, language=None):
        audio = b""
        if os.getenv("LIVE_AGENT_TTS_PROVIDER", "edge_tts").lower() != "openai":
            try:
                audio = await generate_edge_tts_audio(text, voice=os.getenv("LIVE_AGENT_EDGE_VOICE", _VOICES.get(language or self.language, _VOICES["uz"])), speed=1.05)
            except Exception:
                audio = b""
        if not audio:
            response = await self.http.post(
                "https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {key}"},
                json={"model": os.getenv("LIVE_AGENT_TTS_MODEL", "gpt-4o-mini-tts"),
                      "voice": os.getenv("LIVE_AGENT_VOICE", "coral"), "input": text, "response_format": "mp3"},
            )
            response.raise_for_status()
            audio = response.content
        await self.emit({"type": "agent_audio", "turn_id": turn_id, "index": index,
                         "audio": base64.b64encode(audio).decode("ascii")})
