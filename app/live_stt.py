"""Browser microphone -> backend WS -> OpenAI Realtime transcription + Streaming AI Agent + Early/Parallel TTS."""
import asyncio
import base64
import json
import os
from pathlib import Path

import aiohttp
import websockets
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .speech_cleanup import clean_transcript
from .elevenlabs_call import get_character_system_prompt

load_dotenv()

router = APIRouter()

# OpenAI Realtime transcription language whitelist ('uz' is not in OpenAI's ISO whitelist and must use auto-detect + prompt)
_SUPPORTED_REALTIME_LANGS = {
    "af", "ar", "az", "be", "bg", "bs", "ca", "cs", "cy", "da", "de", "el",
    "en", "es", "et", "fa", "fi", "fr", "gl", "he", "hi", "hr", "hu", "hy",
    "id", "is", "it", "iw", "ja", "kk", "kn", "ko", "lt", "lv", "mi", "mk",
    "mr", "ms", "ne", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr",
    "sv", "sw", "ta", "th", "tl", "tr", "uk", "ur", "vi", "zh",
}


def transcription_session():
    language = os.getenv("LIVE_STT_LANGUAGE", "uz").strip().lower()
    prompt = os.getenv("LIVE_STT_VOCABULARY_HINTS", "O'zbek tilida (lotin alifbosida):").strip()
    transcription = {"model": os.getenv("LIVE_STT_MODEL", "gpt-4o-transcribe")}
    if language in _SUPPORTED_REALTIME_LANGS:
        transcription["language"] = language
    if prompt:
        transcription["prompt"] = prompt[:1000]
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": transcription,
                "noise_reduction": {"type": "near_field"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.35,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 600,
                },
            }
        },
    }


@router.get("/live-stt", response_class=HTMLResponse)
async def live_stt_page():
    return Path("app/templates/live_stt.html").read_text(encoding="utf-8")


@router.websocket("/ws/live-stt")
async def live_stt_socket(client: WebSocket):
    await client.accept()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        await client.send_json({"type": "error", "message": "OPENAI_API_KEY serverda sozlanmagan."})
        await client.close(code=1011)
        return

    upstream = None

    response_queue = asyncio.Queue(maxsize=4)
    handled = set()
    default_sys_prompt = get_character_system_prompt("ai_call_center")
    history = [
        {
            "role": "system",
            "content": os.getenv("LIVE_AGENT_PROMPT", default_sys_prompt),
        }
    ]
    send_lock = asyncio.Lock()

    async def emit(data):
        async with send_lock:
            await client.send_json(data)

    async def agent_worker():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=75)) as http:
            while True:
                turn_id, question = await response_queue.get()
                try:
                    history.append({"role": "user", "content": question})
                    history[:] = history[:1] + history[-14:]
                    await emit({"type": "agent_thinking", "turn_id": turn_id})
                    async with http.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={
                            "model": os.getenv("LIVE_AGENT_MODEL", "gpt-4o-mini"),
                            "messages": history,
                            "stream": True,
                            "max_completion_tokens": 220,
                        },
                    ) as reply:
                        if reply.status != 200:
                            raise RuntimeError("Chat API failure")
                        text_parts, pending, tasks = [], "", []
                        async for raw in reply.content:
                            for line in raw.splitlines():
                                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                                    continue
                                try:
                                    delta = (
                                        json.loads(line[6:])["choices"][0]
                                        .get("delta", {})
                                        .get("content")
                                        or ""
                                    )
                                except (ValueError, IndexError, KeyError, TypeError):
                                    continue
                                if not delta:
                                    continue
                                text_parts.append(delta)
                                pending += delta
                                await emit({"type": "agent_text_delta", "turn_id": turn_id, "delta": delta})
                                if len(tasks) < 7 and (any(c in pending for c in ".!?") or len(pending) > 110):
                                    piece, pending = pending, ""
                                    tasks.append(
                                        asyncio.create_task(synthesize(http, turn_id, len(tasks), piece))
                                    )
                        if pending.strip():
                            tasks.append(
                                asyncio.create_task(synthesize(http, turn_id, len(tasks), pending))
                            )
                    answer = "".join(text_parts).strip()
                    if answer:
                        history.append({"role": "assistant", "content": answer})
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        if any(isinstance(x, Exception) for x in results):
                            await emit({"type": "agent_error", "message": "Ovoz sintezida xato."})
                    await emit({"type": "agent_done", "turn_id": turn_id, "chunks": len(tasks)})
                except Exception:
                    await emit({"type": "agent_error", "message": "AI javobini olishda xato."})
                    await emit({"type": "agent_done", "turn_id": turn_id, "chunks": 0})
                finally:
                    response_queue.task_done()

    async def synthesize(http, turn_id, index, text):
        async with http.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.getenv("LIVE_AGENT_TTS_MODEL", "gpt-4o-mini-tts"),
                "voice": os.getenv("LIVE_AGENT_VOICE", "coral"),
                "input": text,
                "response_format": "mp3",
            },
        ) as reply:
            if reply.status != 200:
                raise RuntimeError("TTS API failure")
            audio = await reply.read()
        await emit(
            {
                "type": "agent_audio",
                "turn_id": turn_id,
                "index": index,
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )

    try:
        upstream = await websockets.connect(
            "wss://api.openai.com/v1/realtime?intent=transcription",
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=2**22,
            open_timeout=15,
        )
        await upstream.send(json.dumps({"type": "session.update", "session": transcription_session()}))
        await client.send_json({"type": "ready"})

        async def send_audio():
            while True:
                message = await client.receive()
                if message["type"] == "websocket.disconnect":
                    return
                packet = message.get("bytes")
                if packet:
                    # 24kHz mono signed 16-bit PCM, limited to 1 second per packet.
                    if len(packet) > 48000:
                        await client.send_json({"type": "error", "message": "Audio chunk juda katta."})
                        continue
                    await upstream.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(packet).decode("ascii"),
                            }
                        )
                    )
                elif message.get("text"):
                    try:
                        event = json.loads(message["text"])
                    except ValueError:
                        continue
                    if event.get("type") == "agent_utterance":
                        item_id = str(event.get("item_id") or "")[:120]
                        utterance = str(event.get("text") or "").strip()[:1200]
                        if item_id and utterance and item_id not in handled:
                            handled.add(item_id)
                            if not response_queue.full():
                                await response_queue.put((item_id, utterance))
                        continue
                    if event.get("type") == "stop":
                        return

        async def receive_transcripts():
            async for message in upstream:
                event = json.loads(message)
                kind = event.get("type")
                if kind in (
                    "conversation.item.input_audio_transcription.delta",
                    "conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.failed",
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "error",
                ):
                    raw_tx = event.get("transcript", "")
                    cleaned_tx = (
                        clean_transcript(raw_tx)
                        if kind == "conversation.item.input_audio_transcription.completed"
                        else raw_tx
                    )
                    if kind == "conversation.item.input_audio_transcription.completed":
                        low_tx = cleaned_tx.lower()
                        if "lotin alifbosida" in low_tx or low_tx in {"o'zbek tilida", "o'zbek tilida:"}:
                            continue
                    await emit(
                        {
                            "type": kind,
                            "item_id": event.get("item_id"),
                            "delta": event.get("delta", ""),
                            "transcript": cleaned_tx,
                            "message": (
                                event.get("error", {}).get("message", "")
                                if isinstance(event.get("error"), dict)
                                else ""
                            ),
                        }
                    )

        agent_task = asyncio.create_task(agent_worker())
        tasks = [asyncio.create_task(send_audio()), asyncio.create_task(receive_transcripts())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        agent_task.cancel()
        await asyncio.gather(agent_task, return_exceptions=True)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        pass
    except Exception:
        try:
            await client.send_json(
                {"type": "error", "message": "Streaming transkripsiya ulanmagan. Server logini tekshiring."}
            )
        except Exception:
            pass
    finally:
        if upstream:
            await upstream.close()
        try:
            await client.close()
        except Exception:
            pass
