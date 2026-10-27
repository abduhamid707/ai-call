"""Browser microphone -> backend WS -> OpenAI Realtime transcription, API key stays server-side."""
import asyncio
import base64
import json
import os

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

from dotenv import load_dotenv
from .speech_cleanup import clean_transcript

load_dotenv()

router = APIRouter()

# OpenAI Realtime transcription language whitelist (note: 'uz' is not in OpenAI's ISO whitelist and must use auto-detect + prompt)
_SUPPORTED_REALTIME_LANGS = {
    "af", "ar", "az", "be", "bg", "bs", "ca", "cs", "cy", "da", "de", "el",
    "en", "es", "et", "fa", "fi", "fr", "gl", "he", "hi", "hr", "hu", "hy",
    "id", "is", "it", "iw", "ja", "kk", "kn", "ko", "lt", "lv", "mi", "mk",
    "mr", "ms", "ne", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr",
    "sv", "sw", "ta", "th", "tl", "tr", "uk", "ur", "vi", "zh",
}


def transcription_session(silence_ms: int = 420):
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
                    "threshold": 0.5,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": silence_ms,
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
                    await upstream.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(packet).decode("ascii"),
                    }))
                elif message.get("text"):
                    try:
                        event = json.loads(message["text"])
                    except ValueError:
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
                    cleaned_tx = clean_transcript(raw_tx) if kind == "conversation.item.input_audio_transcription.completed" else raw_tx
                    if kind == "conversation.item.input_audio_transcription.completed":
                        low_tx = cleaned_tx.lower()
                        if "lotin alifbosida" in low_tx or low_tx in {"o'zbek tilida", "o'zbek tilida:"}:
                            continue
                    await client.send_json({
                        "type": kind,
                        "item_id": event.get("item_id"),
                        "delta": event.get("delta", ""),
                        "transcript": cleaned_tx,
                        "message": event.get("error", {}).get("message", "") if isinstance(event.get("error"), dict) else "",
                    })

        tasks = [asyncio.create_task(send_audio()), asyncio.create_task(receive_transcripts())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
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
            await client.send_json({"type": "error", "message": "Streaming transkripsiya ulanmagan. Server logini tekshiring."})
        except Exception:
            pass
    finally:
        if upstream:
            await upstream.close()
        try:
            await client.close()
        except Exception:
            pass
