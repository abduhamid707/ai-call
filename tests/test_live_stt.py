import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import live_stt
from app.live_stt import TranscriptRelay, live_config, transcription_session

def test_live_transcription_session_is_audio_only():
    session = transcription_session()
    assert session["type"] == "transcription"
    audio = session["audio"]["input"]
    assert audio["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio["transcription"]["model"] == "gpt-4o-transcribe"
    assert audio["turn_detection"]["type"] == "server_vad"
    assert audio["turn_detection"]["prefix_padding_ms"] >= 300


def test_local_is_default_and_language_selection_reaches_configuration(monkeypatch):
    monkeypatch.delenv("LIVE_STT_PROVIDER", raising=False)
    assert live_config("ru-RU", "800") == {
        "provider": "local", "language": "ru", "silence_ms": 800, "sample_rate": 16000,
    }
    assert live_config("auto")["language"] == "auto"
    with pytest.raises(ValueError):
        live_config("invalid")
    with pytest.raises(ValueError):
        live_config("uz", "50")


def test_hosted_language_and_model_configuration(monkeypatch):
    assert transcription_session("ru")["audio"]["input"]["transcription"]["language"] == "ru"
    assert "language" not in transcription_session("auto")["audio"]["input"]["transcription"]
    assert "buyurtma" in transcription_session("uz")["audio"]["input"]["transcription"]["prompt"]
    monkeypatch.setenv("LIVE_STT_MODEL", "gpt-live-transcribe")
    with pytest.raises(ValueError, match="server_vad"):
        transcription_session()


def test_long_turn_segments_remain_provisional_and_finalize_once():
    async def scenario():
        events = []
        async def emit(event):
            events.append(event)
        relay = TranscriptRelay(emit)
        await relay({"type": "input_audio_buffer.speech_started", "turn_id": "t", "item_id": "t:0"})
        await relay({"type": "conversation.item.input_audio_transcription.completed", "turn_id": "t", "item_id": "t:0", "transcript": "Buyurtma", "end_of_turn": False})
        assert events[-1]["type"] == "transcript.partial"
        await relay({"type": "transcript.partial", "turn_id": "t", "item_id": "t:1", "transcript": "raqami 17"})
        assert events[-1]["transcript"] == "Buyurtma raqami 17"
        final = {"type": "conversation.item.input_audio_transcription.completed", "turn_id": "t", "item_id": "t:1", "transcript": "raqami 178", "needs_confirmation": True, "end_of_turn": True}
        await relay(final)
        await relay(final)
        finals = [e for e in events if e["type"].endswith(".completed")]
        assert len(finals) == 1
        assert finals[0]["item_id"] == "t"
        assert finals[0]["transcript"] == "Buyurtma raqami 178"
        assert finals[0]["agent_text"] == "Buyurtma raqami 178"
        assert finals[0]["needs_confirmation"]
    asyncio.run(scenario())


def test_out_of_order_finals_keep_speech_order_and_do_not_reply_to_old_turn():
    async def scenario():
        events = []
        async def emit(event):
            events.append(event)
        relay = TranscriptRelay(emit)
        for turn in ("old", "new"):
            await relay({"type": "input_audio_buffer.speech_started", "item_id": turn})
        await relay({"type": "conversation.item.input_audio_transcription.completed", "item_id": "new", "transcript": "Bekor qilmang."})
        assert not any(e["type"].endswith(".completed") for e in events)
        await relay({"type": "conversation.item.input_audio_transcription.completed", "item_id": "old", "transcript": "178 buyurtmam."})
        finals = [e for e in events if e["type"].endswith(".completed")]
        assert [e["item_id"] for e in finals] == ["old", "new"]
        assert finals[0]["superseded"]
        assert not finals[1]["superseded"]
        assert finals[1]["agent_text"] == "178 buyurtmam. Bekor qilmang."
    asyncio.run(scenario())


class FakeRecognizer:
    def __init__(self, config, emit):
        self.config, self.emit = config, emit
        self.audio = []
        self.closed = False
    async def start(self):
        return self.config
    async def feed(self, packet):
        self.audio.append(packet)
    async def flush(self):
        await self.emit({"type": "conversation.item.input_audio_transcription.completed", "item_id": "last", "transcript": "178"})
    async def close(self):
        self.closed = True


def client_and_engine(monkeypatch):
    engines = []
    def create(config, emit):
        engine = FakeRecognizer(config, emit)
        engines.append(engine)
        return engine
    monkeypatch.setattr(live_stt, "create_transcriber", create)
    monkeypatch.setenv("LIVE_STT_PROVIDER", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = FastAPI()
    app.include_router(live_stt.router)
    return TestClient(app), engines


def test_websocket_stop_flushes_tail_before_ack_without_api_key(monkeypatch):
    client, engines = client_and_engine(monkeypatch)
    with client.websocket_connect("/ws/live-stt?language=en&agent=false") as ws:
        assert ws.receive_json()["type"] == "loading"
        assert ws.receive_json()["sample_rate"] == 16000
        ws.send_bytes(b"\x01\x00" * 640)
        # Old browser echoes cannot inject or duplicate agent turns.
        ws.send_json({"type": "agent_utterance", "item_id": "fake", "text": "invented"})
        ws.send_json({"type": "stop"})
        final = ws.receive_json()
        assert final["transcript"] == "178"
        assert ws.receive_json()["type"] == "stopped"
    assert len(engines[0].audio) == 1
    assert engines[0].config["language"] == "en"
    assert engines[0].closed
    assert live_stt._active_sessions == 0


@pytest.mark.parametrize("packet", [b"x", b"\x00" * 32002])
def test_malformed_pcm_errors_instead_of_silently_losing_audio(monkeypatch, packet):
    client, engines = client_and_engine(monkeypatch)
    with client.websocket_connect("/ws/live-stt") as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_bytes(packet)
        assert ws.receive_json()["type"] == "error"
    assert not engines[0].audio
    assert engines[0].closed


def test_cross_origin_socket_is_rejected_before_model_load(monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    client, engines = client_and_engine(monkeypatch)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live-stt", headers={"origin": "https://elsewhere.invalid"}):
            pass
    assert not engines


def test_stop_records_transcript_without_starting_a_new_agent_reply():
    async def scenario():
        events = []
        class Agent:
            async def submit(self, *args):
                raise AssertionError("Agent must not restart while stopping")
        async def emit(event):
            events.append(event)
        relay = TranscriptRelay(emit, Agent())
        relay.stopping = True
        await relay({"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "ha"})
        assert events[0]["transcript"] == "ha"
    asyncio.run(scenario())


def test_cold_start_disconnect_cancels_waiter_and_releases_admission(monkeypatch):
    cancelled = []
    class SlowRecognizer(FakeRecognizer):
        async def start(self):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
    monkeypatch.setattr(live_stt, "create_transcriber", SlowRecognizer)
    app = FastAPI()
    app.include_router(live_stt.router)
    with TestClient(app).websocket_connect("/ws/live-stt?agent=false") as ws:
        assert ws.receive_json()["type"] == "loading"
    assert cancelled == [True]
    assert live_stt._active_sessions == 0


def test_session_capacity_is_explicit_and_recovers_after_stop(monkeypatch):
    client, engines = client_and_engine(monkeypatch)
    monkeypatch.setenv("LIVE_STT_MAX_SESSIONS", "1")
    with client.websocket_connect("/ws/live-stt") as first:
        first.receive_json()
        first.receive_json()
        with client.websocket_connect("/ws/live-stt") as second:
            assert second.receive_json()["type"] == "error"
        assert len(engines) == 1
        first.send_json({"type": "stop"})
        first.receive_json()
        assert first.receive_json()["type"] == "stopped"
    assert live_stt._active_sessions == 0
