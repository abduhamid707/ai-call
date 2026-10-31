import asyncio
import json

from app.openai_live_stt import OpenAITranscriptionSession


class FakeSocket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.ack = True
    def __aiter__(self):
        return self
    async def __anext__(self):
        return json.dumps(await self.events.get())
    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "session.update" and self.ack:
            await self.events.put({"type": "session.updated"})
        if event["type"] == "input_audio_buffer.commit":
            await self.events.put({"type": "input_audio_buffer.committed", "item_id": "a"})
            await self.events.put({"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "178"})
    async def close(self):
        self.closed = True


def test_hosted_ready_waits_for_ack_and_stop_commits_active_speech(monkeypatch):
    async def scenario():
        socket = FakeSocket()
        socket.ack = False
        async def connect(*args, **kwargs):
            return socket
        monkeypatch.setattr("app.openai_live_stt.websockets.connect", connect)
        monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
        events = []
        async def emit(event):
            events.append(event)
        session = OpenAITranscriptionSession("ru", 600, emit)
        start = asyncio.create_task(session.start())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not start.done()
        socket.ack = True
        await socket.events.put({"type": "session.updated"})
        assert (await start)["sample_rate"] == 24000
        await session.feed(b"\x00\x01" * 2400)
        await socket.events.put({"type": "input_audio_buffer.speech_started", "item_id": "a"})
        for _ in range(5):
            await asyncio.sleep(0)
        await session.flush()
        assert events[-1]["transcript"] == "178"
        assert sum(e["type"] == "input_audio_buffer.commit" for e in socket.sent) == 1
        assert not session.pending
        await session.close()
        assert socket.closed
    asyncio.run(scenario())


def test_hosted_silence_only_stop_does_not_commit_empty_buffer(monkeypatch):
    async def scenario():
        socket = FakeSocket()
        async def connect(*args, **kwargs):
            return socket
        monkeypatch.setattr("app.openai_live_stt.websockets.connect", connect)
        monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
        async def emit(event):
            pass
        session = OpenAITranscriptionSession("en", 600, emit)
        await session.start()
        await session.feed(b"\x00" * 4800)
        await session.flush()
        assert all(e["type"] != "input_audio_buffer.commit" for e in socket.sent)
        await session.close()
    asyncio.run(scenario())
