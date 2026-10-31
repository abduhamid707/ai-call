import asyncio
import json

import httpx
import pytest

from app.live_agent import LiveAgent


class SpeechReply(httpx.AsyncByteStream):
    def __init__(self, text, hold=None):
        self.text, self.hold = text, hold
        self.closed = asyncio.Event()

    async def __aiter__(self):
        payload = {"choices": [{"delta": {"content": self.text}}]}
        yield ("data: " + json.dumps(payload) + "\n\n").encode()
        if self.hold:
            await self.hold.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed.set()


async def mock_http(agent, handler):
    await agent.http.aclose()
    agent.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_real_reply_barge_in_cancels_http_stream_and_nested_tts(monkeypatch):
    async def scenario():
        events = []
        tts_started, tts_cancelled = asyncio.Event(), asyncio.Event()
        sse = SpeechReply("Buyurtmangiz raqamini aytib bera olasizmi?", hold=asyncio.Event())

        async def emit(event):
            events.append(event)

        async def synthesis(*args, **kwargs):
            tts_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                tts_cancelled.set()

        def response(request):
            assert request.url.path == "/v1/chat/completions"
            assert json.loads(request.content)["messages"][-1]["content"] == "buyurtma"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=sse)

        monkeypatch.setattr("app.live_agent.get_active_openai_key", lambda: "test-not-a-real-key")
        monkeypatch.setattr("app.live_agent.generate_edge_tts_audio", synthesis)
        monkeypatch.setenv("LIVE_AGENT_TTS_PROVIDER", "edge_tts")
        agent = LiveAgent(emit)
        await mock_http(agent, response)
        try:
            await agent.submit("old", "buyurtma")
            await asyncio.wait_for(tts_started.wait(), 2)
            await asyncio.wait_for(agent.interrupt(), 2)
            assert tts_cancelled.is_set(), "_reply must await cancellation of its real _synthesize child"
            assert sse.closed.is_set(), "cancelled LLM stream must release its connection"
            assert agent.task is None and agent.active_turn is None
            assert agent.pending_answer is None
            assert events[-1] == {"type": "agent_cancelled", "turn_id": "old"}
            assert not any(event["type"] in {"agent_audio", "agent_done"} for event in events)
            assert agent.history == [{"role": "user", "content": "buyurtma"}]
        finally:
            await agent.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("selection,detected,voice", [
    ("auto", "ru", "ru-RU-DmitryNeural"),
    ("auto", "en", "en-US-AndrewMultilingualNeural"),
    ("uz", "ru", "uz-UZ-SardorNeural"),
])
def test_reply_voice_follows_detected_language_only_in_auto_mode(monkeypatch, selection, detected, voice):
    async def scenario():
        voices = []
        async def emit(event):
            pass
        async def synthesis(*args, **kwargs):
            voices.append(kwargs["voice"])
            return b"audio"
        def response(request):
            return httpx.Response(200, stream=SpeechReply("This is a complete answer."))
        monkeypatch.setattr("app.live_agent.get_active_openai_key", lambda: "test-not-a-real-key")
        monkeypatch.setattr("app.live_agent.generate_edge_tts_audio", synthesis)
        monkeypatch.delenv("LIVE_AGENT_EDGE_VOICE", raising=False)
        monkeypatch.setenv("LIVE_AGENT_TTS_PROVIDER", "edge_tts")
        agent = LiveAgent(emit, selection)
        await mock_http(agent, response)
        try:
            await agent.submit("turn", "question", language=detected)
            await asyncio.wait_for(agent.task, 2)
            assert voices == [voice]
        finally:
            await agent.close()
    asyncio.run(scenario())


def test_concurrent_interrupt_and_submit_do_not_cancel_the_new_turn(monkeypatch):
    async def scenario():
        events = []
        cancelling_old, release_old = asyncio.Event(), asyncio.Event()

        async def emit(event):
            events.append(event)

        agent = LiveAgent(emit)

        async def delayed_cleanup(turn, *_):
            try:
                await asyncio.Event().wait()
            finally:
                if turn == "old":
                    cancelling_old.set()
                    await release_old.wait()

        monkeypatch.setattr(agent, "_reply", delayed_cleanup)
        try:
            await agent.submit("old", "birinchi")
            await asyncio.sleep(0)
            interrupted = asyncio.create_task(agent.interrupt())
            await asyncio.wait_for(cancelling_old.wait(), 2)
            submitted = asyncio.create_task(agent.submit("new", "ikkinchi"))
            await asyncio.sleep(0)
            assert not submitted.done(), "new submit must wait for old cleanup under the control lock"
            release_old.set()
            await asyncio.wait_for(asyncio.gather(interrupted, submitted), 2)
            assert agent.active_turn == "new"
            assert agent.task is not None and not agent.task.done()
            assert events == [{"type": "agent_cancelled", "turn_id": "old"}]
        finally:
            release_old.set()
            await agent.close()

    asyncio.run(scenario())


def test_real_reply_only_acknowledged_playback_enters_assistant_history(monkeypatch):
    async def scenario():
        events = []

        async def emit(event):
            events.append(event)

        async def synthesis(*args, **kwargs):
            return b"mock-mp3-no-network"

        def response(request):
            question = json.loads(request.content)["messages"][-1]["content"]
            text = "Birinchi javob to'liq eshitildi." if question == "birinchi" else "Ikkinchi javob hali eshitilmadi."
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SpeechReply(text))

        monkeypatch.setattr("app.live_agent.get_active_openai_key", lambda: "test-not-a-real-key")
        monkeypatch.setattr("app.live_agent.generate_edge_tts_audio", synthesis)
        monkeypatch.setenv("LIVE_AGENT_TTS_PROVIDER", "edge_tts")
        agent = LiveAgent(emit)
        await mock_http(agent, response)
        try:
            await agent.submit("one", "birinchi")
            await asyncio.wait_for(agent.task, 2)
            assert events[-1]["type"] == "agent_done"
            assert not any(item["role"] == "assistant" for item in agent.history)
            await agent.acknowledge("wrong-turn")
            assert not any(item["role"] == "assistant" for item in agent.history)
            await agent.acknowledge("one")
            await agent.acknowledge("one")
            heard = [item for item in agent.history if item["role"] == "assistant"]
            assert heard == [{"role": "assistant", "content": "Birinchi javob to'liq eshitildi."}]
            await agent.submit("two", "ikkinchi")
            await asyncio.wait_for(agent.task, 2)
            assert agent.pending_answer[0] == "two"
            await agent.interrupt()
            await agent.acknowledge("two")
            assert [item for item in agent.history if item["role"] == "assistant"] == heard
            assert agent.pending_answer is None
            assert not any(event["type"] == "agent_error" for event in events)
        finally:
            await agent.close()

    asyncio.run(scenario())
