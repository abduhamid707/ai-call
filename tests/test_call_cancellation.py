import asyncio
import sys
from types import ModuleType, SimpleNamespace

import aiohttp
import pytest

from app.elevenlabs_call import PersistentEdgeTTS, _until_client_disconnect


def test_call_disconnect_cancels_and_awaits_downstream_work():
    async def scenario():
        messages = asyncio.Queue()
        started, cancelled = asyncio.Event(), asyncio.Event()
        request = SimpleNamespace(receive=messages.get)

        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        processing = asyncio.create_task(_until_client_disconnect(request, work()))
        await asyncio.wait_for(started.wait(), 2)
        await messages.put({"type": "http.disconnect"})
        response = await asyncio.wait_for(processing, 2)
        assert response.status_code == 499
        assert cancelled.is_set(), "aborted client must not leave paid LLM/TTS work running"

    asyncio.run(scenario())


def test_completed_call_cancels_the_disconnect_watcher():
    async def scenario():
        watcher_closed = asyncio.Event()

        async def receive():
            try:
                await asyncio.Event().wait()
            finally:
                watcher_closed.set()

        async def work():
            await asyncio.sleep(0)
            return {"reply": "ha"}

        result = await _until_client_disconnect(SimpleNamespace(receive=receive), work())
        assert result == {"reply": "ha"}
        assert watcher_closed.is_set()

    asyncio.run(scenario())


def test_cancelled_pooled_tts_detaches_socket_before_close_and_next_reply_is_fresh(monkeypatch):
    # Only import-time formatting helpers are stubbed; production synthesize,
    # cancellation handling, pool lock and binary frame parsing all execute.
    package = ModuleType("edge_tts")
    package.__path__ = []
    communicate = ModuleType("edge_tts.communicate")
    communicate.connect_id = lambda: "test-request"
    communicate.date_to_string = lambda: "test-date"
    communicate.ssml_headers_plus_data = lambda *_: "test-ssml"
    communicate.mkssml = lambda *_: "test-ssml-body"
    communicate.get_headers_and_data = lambda data, _: ({b"Path": b"audio"}, data[2:])
    communicate.remove_incompatible_characters = lambda text: text
    classes = ModuleType("edge_tts.data_classes")
    classes.TTSConfig = lambda *_: SimpleNamespace()
    for name, module in (("edge_tts", package), ("edge_tts.communicate", communicate), ("edge_tts.data_classes", classes)):
        monkeypatch.setitem(sys.modules, name, module)

    async def scenario():
        pool = PersistentEdgeTTS()

        class Socket:
            def __init__(self):
                self.messages = asyncio.Queue()
                self.sent = asyncio.Event()
                self.closed = False

            async def send_str(self, _):
                self.sent.set()

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self.messages.get()

            async def close(self):
                assert pool._ws is not self, "detach before waiting for remote close"
                await asyncio.sleep(0)
                self.closed = True

        old, fresh = Socket(), Socket()
        pool._ws = old

        async def ensure_socket():
            if pool._ws is None:
                pool._ws = fresh
            return pool._ws

        monkeypatch.setattr(pool, "_ensure_ws", ensure_socket)
        pending = asyncio.create_task(pool.synthesize("old answer", "voice", "+0%"))
        await asyncio.wait_for(old.sent.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert old.closed
        assert pool._ws is None
        # A queued old response must never be consumed/cached as the new answer.
        await old.messages.put(SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"\x00\x00OLD"))
        await fresh.messages.put(SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"\x00\x00NEW"))
        await fresh.messages.put(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Path:turn.end"))
        audio = await asyncio.wait_for(pool.synthesize("new answer", "voice", "+0%"), 2)
        assert audio == b"NEW"
        assert pool._ws is fresh
        await pool._discard_socket()
        assert fresh.closed

    asyncio.run(scenario())
