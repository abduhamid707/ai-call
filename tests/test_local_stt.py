"""Streaming lifecycle tests: no model downloads, microphones, or paid APIs."""
import asyncio
import json
import threading
import unittest
import wave
from dataclasses import replace
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.local_stt import (
    FRAME_SAMPLES, LocalSTTConfig, LocalSTTError, LocalSTTOverloaded,
    LocalSTTSession, StreamingSileroVAD, Transcript, Word, _remove_overlap, _run_native, main,
)


def pcm(frames, amplitude=0.2, tail=0):
    return np.full(frames * FRAME_SAMPLES + tail, int(amplitude * 32767), dtype="<i2").tobytes()


def detector(audio):
    # Deterministic substitute for the separately integrated Silero ONNX model.
    return 0.95 if float(np.max(np.abs(audio))) > 0.01 else 0.01


async def send(session, data, packet_bytes=8192):
    for start in range(0, len(data), packet_bytes):
        await session.feed(data[start:start + packet_bytes])


class LocalSTTTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.sessions = []

    async def asyncTearDown(self):
        for session in self.sessions:
            await session.close()

    async def session(self, transcriber=None, config=None, language="uz"):
        async def emit(event):
            self.events.append(event)

        session = LocalSTTSession(
            language, emit, detector=detector,
            transcriber=transcriber or (lambda *_: Transcript("buyurtma kerak", "uz", -0.2, 0.02)),
            config=config or LocalSTTConfig(),
        )
        self.sessions.append(session)
        self.assertEqual((await session.start())["sample_rate"], 16000)
        return session

    def completed(self):
        return [event for event in self.events if event["type"].endswith(".completed")]

    async def wait_event(self, kind, count=1):
        async def wait():
            while len([event for event in self.events if event["type"] == kind]) < count:
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), timeout=2)

    async def test_silence_and_short_click_do_not_create_transcripts(self):
        calls = []
        session = await self.session(lambda *args: calls.append(args))
        await send(session, pcm(30, 0) + pcm(2) + pcm(30, 0))
        await session.flush()
        self.assertEqual(self.events, [])
        self.assertEqual(calls, [])

    async def test_arbitrary_packet_boundaries_preserve_onset_and_final_once(self):
        seen = []

        def transcribe(audio, language, final):
            seen.append(audio)
            return Transcript("buyurtma", language, -0.2, 0.01)

        session = await self.session(transcribe)
        await send(session, pcm(4, 0) + pcm(10) + pcm(20, 0), packet_bytes=634)
        await session.flush()
        await session.flush()
        final = self.completed()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["transcript"], "buyurtma")
        self.assertTrue(final[0]["end_of_turn"])
        self.assertEqual(final[0]["reason"], "silence")
        self.assertEqual(int(np.count_nonzero(seen[-1])), 10 * FRAME_SAMPLES)
        self.assertTrue(np.all(seen[-1][:4 * FRAME_SAMPLES] == 0))
        self.assertEqual(len([event for event in self.events if event["type"].endswith("speech_started")]), 1)

    async def test_flush_keeps_non_frame_aligned_tail(self):
        audio_seen = []

        def transcribe(audio, *_):
            audio_seen.append(audio)
            return Transcript("ha", "uz", -0.2, 0.01)

        session = await self.session(transcribe)
        await send(session, pcm(6, tail=103), packet_bytes=702)
        await session.flush()
        self.assertEqual(len(audio_seen[-1]), 6 * FRAME_SAMPLES + 103)
        self.assertGreater(float(audio_seen[-1][-1]), 0)
        self.assertEqual(self.completed()[0]["transcript"], "ha")
        self.assertFalse(self.completed()[0]["needs_confirmation"])
        self.assertEqual(self.completed()[0]["reason"], "flush")

    async def test_partial_revisions_replace_text_and_final_is_authoritative(self):
        replies = iter(["buyurt", "buyurtma kerak", "buyurtma kerak emas"])
        session = await self.session(
            lambda *_: Transcript(next(replies), "uz", -0.2, 0.01),
            replace(LocalSTTConfig(), partial_interval_ms=320),
        )
        await send(session, pcm(10))
        await self.wait_event("transcript.partial")
        await send(session, pcm(10))
        await self.wait_event("transcript.partial", 2)
        await session.flush()
        partials = [event for event in self.events if event["type"] == "transcript.partial"]
        self.assertEqual([event["transcript"] for event in partials], ["buyurt", "buyurtma kerak"])
        self.assertNotIn("delta", partials[0])
        self.assertEqual(self.completed()[0]["transcript"], "buyurtma kerak emas")

    async def test_long_turn_splits_without_starting_a_second_turn(self):
        replies = iter([
            Transcript("men buyurtma", "uz", -0.2, 0.01),
            Transcript("buyurtma bermoqchiman", "uz", -0.2, 0.01,
                       (Word("buyurtma", 0, 0.4), Word(" bermoqchiman", 0.55, 0.8))),
        ])
        session = await self.session(
            lambda *_: next(replies),
            replace(LocalSTTConfig(), max_segment_seconds=3, partial_interval_ms=5000),
        )
        await send(session, pcm(96) + pcm(20, 0))
        await session.flush()
        finals = self.completed()
        self.assertEqual(len(finals), 2)
        self.assertEqual(finals[0]["turn_id"], finals[1]["turn_id"])
        self.assertNotEqual(finals[0]["item_id"], finals[1]["item_id"])
        self.assertFalse(finals[0]["end_of_turn"])
        self.assertTrue(finals[1]["end_of_turn"])
        self.assertEqual(finals[1]["transcript"], "bermoqchiman")
        self.assertEqual(finals[1]["raw_transcript"], "buyurtma bermoqchiman")

    async def test_empty_continuation_still_ends_segmented_turn(self):
        session = await self.session(
            lambda *_: Transcript("buyurtma", "uz", -0.2, 0.01, (Word("buyurtma", 0, 0.4),)),
            replace(LocalSTTConfig(), max_segment_seconds=3, partial_interval_ms=5000),
        )
        await send(session, pcm(94))
        await session.flush()
        finals = self.completed()
        self.assertEqual(len(finals), 2)
        self.assertEqual(finals[1]["transcript"], "")
        self.assertTrue(finals[1]["end_of_turn"])

    async def test_bad_packets_and_unknown_language_fail_explicitly(self):
        session = await self.session()
        for data in (b"\0", b"\0" * 32002):
            with self.assertRaises(LocalSTTError):
                await session.feed(data)
        with self.assertRaises(LocalSTTError):
            LocalSTTSession("xx", lambda _: None)

    async def test_uncertainty_preserves_raw_text_and_negation(self):
        session = await self.session(lambda *_: Transcript("buyurtma kerak emas", "uz", -1.2, 0.1))
        await send(session, pcm(6))
        await session.flush()
        event = self.completed()[0]
        self.assertEqual(event["raw_transcript"], "buyurtma kerak emas")
        self.assertEqual(event["transcript"], event["raw_transcript"])
        self.assertTrue(event["needs_confirmation"])
        self.assertGreaterEqual(event["metrics"]["inference_ms"], 0)

    async def test_numpy_confidence_types_are_json_serializable(self):
        session = await self.session(
            lambda *_: Transcript("buyurtma", "uz", np.float32(-0.2), np.float32(0.01))
        )
        await send(session, pcm(6))
        await session.flush()
        event = json.loads(json.dumps(self.completed()[0]))
        self.assertIs(event["needs_confirmation"], False)
        self.assertIsInstance(event["avg_logprob"], float)

    async def test_overload_rejects_instead_of_dropping_a_final_turn(self):
        session = await self.session(config=replace(LocalSTTConfig(), max_pending_segments=1))
        # feed/emit do not yield to the decoder in this deterministic fixture.
        await send(session, pcm(6) + pcm(20, 0))
        with self.assertRaises(LocalSTTOverloaded):
            await send(session, pcm(6) + pcm(20, 0))
        self.assertEqual(session._final_jobs.qsize(), 1)

    async def test_decoder_failure_does_not_hang_flush(self):
        def fail(*_):
            raise RuntimeError("model failed")

        session = await self.session(fail)
        await send(session, pcm(6))
        with self.assertRaises(LocalSTTError):
            await asyncio.wait_for(session.flush(), timeout=2)
        self.assertTrue(any(event.get("code") == "local_stt_failed" for event in self.events))

    async def test_endpoint_supersedes_a_partial_already_running(self):
        entered, release = threading.Event(), threading.Event()

        def decode(audio, language, final):
            if not final:
                entered.set()
                release.wait(3)
            return Transcript("buyurtma kerak" if final else "xato qoralama", "uz", -0.2, 0.01)

        session = await self.session(decode, replace(LocalSTTConfig(), partial_interval_ms=320))
        await send(session, pcm(10))

        async def wait_started():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        try:
            await asyncio.wait_for(wait_started(), timeout=1)
            await send(session, pcm(20, 0))
            release.set()
            await asyncio.wait_for(session.flush(), timeout=2)
            self.assertFalse(any(event["type"] == "transcript.partial" for event in self.events))
            self.assertEqual(self.completed()[0]["transcript"], "buyurtma kerak")
        finally:
            release.set()

    async def test_cancellation_keeps_native_slot_until_inference_finishes(self):
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()

        def first():
            entered.set()
            release.wait(3)

        first_task = asyncio.create_task(_run_native(first))
        second_task = None
        try:
            async def wait_started():
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(wait_started(), timeout=1)
            first_task.cancel()
            await asyncio.gather(first_task, return_exceptions=True)
            second_task = asyncio.create_task(_run_native(second_entered.set))
            await asyncio.sleep(0.03)
            self.assertFalse(second_entered.is_set())
            release.set()
            await asyncio.wait_for(second_task, timeout=2)
            self.assertTrue(second_entered.is_set())
        finally:
            release.set()
            if second_task:
                await asyncio.gather(second_task, return_exceptions=True)


class LocalConfigTests(unittest.TestCase):
    def test_cli_never_loads_a_model_without_explicit_warmup(self):
        with patch("app.local_stt._warmup") as warmup, redirect_stdout(StringIO()):
            self.assertEqual(main([]), 2)
            warmup.assert_not_called()

    def test_cli_warmup_reports_hardware_and_validates_language(self):
        model = SimpleNamespace(
            model=SimpleNamespace(device="cpu", compute_type="int8"),
            supported_languages=["uz", "ru", "en"],
        )
        output = StringIO()
        with patch("app.local_stt._warmup") as warmup, patch("app.local_stt._get_model", return_value=model), redirect_stdout(output):
            self.assertEqual(main(["--warmup", "--language", "uz"]), 0)
            self.assertEqual(warmup.call_args.args[1], "uz")
        self.assertIn('"device": "cpu"', output.getvalue())
        self.assertIn('"ready": true', output.getvalue())

    def test_nonfinite_configuration_is_rejected(self):
        with patch.dict("os.environ", {"LIVE_STT_VAD_THRESHOLD": "nan"}):
            with self.assertRaises(LocalSTTError):
                LocalSTTConfig.from_env()

    def test_overlap_preserves_real_repetition_after_context_window(self):
        result = Transcript("12 12", words=(Word("12", 0, 0.3), Word(" 12", 0.6, 0.8)))
        self.assertEqual(_remove_overlap("raqam 12", result), "12")
        # The same words without timing evidence cannot be safely deleted.
        self.assertEqual(_remove_overlap("raqam 12", Transcript("12 12")), "12 12")
        result = Transcript("emas", words=(Word("emas", 0.6, 0.8),))
        self.assertEqual(_remove_overlap("kerak emas", result), "emas")

    def test_bundled_silero_silence_and_state_are_per_session(self):
        try:
            from faster_whisper.vad import get_vad_model
        except ImportError:
            self.skipTest("Install project dependencies for the bundled Silero integration test")
        model = get_vad_model()
        left, right = StreamingSileroVAD(model), StreamingSileroVAD(model)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for _ in range(10):
            self.assertLess(left(silence), 0.5)
        self.assertFalse(np.shares_memory(left._state, right._state))
        self.assertTrue(np.all(right._state == 0))
        self.assertLess(right(silence), 0.5)

    def test_faster_whisper_decodes_and_resamples_wav_with_supported_pyav(self):
        try:
            from faster_whisper.audio import decode_audio
        except ImportError:
            self.skipTest("Install project dependencies for the audio decoder integration test")
        encoded = BytesIO()
        with wave.open(encoded, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(44100)
            output.writeframes(np.full((4410, 2), 2000, dtype="<i2").tobytes())
        encoded.seek(0)
        audio = decode_audio(encoded, sampling_rate=16000)
        self.assertEqual(audio.shape, (1600,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(audio)), 2000 / 32768, places=4)


if __name__ == "__main__":
    unittest.main()
