"""Bounded, incremental microphone ASR using faster-whisper and Silero VAD.

Whisper is an offline decoder: partials below are revisable hypotheses over a
growing audio window, not native token streaming. Only endpointed transcripts
are final. No model is imported/downloaded until ``start`` is called.
"""
from __future__ import annotations

import asyncio
import functools
import math
import os
import threading
import time
import uuid
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .speech_cleanup import clean_transcript

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero v5's 32 ms frame, at 16 kHz.
MAX_PACKET_BYTES = SAMPLE_RATE * 2


class LocalSTTError(RuntimeError):
    """Local ASR could not safely process the stream."""


class LocalSTTOverloaded(LocalSTTError):
    """The decoder cannot keep up; never silently discard audio/final turns."""


def _number(name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise LocalSTTError(f"Invalid {name}") from exc
    if not math.isfinite(value) or not lower <= value <= upper:
        raise LocalSTTError(f"{name} must be between {lower} and {upper}")
    return value


@dataclass(frozen=True)
class LocalSTTConfig:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 4
    beam_size: int = 5
    partial_interval_ms: int = 800
    silence_ms: int = 600
    min_speech_ms: int = 128
    prefix_ms: int = 320
    max_segment_seconds: float = 15
    max_turn_seconds: float = 120
    max_pending_segments: int = 4
    vad_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "LocalSTTConfig":
        config = cls(
            model=os.getenv("LIVE_STT_LOCAL_MODEL", "large-v3").strip(),
            device=os.getenv("LIVE_STT_LOCAL_DEVICE", "auto").strip().lower(),
            compute_type=os.getenv("LIVE_STT_LOCAL_COMPUTE_TYPE", "auto").strip(),
            cpu_threads=int(_number("LIVE_STT_LOCAL_CPU_THREADS", 4, 1, 64)),
            beam_size=int(_number("LIVE_STT_LOCAL_BEAM_SIZE", 5, 1, 10)),
            partial_interval_ms=int(_number("LIVE_STT_PARTIAL_INTERVAL_MS", 800, 320, 5000)),
            silence_ms=int(_number("LIVE_STT_SILENCE_MS", 600, 300, 2000)),
            min_speech_ms=int(_number("LIVE_STT_MIN_SPEECH_MS", 128, 64, 500)),
            prefix_ms=int(_number("LIVE_STT_PREFIX_MS", 320, 160, 1000)),
            max_segment_seconds=_number("LIVE_STT_MAX_SEGMENT_SECONDS", 15, 3, 25),
            max_turn_seconds=_number("LIVE_STT_MAX_TURN_SECONDS", 120, 15, 300),
            max_pending_segments=int(_number("LIVE_STT_MAX_PENDING_SEGMENTS", 4, 1, 8)),
            vad_threshold=_number("LIVE_STT_VAD_THRESHOLD", 0.5, 0.1, 0.95),
        )
        if not config.model or config.device not in {"auto", "cpu", "cuda"}:
            raise LocalSTTError("Set a local model and device auto, cpu, or cuda")
        return config


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = "uz"
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: tuple[Word, ...] = ()


# One process-wide native worker. Its slot stays occupied until native inference
# actually finishes, even if a disconnected WebSocket cancels its asyncio task.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-asr")
_LOOP_SLOTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_KEY = None


async def _run_native(function, *args, metrics=None):
    loop = asyncio.get_running_loop()
    queued_at = time.perf_counter()
    slot = _LOOP_SLOTS.setdefault(loop, asyncio.Semaphore(1))
    await slot.acquire()

    def execute():
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            if metrics is not None:
                metrics["native_wait_ms"] = round((started - queued_at) * 1000)
                metrics["inference_ms"] = round((time.perf_counter() - started) * 1000)

    try:
        future = loop.run_in_executor(_EXECUTOR, execute)
    except BaseException:
        slot.release()
        raise

    def finished(done):
        slot.release()
        # Retrieve failures even when the awaiting client has been cancelled.
        if not done.cancelled():
            done.exception()

    future.add_done_callback(finished)
    return await asyncio.shield(future)


def _get_model(config: LocalSTTConfig):
    global _MODEL, _MODEL_KEY
    key = (config.model, config.device, config.compute_type, config.cpu_threads)
    with _MODEL_LOCK:
        if _MODEL is not None:
            if key != _MODEL_KEY:
                raise LocalSTTError("Restart the server after changing local ASR model settings")
            return _MODEL
        import ctranslate2
        from faster_whisper import WhisperModel

        device = config.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        compute = config.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        # Fail explicitly on unavailable hardware. Never substitute base/small.
        model = WhisperModel(
            config.model, device=device, compute_type=compute,
            cpu_threads=config.cpu_threads, num_workers=1,
        )
        # Force the lazy generator to execute before declaring the socket ready.
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), language="en",
            beam_size=1, condition_on_previous_text=False, vad_filter=False,
        )
        list(segments)
        _MODEL, _MODEL_KEY = model, key
        return model


class StreamingSileroVAD:
    """Per-stream recurrent state for bundled faster-whisper 1.1.1 Silero v5.

    ONNX sessions contain the weights, while context/state belong exclusively to
    this microphone. Calling the offline VAD on each frame would reset state.
    """

    def __init__(self, model):
        self._model = model
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        window = np.concatenate((self._context, audio.reshape(1, -1)), axis=1)
        encoded = self._model.encoder_session.run(None, {"input": window})[0]
        probability, self._state = self._model.decoder_session.run(
            None, {"input": encoded.reshape(1, 128), "state": self._state}
        )
        self._context = audio[-64:].reshape(1, -1).copy()
        return float(np.asarray(probability).reshape(-1)[0])


def _warmup(config: LocalSTTConfig, language: str = "auto"):
    model = _get_model(config)
    if language != "auto" and language not in model.supported_languages:
        raise LocalSTTError(f"Configured ASR model does not support language {language}")
    from faster_whisper.vad import get_vad_model
    return StreamingSileroVAD(get_vad_model())


def vocabulary_hints(language: str) -> str:
    explicit = os.getenv("LIVE_STT_VOCABULARY_HINTS", os.getenv("STT_VOCABULARY_HINTS", ""))
    if explicit.strip():
        return explicit.strip()[:500]
    terms = {
        "uz": "buyurtma, buyurtmani, buyurtmam, yetkazib berish, to'lov, manzil, telefon raqami",
        "ru": "заказ, доставка, оплата, адрес, номер телефона",
        "en": "order, delivery, payment, address, phone number",
        "tr": "sipariş, teslimat, ödeme, adres, telefon numarası",
    }
    return terms.get(language, ", ".join(terms[key] for key in ("uz", "ru", "en")))


def _decode(audio, language: str, final: bool, config: LocalSTTConfig) -> Transcript:
    model = _get_model(config)
    segments, info = model.transcribe(
        audio, language=None if language == "auto" else language,
        task="transcribe", beam_size=config.beam_size if final else 1,
        best_of=1, temperature=0.0, condition_on_previous_text=False,
        hotwords=vocabulary_hints(language), vad_filter=False,
        no_speech_threshold=0.6, log_prob_threshold=-1.0, word_timestamps=True,
    )
    segments = list(segments)  # Native inference is lazy: consume in the worker.
    weights = [max(segment.end - segment.start, 0.01) for segment in segments]
    weight_sum = sum(weights)
    return Transcript(
        text=" ".join(segment.text.strip() for segment in segments).strip(),
        language=info.language,
        avg_logprob=(float(sum(s.avg_logprob * w for s, w in zip(segments, weights)) / weight_sum)
                     if weight_sum else None),
        no_speech_prob=(float(sum(s.no_speech_prob * w for s, w in zip(segments, weights)) / weight_sum)
                       if weight_sum else None),
        words=tuple(Word(word.word, float(word.start), float(word.end))
                    for segment in segments for word in (segment.words or [])),
    )


@dataclass(frozen=True)
class _Job:
    item_id: str
    turn_id: str
    segment_index: int
    audio: np.ndarray
    final: bool
    end_of_turn: bool
    reason: str
    queued_at: float


def _remove_overlap(previous: str, result: Transcript) -> str:
    """Trim a repeated word only when its timestamp lies in repeated audio.

    Text matching alone can erase a genuine repeated number or negation after a
    chunk boundary. Missing/ambiguous timestamps therefore preserve the words.
    """
    before = previous.split()
    overlap_words = []
    for word in result.words:
        if word.end > 0.5 or len(overlap_words) >= 12:
            break
        overlap_words.append(word.text.strip())
    for size in range(min(len(before), len(overlap_words)), 0, -1):
        if [x.casefold().strip(",.!?;:") for x in before[-size:]] == [
            x.casefold().strip(",.!?;:") for x in overlap_words[:size]
        ]:
            return "".join(word.text for word in result.words[size:]).strip()
    return result.text.strip()


class LocalSTTSession:
    """One microphone's PCM framing, VAD, queue, and partial/final lifecycle.

    ``detector`` and ``transcriber`` are test seams. A transcriber is a synchronous
    callable ``(float32_audio, language, final) -> Transcript`` and runs on the
    same bounded native worker as production inference.
    """

    def __init__(
        self, language: str, emit: Callable[[dict], Awaitable[None]], *,
        silence_ms: int | None = None, config: LocalSTTConfig | None = None,
        detector=None, transcriber=None,
    ):
        self.config = config or LocalSTTConfig.from_env()
        language = (language or "uz").lower().split("-")[0]
        if language not in {"uz", "ru", "en", "tr", "auto"}:
            raise LocalSTTError("Local ASR supports language uz, ru, en, tr, or auto")
        self.language = language
        self.emit = emit
        self.silence_ms = self.config.silence_ms if silence_ms is None else int(silence_ms)
        if not 300 <= self.silence_ms <= 2000:
            raise LocalSTTError("silence_ms must be between 300 and 2000")
        self._detector = detector
        self._transcriber = transcriber or functools.partial(_decode, config=self.config)
        self._needs_model = transcriber is None
        self._worker_task = None
        self._closed = False
        self._error = None
        self._wake = asyncio.Event()
        self._final_jobs = asyncio.Queue(maxsize=self.config.max_pending_segments)
        self._partial_job = None
        self._finalizing = set()
        self._prior_text = {}
        self._prior_partial = {}
        self._pcm_pending = b""
        self._pre_roll = deque(maxlen=math.ceil(
            (self.config.prefix_ms + self.config.min_speech_ms) * SAMPLE_RATE / 1000 / FRAME_SAMPLES
        ))
        self._active = False
        self._speech_run = 0
        self._silence_samples = 0
        self._audio = []
        self._audio_samples = 0
        self._turn_samples = 0
        self._last_partial_samples = 0
        self._stream_samples = 0
        self._turn_id = ""
        self._item_id = ""
        self._segment_index = 0

    async def start(self) -> dict:
        if self._closed:
            raise LocalSTTError("The local ASR session is closed")
        if self._worker_task is not None:
            raise LocalSTTError("The local ASR session has already started")
        if self._needs_model or self._detector is None:
            vad = await _run_native(_warmup, self.config, self.language)
            self._detector = self._detector or vad
        self._worker_task = asyncio.create_task(self._worker(), name="local-stt-decoder")
        return {"sample_rate": SAMPLE_RATE, "provider": "local", "model": self.config.model,
                "language": self.language, "silence_ms": self.silence_ms}

    def _check(self):
        if self._error:
            raise LocalSTTError("Local transcription failed; reconnect after checking the server") from self._error
        if self._closed or self._worker_task is None:
            raise LocalSTTError("The local ASR session is not running")

    async def feed(self, packet: bytes):
        self._check()
        if len(packet) % 2 or len(packet) > MAX_PACKET_BYTES:
            raise LocalSTTError("Expected at most one second of 16 kHz mono PCM16 audio per packet")
        self._pcm_pending += packet
        frame_bytes = FRAME_SAMPLES * 2
        while len(self._pcm_pending) >= frame_bytes:
            frame, self._pcm_pending = self._pcm_pending[:frame_bytes], self._pcm_pending[frame_bytes:]
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            await self._process_frame(samples)

    async def _process_frame(self, samples):
        count = len(samples)
        padded = np.pad(samples, (0, FRAME_SAMPLES - count)) if count < FRAME_SAMPLES else samples
        probability = self._detector(padded)
        self._stream_samples += count
        if not self._active:
            self._pre_roll.append(samples)
            self._speech_run = self._speech_run + count if probability >= self.config.vad_threshold else 0
            if self._speech_run < self.config.min_speech_ms * SAMPLE_RATE / 1000:
                return
            self._active = True
            self._turn_id = uuid.uuid4().hex
            self._segment_index = 0
            self._item_id = f"{self._turn_id}:0"
            self._audio = list(self._pre_roll)
            self._audio_samples = sum(len(x) for x in self._audio)
            self._turn_samples = self._audio_samples
            self._last_partial_samples = 0
            self._silence_samples = 0
            await self.emit({"type": "input_audio_buffer.speech_started", "item_id": self._item_id,
                             "turn_id": self._turn_id,
                             "audio_start_ms": round((self._stream_samples - self._audio_samples) * 1000 / SAMPLE_RATE)})
            return

        self._audio.append(samples)
        self._audio_samples += count
        self._turn_samples += count
        negative_threshold = max(0.01, self.config.vad_threshold - 0.15)
        self._silence_samples = self._silence_samples + count if probability < negative_threshold else 0
        if self._turn_samples > self.config.max_turn_seconds * SAMPLE_RATE:
            raise LocalSTTOverloaded("One uninterrupted turn exceeded the configured duration limit")
        if self._silence_samples >= self.silence_ms * SAMPLE_RATE / 1000:
            await self._finish("silence")
        elif self._audio_samples >= self.config.max_segment_seconds * SAMPLE_RATE:
            self._enqueue_final(end_of_turn=False, reason="segment_limit")
            # Re-decode 500 ms at the boundary so a word straddling it has context.
            overlap = np.concatenate(self._audio)[-SAMPLE_RATE // 2:].copy()
            self._audio = [overlap]
            self._audio_samples = len(overlap)
            self._last_partial_samples = self._audio_samples
            self._segment_index += 1
            self._item_id = f"{self._turn_id}:{self._segment_index}"
        elif self._audio_samples - self._last_partial_samples >= self.config.partial_interval_ms * SAMPLE_RATE / 1000:
            self._last_partial_samples = self._audio_samples
            # A single latest hypothesis replaces stale provisional work. Audio
            # and all final jobs are retained; only obsolete partial jobs coalesce.
            self._partial_job = self._job(final=False, end_of_turn=False, reason="partial")
            self._wake.set()

    def _job(self, *, final, end_of_turn, reason):
        return _Job(self._item_id, self._turn_id, self._segment_index,
                    np.concatenate(self._audio).copy(), final, end_of_turn, reason, time.perf_counter())

    def _enqueue_final(self, *, end_of_turn, reason):
        if self._final_jobs.full():
            raise LocalSTTOverloaded("Local ASR is slower than incoming speech; use a faster device/model")
        job = self._job(final=True, end_of_turn=end_of_turn, reason=reason)
        self._finalizing.add(job.item_id)
        self._final_jobs.put_nowait(job)
        if self._partial_job and self._partial_job.item_id == job.item_id:
            self._partial_job = None
        self._wake.set()

    async def _finish(self, reason):
        self._enqueue_final(end_of_turn=True, reason=reason)
        self._active = False
        self._pre_roll.clear()
        self._speech_run = 0
        self._audio = []
        self._audio_samples = 0
        await self.emit({"type": "input_audio_buffer.speech_stopped", "item_id": self._item_id,
                         "turn_id": self._turn_id,
                         "audio_end_ms": round(self._stream_samples * 1000 / SAMPLE_RATE)})

    async def flush(self):
        """Decode the final tail immediately, even without additional silence."""
        self._check()
        if self._pcm_pending:
            samples = np.frombuffer(self._pcm_pending, dtype="<i2").astype(np.float32) / 32768.0
            self._pcm_pending = b""
            await self._process_frame(samples)
        if self._active:
            await self._finish("flush")
        await self._final_jobs.join()
        self._check()

    async def _worker(self):
        try:
            while True:
                await self._wake.wait()
                job = None
                try:
                    job = self._final_jobs.get_nowait()
                except asyncio.QueueEmpty:
                    job, self._partial_job = self._partial_job, None
                if job is None:
                    self._wake.clear()
                    continue
                started = time.perf_counter()
                try:
                    native_metrics = {}
                    result = await _run_native(
                        self._transcriber, job.audio, self.language, job.final,
                        metrics=native_metrics,
                    )
                    # An endpoint supersedes any still-running provisional pass.
                    if not job.final and (job.item_id in self._finalizing or job.item_id != self._item_id):
                        continue
                    raw = result.text.strip()
                    text = clean_transcript(raw)
                    previous = self._prior_text.get(job.turn_id, "")
                    if job.segment_index:
                        text = clean_transcript(_remove_overlap(previous, result))
                    event = {
                        "type": ("conversation.item.input_audio_transcription.completed"
                                 if job.final else "transcript.partial"),
                        "item_id": job.item_id, "turn_id": job.turn_id,
                        "segment_index": job.segment_index, "end_of_turn": job.end_of_turn,
                        "transcript": text, "raw_transcript": raw,
                        "language": result.language,
                        "avg_logprob": float(result.avg_logprob) if result.avg_logprob is not None else None,
                        "no_speech_prob": float(result.no_speech_prob) if result.no_speech_prob is not None else None,
                        "needs_confirmation": bool(not raw or
                            (result.avg_logprob is not None and result.avg_logprob < -0.9) or
                            (result.no_speech_prob is not None and result.no_speech_prob > 0.55)),
                        "reason": job.reason,
                        "metrics": {
                            "audio_ms": round(len(job.audio) * 1000 / SAMPLE_RATE),
                            "queue_ms": round((started - job.queued_at) * 1000) + native_metrics["native_wait_ms"],
                            "inference_ms": native_metrics["inference_ms"],
                        },
                    }
                    if job.final:
                        self._prior_partial.pop(job.item_id, None)
                        self._finalizing.discard(job.item_id)
                        if job.end_of_turn:
                            self._prior_text.pop(job.turn_id, None)
                        else:
                            self._prior_text[job.turn_id] = (previous + " " + text).strip()
                    else:
                        before = self._prior_partial.get(job.item_id, "").split()
                        now = text.split()
                        stable = []
                        for left, right in zip(before, now):
                            if left != right:
                                break
                            stable.append(right)
                        event["stable_transcript"] = " ".join(stable)
                        self._prior_partial[job.item_id] = text
                    await self.emit(event)
                finally:
                    if job.final:
                        self._final_jobs.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc
            try:
                await self.emit({"type": "error", "code": "local_stt_failed",
                                 "message": "Mahalliy transkripsiyada xato. Server sozlamalarini tekshiring."})
            except Exception:
                pass
        finally:
            while not self._final_jobs.empty():
                self._final_jobs.get_nowait()
                self._final_jobs.task_done()

    async def close(self):
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        while not self._final_jobs.empty():
            self._final_jobs.get_nowait()
            self._final_jobs.task_done()
        self._partial_job = None
        self._audio.clear()
        self._pre_roll.clear()
        self._prior_text.clear()
        self._prior_partial.clear()
        self._finalizing.clear()
        self._pcm_pending = b""


def main(argv=None) -> int:
    """Download/warm the configured model before opening a browser session."""
    import argparse
    import json
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Prepare the local speech recognition model and VAD")
    parser.add_argument("--warmup", action="store_true", help="Load and validate the configured model")
    parser.add_argument(
        "--language", choices=("uz", "ru", "en", "tr", "auto"),
        default=os.getenv("LIVE_STT_LANGUAGE", "uz").lower().split("-")[0],
        help="Language to validate against the chosen model (default: LIVE_STT_LANGUAGE or uz)",
    )
    args = parser.parse_args(argv)
    if not args.warmup:
        parser.print_help()
        return 2
    if args.language not in {"uz", "ru", "en", "tr", "auto"}:
        parser.error("LIVE_STT_LANGUAGE must be uz, ru, en, tr, or auto")
    try:
        config = LocalSTTConfig.from_env()
        _warmup(config, args.language)
        model = _get_model(config)
        print(json.dumps({
            "ready": True,
            "model": config.model,
            "device": model.model.device,
            "compute_type": model.model.compute_type,
            "language": args.language,
            "sample_rate": SAMPLE_RATE,
            "vad": "silero-v5",
            "supported_call_languages": [code for code in ("uz", "ru", "en", "tr")
                                         if code in model.supported_languages],
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        detail = str(exc) if isinstance(exc, LocalSTTError) else type(exc).__name__
        print(f"Local ASR warmup failed: {detail}. Check model, hardware, and download/cache settings.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
