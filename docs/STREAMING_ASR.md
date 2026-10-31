# Jonli nutq: o‘zbekcha, ruscha va inglizcha

This integration adds a local streaming transcription path to `ai-call` using
the repository's existing `faster-whisper==1.1.1` dependency. The default model
is multilingual `large-v3`. A call uses a continuous microphone stream,
revisable partial transcripts and a finalized user turn before generating a
reply. Local transcription does not require an OpenAI API key; the existing
conversational LLM and speech synthesis providers have their own requirements.

**Delivery status:** these are implementation and evaluation instructions for
the feature branch. They do not assert that the production server has been
updated or that Uzbek accuracy/latency has been measured. Unit tests verify
software behavior; representative microphone recordings are needed to verify
recognition quality.

## What changes in the call

The microphone feeds one streaming recognition path. `/call` uses that path for
the user's conversation, while `/live-stt` can be used to inspect transcription
and exercise the agent separately. The local engine accepts mono 16 kHz PCM;
the optional hosted transcription path uses mono 24 kHz PCM. The browser must
resample from its actual audio-context rate, which can differ from a requested
microphone rate.

The recognition lifecycle is:

1. Capture speech continuously, including a short pre-roll before speech starts.
2. Detect speech using Silero VAD and accumulate acoustic context.
3. Produce partial text while the person is speaking. Partial text may change.
4. Finalize after the configured silence interval, retaining the whole user turn.
5. Pass the finalized turn to the conversational agent. If the person interrupts,
   cancel obsolete output rather than allowing it to speak over the new turn.

`buyurtma`, `buyurtmani`, `buyurtmam` and relevant business vocabulary can be
provided as decoder hints. They are not forced substitutions. A customer saying
“buyurtmani bekor **qilmang**” must not be changed to “buyurtmani bekor **qiling**”.
Raw recognition output remains available for evaluation; an LLM should not
invent a phone number, amount, product or order ID to make a transcript look neat.

Low-confidence recognition is flagged using average decoder log probability
and no-speech scores. These are **heuristics, not calibrated word-accuracy
probabilities**. They cannot guarantee that every wrong word, number or negation
will be detected. Clarification behavior and important values need their own
acceptance checks.

In the standalone live agent, an assistant answer is added to conversation
history only after the browser acknowledges that its audio finished playing.
An interrupted or failed playback is not recorded as a fully delivered answer.
Disconnect cancellation also stops obsolete call response work. These controls
support conversational consistency; they do not improve the acoustic model's
recognition accuracy.

## Run in the existing application

Use the project's Python 3.11+ installation instructions in [INSTALL.md](../INSTALL.md).
Keep the current chat/TTS configuration in `.env`. Add the streaming settings
from [.env.example](../.env.example), or start with that file only when no `.env`
already exists. Restart the server after changing the environment.

From the repository directory, after activating its virtual environment:

```bash
python -m pip install -r requirements.txt
python -m app.local_stt --warmup --language uz
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run the warmup command before opening a live session. It reads `.env`, downloads
the configured model if necessary, executes initialization, loads the bundled
Silero VAD and verifies that the selected model supports `uz`. Other choices are
`--language ru`, `--language en`, `--language tr` and `--language auto`.
It needs no microphone or API key.
An English-only model is rejected for an Uzbek session.

The command is a **separate process**: its downloaded weights remain in the
model cache, but its loaded model does not remain in the later server process.
The server still loads and initializes its own model before its first session
becomes ready. Warmup success verifies configuration/model loading, not Uzbek
recognition quality or live serving capacity.

Keep the repository's `av>=14.0.0,<19` constraint when installing dependencies.
The pinned faster-whisper 1.1.1 file decoder passes `metadata_errors` to PyAV;
a file-decoding smoke check reproduced a `TypeError` with PyAV 19.0.2. The
checked-in lock resolves PyAV 16.0.1, and validation also exercised PyAV 18.1.0.
The microphone's NumPy-array input bypasses that file decoder, but the offline
benchmark and other file-transcription paths require a compatible PyAV version.

Open `http://localhost:8000/call` for conversation and
`http://localhost:8000/live-stt` for the transcription interface. Browser
microphone capture requires localhost or HTTPS and microphone permission.
Do not start speaking before the interface reports readiness: downloading and
loading a model is different from opening a WebSocket.

For a key-free local transcription check, open `/live-stt` and uncheck
**“AI ovozli javob bersin”** before starting. That disables the standalone
agent while keeping microphone transcription enabled. The server exposes
`GET /api/live-stt/config` for provider/sample-rate configuration; this GET does
not load the model or prove model readiness. The WebSocket readiness event is
sent after initialization.

Server initialization has an 80-second session timeout and observes a browser
disconnect while loading. The browser's readiness timeout is 90 seconds.
An abandoned session therefore releases its admission slot. A native model
download/inference already running can finish in the bounded background worker;
cancelling the WebSocket does not forcibly terminate native computation.
Predownload with the warmup command so a multi-gigabyte download is not part of
a customer's first call.

The first model load may download several gigabytes depending on the selected
model. Keep the model cache persistent across application restarts. Alternatively,
point `LIVE_STT_LOCAL_MODEL` to an existing CTranslate2 model directory.
If a CUDA/model initialization error occurs, correct it or explicitly choose a
CPU/model configuration; the application does not silently replace the requested
model with a smaller model.

### CPU configuration

```env
LIVE_STT_PROVIDER=local
LIVE_STT_LANGUAGE=uz
LIVE_STT_LOCAL_MODEL=large-v3
LIVE_STT_LOCAL_DEVICE=cpu
LIVE_STT_LOCAL_COMPUTE_TYPE=int8
LIVE_STT_LOCAL_CPU_THREADS=4
```

CPU support is useful for development and evaluation. It is **not** a promise
that `large-v3` will keep up with live calls on a particular CPU. Compare a
deliberately selected `large-v3-turbo` model if compute is insufficient, and
measure its Uzbek accuracy before adopting it. The application should not build
an unlimited inference backlog when the hardware is slower than incoming audio.

### NVIDIA GPU configuration

```env
LIVE_STT_PROVIDER=local
LIVE_STT_LANGUAGE=uz
LIVE_STT_LOCAL_MODEL=large-v3
LIVE_STT_LOCAL_DEVICE=cuda
LIVE_STT_LOCAL_COMPUTE_TYPE=float16
```

The current faster-whisper installation guidance calls for CUDA 12 cuBLAS and
cuDNN 9 for the corresponding recent CTranslate2 runtime. Installing a CUDA
PyTorch wheel alone does not establish that CTranslate2 can load its required
libraries. Follow the [faster-whisper GPU installation guidance](https://github.com/SYSTRAN/faster-whisper#gpu)
for the actual environment. `int8_float16` is another explicit configuration to
benchmark when memory is constrained.

Do not infer serving capacity from a model's parameter count. Reserve memory
for the loaded model and active inference, then test concurrent calls on the
intended hardware. The engine shares a loaded model within one application
process; extra server processes can each load another copy. Start with one
application worker until memory and throughput have been measured.

### Optional hosted transcription

```env
LIVE_STT_PROVIDER=openai
LIVE_STT_LANGUAGE=uz
LIVE_STT_MODEL=gpt-4o-transcribe
OPENAI_API_KEY=your_existing_server_side_key
```

This is an explicit alternative provider. It does not make the local model a
paid API. Hosted STT, chat and TTS usage depends on the configured providers.
The legacy WebRTC Realtime page has its own settings; see
[LOCAL_REALTIME.md](LOCAL_REALTIME.md) for that separate integration.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `LIVE_STT_PROVIDER` | `local` | Local streaming recognition or explicitly selected `openai` |
| `LIVE_STT_LANGUAGE` | `uz` | `uz`, `ru`, `en` or `auto`; existing Turkish (`tr`) support is also retained |
| `LIVE_STT_LOCAL_MODEL` | `large-v3` | Multilingual model name or CTranslate2 model directory |
| `LIVE_STT_LOCAL_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `LIVE_STT_LOCAL_COMPUTE_TYPE` | `auto` | Engine chooses float16 for CUDA, int8 for CPU; explicit types accepted |
| `LIVE_STT_LOCAL_CPU_THREADS` | `4` | Local CPU inference threads |
| `LIVE_STT_LOCAL_BEAM_SIZE` | `5` | Decoding search width; larger is not an accuracy guarantee |
| `LIVE_STT_PARTIAL_INTERVAL_MS` | `800` | Target scheduling interval for partial decoding |
| `LIVE_STT_SILENCE_MS` | `600` | Silence used to end a speech turn; route accepts 300–1500 ms |
| `LIVE_STT_MIN_SPEECH_MS` | `128` | Minimum detected speech before accepting a turn |
| `LIVE_STT_PREFIX_MS` | `320` | Audio context retained before speech begins |
| `LIVE_STT_MAX_SEGMENT_SECONDS` | `15` | Bound on an individual decoding segment |
| `LIVE_STT_MAX_TURN_SECONDS` | `120` | Upper bound on one continuous turn |
| `LIVE_STT_MAX_PENDING_SEGMENTS` | `4` | Bound on queued finalized decoding segments |
| `LIVE_STT_MAX_SESSIONS` | `2` | Concurrent streaming sessions per application process; a cap, not measured serving capacity |
| `LIVE_STT_VAD_THRESHOLD` | `0.5` | Voice activity threshold |
| `LIVE_STT_VOCABULARY_HINTS` | Locale-specific terms | Short domain vocabulary; falls back to `STT_VOCABULARY_HINTS` |

`800 ms` is a scheduling setting, **not** measured time to first text. Similarly,
`600 ms` of endpoint silence is only one component of response delay. Capture,
network transfer, decoding, LLM generation and audio synthesis also take time.

Increase the silence interval if natural pauses split a sentence too early.
Decrease it only after testing quiet/hesitant speakers and numbers. VAD detects
speech activity, not whether the speaker has completed their meaning. A longer
pre-roll helps preserve first consonants; extremely aggressive noise handling
can remove quiet speech.

### Language choice and mixed speech

Whisper has Uzbek, Russian and English language tokens. That does not establish
equal recognition quality in those languages. An explicit `uz` setting avoids
relying on a short greeting to identify Uzbek. Select `ru` or `en` for those
calls; use `auto` for evaluation of mixed/unknown language.

Test phrases such as “buyurtmamni statusini tekshiring”, “заказ hali kelmadi”,
names and English product names. Automatic language identification can be
unstable on short or noisy utterances, and a monolingual Uzbek fine-tune may
damage Russian/English recognition. The default therefore remains multilingual;
specialist routing needs separate evidence.

## Research and model choice

Research checked on 2026-10-31. These are upstream descriptions/model-card
results, not measurements of Sahiy calls or this deployment. A model appearing
on GitHub/Hugging Face does not by itself establish permissive commercial terms.

| Project/model | Relevant capability | Limitation and decision |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [Whisper large-v3](https://github.com/openai/whisper) | MIT; CPU/GPU inference, quantization, multilingual model; Whisper's [tokenizer lists uz/ru/en](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) | Whisper needs a streaming wrapper and acoustic context. Selected as the configurable baseline; no Uzbek call accuracy claim. |
| [WhisperLive](https://github.com/collabora/WhisperLive) | MIT; nearly-live WebSocket transcription, hotwords, partial/completed segments | Useful external server alternative. This project reuses its existing faster-whisper dependency instead of adding a second ASR server. |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | Apache-2.0; streaming policies, VAD, native WS, model backends | A broader streaming framework. Compatible APIs are subsets; file transcription compatibility alone does not provide live streaming. |
| [SimulStreaming](https://github.com/ufal/SimulStreaming) | Streaming policy for offline speech models; current [license is MIT](https://github.com/ufal/SimulStreaming/blob/main/LICENCE.txt) | An alternative streaming-policy project. Upstream speed claims do not establish this application's latency. |
| [BuzzASR/uzbek](https://huggingface.co/BuzzASR/uzbek) | MIT Uzbek-only Whisper-large-v3 fine-tune. Card reports normalized WER 25.64% on FLEURS, 11.84% on Common Voice 25, 16.0% combined | Strong Uzbek specialist candidate, but monolingual and evaluated on those datasets. CT2 conversion/inference and mixed-language call behavior must be tested before adoption. |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0; streaming; Russian/English in official support list | Uzbek is absent from the base model's official language list. It is not selected as the trilingual default. |
| [Gearnode/qwen3-asr-uzbek-v2](https://huggingface.co/Gearnode/qwen3-asr-uzbek-v2) | Apache-2.0 Uzbek Qwen fine-tune | Card offers subjective quality labels but no numeric WER. Fine-tuned streaming behavior and Russian/English retention are unverified here. |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Apache-2.0, native streaming, Russian/English | Uzbek absent from the 13 supported languages; unsuitable as the universal recognizer here. |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0, 25 languages including Russian/English | Uzbek absent from the supported list. |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) | Apache-2.0, broad language coverage | Its [model documentation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/README.md) says the supplied inference pipeline does not currently support live streaming. |
| [Uzbek GigaAM fine-tune](https://huggingface.co/rustam1221/uzbek-asr-gigaam) | Uzbek specialist with published evaluation | Card explicitly says no streaming and requires a written license for commercial use; not adopted as a freely usable commercial default. |

The recognition layer also uses the Silero model bundled with faster-whisper.
[Silero VAD](https://github.com/snakers4/silero-vad) is MIT and supports 8/16 kHz
voice activity detection. It does not recognize words or fix transcripts.

Hotwords are model hints, not guaranteed recognition. The
[faster-whisper transcription implementation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
documents `hotwords`/`initial_prompt`; hotwords are ineffective when an explicit
`prefix` is used. Avoid long scripts that could bias the model toward words a
customer never spoke.

## Reproducible offline evaluation

Use [scripts/benchmark_stt.py](../scripts/benchmark_stt.py) to evaluate local
model output. It uses faster-whisper directly, does not call a paid speech API,
and permits model downloads only with the explicit `--allow-model-download`
option. By default the model must already be cached or provided as a local CT2
directory. It does not start the app or load chat/TTS credentials.

Prepare **50–100 representative recordings** with manually checked transcripts
as the first evaluation set: several speakers, quiet/normal speech, actual
headsets/phones, background noise, Uzbek/Russian/English and mixed utterances.
Use customer recordings only when available for this purpose; retain the data
outside the repository. A small initial set is useful for finding failures, not
for proving a universal accuracy guarantee.

Put the JSONL manifest beside its audio folder. Each line describes one file:

```jsonl
{"id":"uz-001","path":"audio/uz-001.wav","language":"uz","text":"AB-00123 raqamli buyurtmani bekor qilmang.","keywords":["buyurtmani"],"critical_values":["AB-00123"],"critical_phrases":["bekor qilmang"],"forbidden_phrases":["bekor qiling"]}
{"id":"ru-001","path":"audio/ru-001.wav","language":"ru","text":"Не отменяйте заказ 1042.","keywords":["заказ"],"critical_values":["1042"],"critical_phrases":["не отменяйте"]}
{"id":"en-001","path":"audio/en-001.wav","language":"en","text":"Do not cancel order 207.","keywords":["order"],"critical_values":["207"],"critical_phrases":["do not cancel"]}
{"id":"silence-001","path":"audio/silence.wav","language":"uz","text":"","keywords":["buyurtma"]}
```

For mixed speech set `language` to `auto` and retain the words actually spoken
in the reference. Do not translate the reference. Record numbers as spoken in
the ground truth according to one declared transcription convention: this
harness deliberately does not convert “ikki” into `2` or strip leading zeros.
If the model's number-formatting convention differs, investigate that separately
from an acoustic recognition failure.

Example, after downloading/loading the selected model once:

```bash
python scripts/benchmark_stt.py /path/to/evaluation/manifest.jsonl \
  --model large-v3 --device cuda --compute-type float16 \
  --hotwords "buyurtma, buyurtmani, Sahiy, kuryer" \
  --keyword buyurtma --keyword buyurtmani \
  --output outputs/asr-large-v3.json
```

For CPU use `--device cpu --compute-type int8`. Use a different output filename
for each model/configuration. `--overwrite` is required to replace an existing
report. `--warmup-file /path/to/warmup.wav` can run one unscored decode before
timing the evaluated recordings. Otherwise the first measured decode can include
first-inference overhead. Model initialization and optional warmup times are
reported separately.

To evaluate already captured live transcripts without downloading a model or
opening audio, add a `hypothesis` string to each record and run:

```bash
python scripts/benchmark_stt.py /path/to/evaluation/live-results.jsonl \
  --score-only --output outputs/asr-live-transcripts.json
```

The JSON report retains the reference and hypothesis verbatim and reports:

- **Corpus-weighted WER/CER**, overall and by declared language. CER includes
  spaces after normalization. Unicode compatibility, case, punctuation and
  Uzbek apostrophe formatting are normalized; words, digit strings and negation
  are not rewritten. An error rate can exceed 1.0. Undefined rates are `null`.
- **Domain term recall, precision and false positives.** Each configured phrase
  is checked once per sample using exact normalized tokens. `buyurtma` and
  `buyurtmani` are separate terms; no stemming guesses are made. A term that was
  absent in the reference but appears in the output counts as a false positive.
- **Critical value and phrase checks.** IDs preserve punctuation, leading zeros
  and case. Explicit required/forbidden phrases can catch negation failures.
  These presence checks do not validate semantic slot binding, all contradictions
  or the dialogue agent's interpretation; inspect important failures manually.
- **Silence hallucinations**, counted separately when an empty reference produces
  words. WER for a nonempty hypothesis against an empty reference is undefined,
  not zero; the insertion count remains in the report.
- **Offline decoding wall-time and real-time factor (RTF)**. RTF is total local
  file decode/inference time divided by total audio duration. It is **not live
  latency**, excludes model load, and does not include the microphone, WebSocket,
  endpointing, LLM or TTS. The harness decodes complete files without the live
  endpoint's external VAD; test the streaming path separately.

Reports contain the original text; keep real customer data and generated reports
out of commits. `outputs/` is already ignored by the repository.

## Live acceptance before production rollout

Run the same representative speech through the actual browser path. Record
capture device, browser, CPU/GPU, model/compute settings, concurrency and network
conditions together with the results. Compare to the previous setup using the
same recordings and reference transcripts.

1. Measure time to first partial text and end-of-speech to final text.
2. Measure end-of-speech to first audible agent reply separately. Inspect the
   contribution of endpointing, ASR, chat generation and TTS.
3. Check no lost first consonants, duplicated finals, truncated long turns or
   incorrect transcript ordering. Test pause/resume and several short turns.
   Include continuous speech longer than 15 seconds, with an order number or
   negation crossing the configured segment boundary. The 500 ms overlap trims
   duplicates when timestamps and text agree; a boundary-straddling word can
   still be revised after preceding segment text has been committed. This is
   conservative overlap handling, not a guarantee of perfect seam correction.
4. Check `buyurtma` and its forms, product names, phone/order numbers and amounts.
   Test both positive and negative instructions: “bekor qiling” / “bekor qilmang”.
5. Interrupt an audible reply and confirm obsolete text/audio does not resume.
   Confirm that an unplayed answer is not treated as delivered conversation
   history. Test disconnection during a reply or model loading, reconnection,
   denied microphone permission, silence and model load errors.
6. Run concurrent sessions on the intended server and check bounded memory and
   backlog. Set acceptance thresholds from the actual product requirements and
   measurements; none are claimed as achieved in this change.

A release decision needs the real audio evaluation and deployment environment.
Changing the branch or passing unit tests does not by itself deploy the service
or prove recognition accuracy. Preserve the previous configuration for an
explicit rollback, and enable the selected model only after these checks.
