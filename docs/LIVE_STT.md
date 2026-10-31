# Live speech-to-text

The current integration and research/deployment guide is
[STREAMING_ASR.md](STREAMING_ASR.md).

- `LIVE_STT_PROVIDER=local` selects the local faster-whisper streaming engine
  (multilingual `large-v3` by default). Local ASR itself needs no OpenAI key.
- `LIVE_STT_PROVIDER=openai` explicitly selects the existing hosted transcription
  adapter and requires the server-side `OPENAI_API_KEY`.
- `/call` uses the streaming recognition path for conversation. `/live-stt`
  provides a standalone interface, including transcript-only use and an agent
  mode. Agent chat and speech synthesis keep their own provider requirements.
- Language choices are Uzbek (`uz`), Russian (`ru`), English (`en`) and automatic
  (`auto`). Test mixed speech separately; language coverage is not an accuracy
  guarantee.

Microphone access needs localhost or HTTPS and browser permission. Wait for
model readiness before speaking. See the current guide for `.env` settings,
CPU/GPU installation, evaluation commands, model/license comparisons, and live
acceptance criteria.

The separate WebRTC Realtime page is configured independently; see
[LOCAL_REALTIME.md](LOCAL_REALTIME.md).
