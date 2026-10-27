# Live OpenAI Speech-to-Text

Start server with `OPENAI_API_KEY` configured, then open **/live-stt** (e.g. `http://localhost:8000/live-stt`). Microphone requires localhost or HTTPS and browser permission. This mode uses billable OpenAI API traffic (a ChatGPT subscription does not automatically include API credit).

Browser microphone -> PCM mono 24 kHz chunks -> local FastAPI WebSocket `/ws/live-stt` -> server-side OpenAI Realtime transcription WS -> incremental delta and completed events -> live transcript UI.

Environment:
- `OPENAI_API_KEY` required server-side.
- `LIVE_STT_MODEL=gpt-4o-transcribe` default
- `LIVE_STT_LANGUAGE=uz` default; clear for multilingual auto detection
- `LIVE_STT_VOCABULARY_HINTS="tovarim, buyurtma, mahsulot"` optional vocabulary hints

No API key exposed in the browser. The transcript panel is intentionally standalone; it does **not** yet replace legacy record-and-transcribe on the main chat page or inject transcripts into the assistant conversation. WebRTC realtime conversation is also unchanged. Review this scope before enabling for phone calls.

Quick validation: `pytest -q tests/test_live_stt.py tests/test_speech_cleanup.py`. Live tests must be run with a microphone and funded API key. Check Uzbek 'tovarim', pauses, and quiet speech and confirm partial/final update ordering.
