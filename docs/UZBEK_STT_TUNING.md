# Uzbek speech recognition tuning

This change targets the non-Realtime microphone transcription path (`app/transcription.py`).

- `STT_SILENCE_THRESHOLD` defaults to 140 instead of 300, and enhanced recording measures background noise to adjust its speech gate.
- A 400 ms pre-roll prevents clipping the first consonant when speech starts.
- Local Faster-Whisper defaults to multilingual `small` (higher accuracy, higher CPU/RAM and slower than `base`).
- `STT_LANGUAGE=uz` can help if speakers consistently use Uzbek; omit for mixed-language calls.
- `STT_VOCABULARY_HINTS="tovarim, buyurtma, mahsulot, ..."` can bias recognition towards industry words for supported transcription models. Keep the list short and truthful.
- Standalone hesitation sounds are removed conservatively; semantic word corrections are intentionally **not** guessed.

This does not modify the separate WebRTC OpenAI Realtime audio pipeline. Test that mode separately.

## Quality assurance

Run `pytest -q tests/test_speech_cleanup.py`. Test with recorded Uzbek samples of whispering/quiet words, partial pronunciations, filler-only speech, nearby noise, and meaningful words containing 'em' or 'hm'. Measure word error rate before and after. The changes are not a guarantee that uncertain audio can be recovered.
