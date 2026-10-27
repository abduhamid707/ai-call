from app.live_stt import transcription_session

def test_live_transcription_session_is_audio_only():
    session = transcription_session()
    assert session["type"] == "transcription"
    audio = session["audio"]["input"]
    assert audio["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio["transcription"]["model"] == "gpt-4o-transcribe"
    assert audio["turn_detection"]["type"] == "server_vad"
    assert audio["turn_detection"]["prefix_padding_ms"] >= 300
