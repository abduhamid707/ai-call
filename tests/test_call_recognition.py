import asyncio

import pytest

from app import elevenlabs_call as call


@pytest.mark.parametrize("text", ["ha", "yo'q", "7", "178", "+998991134543", "thank you", "www.sahiy.uz", "Buyurtmani bekor qilmang"])
def test_valid_short_answers_and_identifiers_are_not_blacklisted(text):
    assert not call._is_hallucinated_transcript(text)


@pytest.mark.parametrize("text", ["", "   ", "..."])
def test_empty_transcripts_are_ignored(text):
    assert call._is_hallucinated_transcript(text)


def test_language_and_negation_guidance_do_not_rewrite_customer_transcript():
    transcript = "178 buyurtmani bekor qilmang"
    messages = call._build_call_messages(transcript, "ai_call_center", [], "ru-RU")
    assert "Отвечайте на русском" in messages[0]["content"]
    assert "negation" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": transcript}
    assert sum(m["role"] == "system" for m in messages) == 1


def test_uncertain_asr_adds_clarification_instruction_without_mutating_words(monkeypatch):
    seen = []
    async def llm(messages, **kwargs):
        seen.extend(messages)
        return "Buyurtma raqamini qaytarasizmi?"
    async def tts(**kwargs):
        return b"audio"
    monkeypatch.setattr(call, "fast_llm_reply_async", llm)
    monkeypatch.setattr(call, "generate_edge_tts_audio", tts)
    result = asyncio.run(call._process_text_turn("178", "ai_call_center", [], "uz", "free_ai", "test", "voice", 1, False, needs_confirmation=True))
    assert seen[-1]["content"] == "178"
    assert "flagged this utterance as uncertain" in seen[0]["content"]
    assert result["user_text"] == "178"
