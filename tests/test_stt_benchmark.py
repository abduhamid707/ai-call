"""Metric tests run without model weights, microphone, GPU or paid API calls."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_stt import (
    edit_distance, literal_present, load_manifest, normalize_text,
    phrase_present, score_transcript, summarize,
)


def test_normalization_keeps_uzbek_negation_digits_and_other_languages():
    assert normalize_text("O‘ZBEKCHA: Buyurtmani bekor QILMANG! 00123") == "o'zbekcha buyurtmani bekor qilmang 00123"
    assert normalize_text("Заказ № 12. Don't cancel!") == "заказ no 12 don't cancel"
    assert normalize_text("oʻzbek oʼzbek o`zbek") == "o'zbek o'zbek o'zbek"


def test_levenshtein_counts_substitutions_deletions_and_insertions():
    assert edit_distance(["buyurtmani", "bekor", "qilmang"], ["buyurtmani", "bekor", "qiling"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "c", "d", "e"]) == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("kitten", "sitting") == 3


def test_empty_reference_insertions_are_not_reported_as_zero_wer():
    result = score_transcript("", "Buyurtma")
    assert result["wer"] is None
    assert result["word_errors"] == 1
    assert result["empty_reference_insertion"] is True
    assert score_transcript("", "")["wer"] == 0


def test_aggregate_wer_is_weighted_by_reference_words():
    result = summarize([
        score_transcript("one", "wrong"),
        score_transcript("one two three four five six seven eight nine", "one two three four five six seven eight nine"),
    ])
    assert result["wer"] == 0.1
    assert result["reference_words"] == 10
    assert result["audio_seconds"] is None
    assert result["offline_decode_rtf"] is None


def test_keyword_recall_and_false_positives_are_exact_token_phrases():
    results = [
        score_transcript("Buyurtma kerak", "Buyurtma kerak", keywords=["buyurtma", "buyurtma"]),
        score_transcript("Buyurtma kerak", "Yordam kerak", keywords=["buyurtma"]),
        score_transcript("Salom", "Buyurtma", keywords=["buyurtma"]),
    ]
    summary = summarize(results)
    assert summary["keyword_true_positives"] == 1
    assert summary["keyword_false_negatives"] == 1
    assert summary["keyword_false_positives"] == 1
    assert summary["keyword_recall"] == 0.5
    assert summary["keyword_precision"] == 0.5
    assert not phrase_present("buyurtma", "buyurtmani")
    assert phrase_present("yetkazib berish", "Yetkazib, berish haqida")


def test_critical_values_preserve_leading_zeros_id_punctuation_and_case():
    assert literal_present("AB-00123", "ID AB-00123.")
    assert not literal_present("AB-00123", "ID AB00123")
    assert not literal_present("AB-00123", "ID AB-001234")
    assert not literal_present("00123", "ID 123")
    assert not literal_present("AB-00123", "ID ab-00123")
    assert not literal_present("+998901234567", "+9989012345678")
    assert not literal_present("123", "-123")
    assert not literal_present("123", "AB-123")
    assert not literal_present("123", "123.4")
    assert not literal_present("123", "123,000")
    assert not literal_present("4", "123.4")
    assert not literal_present("123", "123/4")
    assert literal_present("123", "Order 123.")


def test_critical_negation_failure_is_explicit():
    result = score_transcript(
        "Buyurtmani bekor qilmang", "Buyurtmani bekor qiling",
        critical_phrases=["bekor qilmang"], forbidden_phrases=["bekor qiling"],
    )
    assert result["wer"] == pytest.approx(1 / 3)
    assert [check["passed"] for check in result["critical_checks"]] == [False, False]
    assert summarize([result])["critical_checks_failed"] == 2


def test_offline_rtf_uses_total_audio_not_average_of_per_file_rates():
    first = {**score_transcript("a", "a"), "audio_seconds": 1.0, "decode_seconds": 1.0}
    second = {**score_transcript("b", "b"), "audio_seconds": 9.0, "decode_seconds": 1.0}
    assert summarize([first, second])["offline_decode_rtf"] == 0.2


def test_manifest_resolves_paths_relative_to_manifest(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps({"path": "audio.wav", "text": "salom", "language": "uz"}) + "\n", encoding="utf-8")
    records = load_manifest(manifest)
    assert records[0]["audio_path"] == str(audio.resolve())
    assert records[0]["keywords"] == []


@pytest.mark.parametrize("record", [
    {"path": "x.wav", "text": "salom", "language": "xx"},
    {"path": "x.wav", "text": "salom", "language": "uz", "keywords": "salom"},
    {"path": "x.wav", "text": "ID 123", "language": "uz", "critical_values": ["00123"]},
    {"path": "x.wav", "text": "bekor qilmang", "language": "uz", "forbidden_phrases": ["bekor qilmang"]},
])
def test_manifest_rejects_inconsistent_expectations(tmp_path, record):
    manifest = tmp_path / "samples.jsonl"
    record["hypothesis"] = ""
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(manifest, score_only=True)


def test_score_only_cli_needs_no_audio_or_model_and_preserves_raw_output(tmp_path):
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps({
        "path": "not-present.wav", "text": "Buyurtmani bekor qilmang. AB-00123",
        "language": "uz", "hypothesis": "Buyurtmani bekor qiling. AB-00123",
        "critical_values": ["AB-00123"], "critical_phrases": ["bekor qilmang"],
    }), encoding="utf-8")
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_stt.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(manifest), "--output", str(output), "--score-only"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "score_only"
    assert report["aggregate"]["critical_checks_failed"] == 1
    assert report["samples"][0]["hypothesis_raw"] == "Buyurtmani bekor qiling. AB-00123"
    assert report["configuration"]["model"] is None
    assert report["aggregate"]["offline_decode_rtf"] is None
