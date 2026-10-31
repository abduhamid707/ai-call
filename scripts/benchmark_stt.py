#!/usr/bin/env python3
"""Evaluate local ASR against a JSONL manifest; never call a speech API.

The report contains recordings/transcripts and must be kept out of source control.
Offline decoding real-time factor is NOT live end-of-turn latency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


APOSTROPHES = str.maketrans({char: "'" for char in "‘’ʻʼ`"})
LANGUAGES = {"uz", "ru", "en", "auto"}


def normalize_text(text: str) -> str:
    """Case/punctuation normalization only: retain digits, words and negation."""
    text = unicodedata.normalize("NFKC", text).translate(APOSTROPHES).casefold()
    cleaned = []
    for index, char in enumerate(text):
        if unicodedata.category(char)[0] in {"L", "N", "M"}:
            cleaned.append(char)
        elif (
            char == "'"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalpha()
            and text[index + 1].isalpha()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Levenshtein distance with O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for col, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def error_rate(errors: int, denominator: int) -> float | None:
    # An insertion into empty-reference silence is important, but WER is undefined.
    if denominator == 0:
        return 0.0 if errors == 0 else None
    return errors / denominator


def phrase_present(phrase: str, text: str) -> bool:
    wanted, actual = normalize_text(phrase).split(), normalize_text(text).split()
    return bool(wanted) and any(
        actual[start:start + len(wanted)] == wanted
        for start in range(len(actual) - len(wanted) + 1)
    )


def literal_present(value: str, text: str) -> bool:
    """Case-sensitive literal match, preserving ID punctuation and leading zeros.

    Only Unicode compatibility and whitespace are normalized. Word boundaries
    prevent an expected 123 from passing when the transcript actually says 1234.
    This checks presence, not whether a value belongs to the correct semantic slot.
    """
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    # Avoid accepting 123 inside -123, AB-123, 123.4, 123/4, or 123,000.
    # A sentence-ending comma/full stop still counts as a word boundary.
    return bool(value) and re.search(
        r"(?<![\w+\-/])(?<!\d[.,])" + re.escape(value)
        + r"(?![\w+\-/]|[.,]\d)", text,
    ) is not None


def score_transcript(
    reference: str,
    hypothesis: str,
    *,
    keywords: Sequence[str] = (),
    critical_values: Sequence[str] = (),
    critical_phrases: Sequence[str] = (),
    forbidden_phrases: Sequence[str] = (),
) -> dict[str, Any]:
    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)
    words = reference_normalized.split()
    word_errors = edit_distance(words, hypothesis_normalized.split())
    character_errors = edit_distance(reference_normalized, hypothesis_normalized)
    terms = []
    for term in dict.fromkeys(keywords):
        expected = phrase_present(term, reference)
        found = phrase_present(term, hypothesis)
        terms.append({
            "term": term,
            "reference_present": expected,
            "hypothesis_present": found,
            "status": "matched" if expected and found else "missed" if expected
            else "false_positive" if found else "absent",
        })
    checks = [
        {"kind": "critical_value", "expected": value, "passed": literal_present(value, hypothesis)}
        for value in critical_values
    ] + [
        {"kind": "critical_phrase", "expected": phrase, "passed": phrase_present(phrase, hypothesis)}
        for phrase in critical_phrases
    ] + [
        {"kind": "forbidden_phrase", "forbidden": phrase, "passed": not phrase_present(phrase, hypothesis)}
        for phrase in forbidden_phrases
    ]
    return {
        "reference_raw": reference,
        "hypothesis_raw": hypothesis,
        "reference_normalized": reference_normalized,
        "hypothesis_normalized": hypothesis_normalized,
        "reference_words": len(words),
        "word_errors": word_errors,
        "wer": error_rate(word_errors, len(words)),
        "reference_characters": len(reference_normalized),
        "character_errors": character_errors,
        "cer": error_rate(character_errors, len(reference_normalized)),
        "empty_reference_insertion": not words and bool(hypothesis_normalized),
        "keyword_checks": terms,
        "critical_checks": checks,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    word_errors = sum(item["word_errors"] for item in results)
    words = sum(item["reference_words"] for item in results)
    character_errors = sum(item["character_errors"] for item in results)
    characters = sum(item["reference_characters"] for item in results)
    terms = [term for item in results for term in item["keyword_checks"]]
    matched = sum(term["status"] == "matched" for term in terms)
    missed = sum(term["status"] == "missed" for term in terms)
    false_positives = sum(term["status"] == "false_positive" for term in terms)
    checks = [check for item in results for check in item["critical_checks"]]
    measured = [item for item in results if item.get("decode_seconds") is not None]
    audio_seconds = sum(item["audio_seconds"] for item in measured)
    decode_seconds = sum(item["decode_seconds"] for item in measured)
    return {
        "samples": len(results),
        "reference_words": words,
        "word_errors": word_errors,
        "wer": error_rate(word_errors, words),
        "reference_characters": characters,
        "character_errors": character_errors,
        "cer": error_rate(character_errors, characters),
        "empty_reference_insertions": sum(item["empty_reference_insertion"] for item in results),
        "keyword_true_positives": matched,
        "keyword_false_negatives": missed,
        "keyword_false_positives": false_positives,
        "keyword_recall": matched / (matched + missed) if matched + missed else None,
        "keyword_precision": matched / (matched + false_positives) if matched + false_positives else None,
        "critical_checks": len(checks),
        "critical_checks_failed": sum(not check["passed"] for check in checks),
        "audio_seconds": audio_seconds if measured else None,
        "decode_seconds": decode_seconds if measured else None,
        "offline_decode_rtf": decode_seconds / audio_seconds if audio_seconds > 0 else None,
    }


def _string_list(record: dict[str, Any], key: str, line: int) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Manifest line {line}: {key} must be an array of nonempty strings")
    return value


def load_manifest(path: Path, *, score_only: bool = False) -> list[dict[str, Any]]:
    records, ids = [], set()
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Manifest line {line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number}: expected an object")
            for key in ("path", "text", "language"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"Manifest line {line_number}: {key} must be a string")
            if not record["path"].strip():
                raise ValueError(f"Manifest line {line_number}: path is empty")
            if record["language"] not in LANGUAGES:
                raise ValueError(f"Manifest line {line_number}: language must be uz, ru, en or auto")
            if score_only and not isinstance(record.get("hypothesis"), str):
                raise ValueError(f"Manifest line {line_number}: score-only needs a hypothesis string")
            record_id = record.get("id", str(line_number))
            if not isinstance(record_id, str) or not record_id.strip() or record_id in ids:
                raise ValueError(f"Manifest line {line_number}: id must be a unique nonempty string")
            ids.add(record_id)
            audio_path = Path(record["path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            audio_path = audio_path.resolve()
            if not score_only and not audio_path.is_file():
                raise ValueError(f"Manifest line {line_number}: audio file does not exist: {audio_path}")
            record = {**record, "id": record_id, "audio_path": str(audio_path)}
            for key in ("keywords", "critical_values", "critical_phrases", "forbidden_phrases"):
                record[key] = _string_list(record, key, line_number)
            for value in record["critical_values"]:
                if not literal_present(value, record["text"]):
                    raise ValueError(f"Manifest line {line_number}: critical value {value!r} is absent from reference")
            for phrase in record["critical_phrases"]:
                if not phrase_present(phrase, record["text"]):
                    raise ValueError(f"Manifest line {line_number}: critical phrase {phrase!r} is absent from reference")
            for phrase in record["forbidden_phrases"]:
                if phrase_present(phrase, record["text"]):
                    raise ValueError(f"Manifest line {line_number}: forbidden phrase {phrase!r} occurs in reference")
            records.append(record)
    if not records:
        raise ValueError("Manifest has no samples")
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL with path, text and language per recording")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; contains raw transcripts")
    parser.add_argument("--model", default="large-v3", help="Cached faster-whisper model name or local CTranslate2 directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="default", help="CTranslate2 compute type, e.g. float16 or int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--hotwords", default="", help="Decoder hints; separate from terms evaluated by --keyword")
    parser.add_argument("--keyword", action="append", default=[], help="Phrase to evaluate in every sample; repeat as needed")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Hugging Face model downloads; disabled by default")
    parser.add_argument("--warmup-file", type=Path, help="Optional audio to decode before timing evaluation samples")
    parser.add_argument("--score-only", action="store_true", help="Score manifest hypothesis strings without model/audio access")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cpu_threads < 1 or args.beam_size < 1:
            raise ValueError("cpu-threads and beam-size must be positive")
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output already exists; choose a new path or explicitly use --overwrite")
        records = load_manifest(args.manifest.resolve(), score_only=args.score_only)
        if args.warmup_file and not args.warmup_file.is_file():
            raise ValueError("warmup-file does not exist")
        if args.score_only and (args.warmup_file or args.allow_model_download):
            raise ValueError("score-only cannot be combined with warmup-file or allow-model-download")

        model, model_load_seconds, warmup_seconds = None, None, None
        if not args.score_only:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ValueError("Install the repository dependencies (faster-whisper is required for decoding)") from exc
            started = time.perf_counter()
            model = WhisperModel(
                args.model,
                device=args.device,
                compute_type=args.compute_type,
                cpu_threads=args.cpu_threads,
                local_files_only=not args.allow_model_download,
            )
            model_load_seconds = time.perf_counter() - started
            if args.warmup_file:
                started = time.perf_counter()
                segments, _ = model.transcribe(
                    str(args.warmup_file), beam_size=args.beam_size,
                    temperature=0.0, condition_on_previous_text=False,
                )
                list(segments)
                warmup_seconds = time.perf_counter() - started

        results = []
        for index, record in enumerate(records, 1):
            decode_seconds, audio_seconds, detected_language = None, None, None
            if args.score_only:
                hypothesis = record["hypothesis"]
            else:
                started = time.perf_counter()
                segments, info = model.transcribe(
                    record["audio_path"],
                    language=None if record["language"] == "auto" else record["language"],
                    task="transcribe", beam_size=args.beam_size, temperature=0.0,
                    condition_on_previous_text=False,
                    hotwords=args.hotwords or None,
                    # Evaluate the complete acoustic input, including silence. The
                    # live endpoint's external VAD is tested separately in the app.
                    vad_filter=False,
                )
                hypothesis = "".join(segment.text for segment in segments).strip()
                decode_seconds = time.perf_counter() - started
                audio_seconds = float(info.duration)
                detected_language = info.language
                if not math.isfinite(audio_seconds) or audio_seconds < 0:
                    raise ValueError(f"Invalid audio duration for sample {record['id']}")
            result = score_transcript(
                record["text"], hypothesis,
                keywords=[*args.keyword, *record["keywords"]],
                critical_values=record["critical_values"],
                critical_phrases=record["critical_phrases"],
                forbidden_phrases=record["forbidden_phrases"],
            )
            result.update({
                "id": record["id"], "path": record["path"],
                "language": record["language"], "detected_language": detected_language,
                "audio_seconds": audio_seconds, "decode_seconds": decode_seconds,
                "offline_decode_rtf": decode_seconds / audio_seconds if audio_seconds else None,
            })
            results.append(result)
            print(f"Scored {index}/{len(records)}: {record['id']}", file=sys.stderr)

        languages = defaultdict(list)
        for result in results:
            languages[result["language"]].append(result)
        report = {
            "schema_version": 1,
            "mode": "score_only" if args.score_only else "offline_local_decode",
            "normalization": "Unicode NFKC; casefold; punctuation to spaces; Uzbek apostrophes unified; no digit/semantic rewriting. CER includes normalized spaces.",
            "metric_notes": "Corpus-weighted WER/CER, not average sentence rates. Null means undefined/no applicable data. Term scores count one presence check per configured phrase per sample. Exact checks do not validate semantic slot binding. Offline RTF is decoding wall-time/audio duration, NOT live latency.",
            "configuration": {
                "model": args.model if not args.score_only else None,
                "device_requested": args.device if not args.score_only else None,
                "compute_type_requested": args.compute_type if not args.score_only else None,
                "cpu_threads": args.cpu_threads, "beam_size": args.beam_size,
                "hotwords": args.hotwords, "global_keywords": args.keyword,
                "vad_filter": False, "condition_on_previous_text": False,
                "model_download_allowed": args.allow_model_download,
                "model_load_seconds": model_load_seconds, "warmup_seconds": warmup_seconds,
            },
            "aggregate": summarize(results),
            "by_language": {language: summarize(items) for language, items in sorted(languages.items())},
            "samples": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        with args.output.open(mode, encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Report: {args.output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
