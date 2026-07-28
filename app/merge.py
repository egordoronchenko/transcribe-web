"""Merge two transcripts of the same audio: Whisper timings + Gemini speakers.

Whisper returns frame-accurate timestamps but no idea who is talking. Gemini
labels speakers but its timestamps drift by up to a quarter of the recording,
so the two cannot be joined by time. They are joined by text instead: both
transcribe the same words, so aligning the word sequences maps every Whisper
segment onto the speaker who said it.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path

from .core import to_srt


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _flatten(segments: list[dict], speaker_of) -> tuple[list[str], list]:
    """Word list plus a per-word tag (segment index or speaker)."""
    words: list[str] = []
    tags: list = []
    for i, s in enumerate(segments):
        w = _words(s.get("text") or "")
        words.extend(w)
        tags.extend([speaker_of(i, s)] * len(w))
    return words, tags


def assign_speakers(whisper_segments: list[dict], gemini_segments: list[dict]) -> tuple[list[dict], float]:
    """Copy speaker labels from gemini_segments onto whisper_segments.

    Returns (segments, coverage) where coverage is the share of Whisper words
    that were matched to Gemini text — a low value means the two transcripts
    disagree too much for the labels to be trusted.
    """
    w_words, w_tags = _flatten(whisper_segments, lambda i, s: i)
    g_words, g_tags = _flatten(gemini_segments, lambda i, s: s.get("speaker") or "S?")
    if not w_words or not g_words:
        return [dict(s, speaker="S1") for s in whisper_segments], 0.0

    speaker_per_word: list[str | None] = [None] * len(w_words)
    matcher = difflib.SequenceMatcher(None, w_words, g_words, autojunk=False)
    matched = 0
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            speaker_per_word[a + k] = g_tags[b + k]
        matched += size

    # words Gemini phrased differently inherit the nearest label before them
    last = None
    for i, sp in enumerate(speaker_per_word):
        if sp is None:
            speaker_per_word[i] = last
        else:
            last = sp
    fallback = next((s for s in speaker_per_word if s), "S1")
    speaker_per_word = [s or fallback for s in speaker_per_word]

    # majority vote inside each Whisper segment
    votes: dict[int, dict[str, int]] = {}
    for idx, sp in zip(w_tags, speaker_per_word):
        votes.setdefault(idx, {})[sp] = votes.setdefault(idx, {}).get(sp, 0) + 1
    out = []
    for i, s in enumerate(whisper_segments):
        tally = votes.get(i)
        speaker = max(tally, key=tally.get) if tally else fallback
        out.append({"speaker": speaker, "start": s["start"], "end": s["end"], "text": s["text"]})
    return out, matched / len(w_words)


def to_text(segments: list[dict]) -> str:
    blocks: list[list] = []
    for s in segments:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if blocks and blocks[-1][0] == s["speaker"]:
            blocks[-1][1].append(t)
        else:
            blocks.append([s["speaker"], [t]])
    return "\n\n".join(f"[{b[0]}] " + " ".join(b[1]) for b in blocks) + "\n"


def write_merged(
    out_dir: Path,
    stem: str,
    whisper_result: dict,
    gemini_result: dict,
    cfg,
    log=lambda s: None,
    suffix: str = "merged",
) -> dict:
    t0 = time.perf_counter()
    segments, coverage = assign_speakers(whisper_result["segments"], gemini_result["segments"])
    speakers = gemini_result.get("speakers") or []
    changes = sum(1 for a, b in zip(segments, segments[1:]) if a["speaker"] != b["speaker"])
    log(f"слияние: размечено {len(segments)} сегментов, {changes} смен говорящего, "
        f"совпадение текстов {coverage * 100:.0f}%")
    if coverage < 0.5:
        log("внимание: расшифровки сильно расходятся, метки говорящих могут быть неточными")

    text = to_text(segments)
    srt = to_srt([{**s, "text": f"[{s['speaker']}] {s['text']}"} for s in segments])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.{suffix}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.{suffix}.srt").write_text(srt, encoding="utf-8")
    payload = {
        "mode": suffix,
        "built_from": {"timings": cfg.model, "speakers": cfg.gemini_model},
        "language": cfg.language,
        "timestamps": "точные (от Whisper)",
        "text_match": round(coverage, 3),
        "speakers": speakers,
        "segments": segments,
        "text": text,
    }
    (out_dir / f"{stem}.{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "success": True,
        "mode": suffix,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "audio_duration_sec": whisper_result["meta"]["audio_duration_sec"],
        "timings_model": cfg.model,
        "speakers_model": cfg.gemini_model,
        "segments": len(segments),
        "speaker_changes": changes,
        "speakers": len(speakers),
        "text_match": round(coverage, 3),
    }
    (out_dir / f"{stem}.{suffix}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": meta, "segments": segments, "speakers": speakers}
