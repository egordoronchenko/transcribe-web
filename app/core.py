"""Transcription pipeline: fixed chunks + overlap + initial prompt + stitch.

Adapted from scripts/or_transcribe_pro.py (the reference cloud-pro pipeline).
Polish is intentionally not implemented: raw Whisper output only.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from openai import AsyncOpenAI
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_random_exponential

CHUNK_SEC = 600.0
OVERLAP_SEC = 8.0
CONCURRENCY = 3
TRANSCRIPTS_DIR = "transcripts"


@dataclass
class Config:
    api_key: str
    base_url: str
    model: str
    language: str
    prompt: str | None

    @classmethod
    def from_env(cls) -> "Config":
        prompt = None
        prompt_file = os.environ.get("PROMPT_FILE", "/app/prompts/whisper_initial_prompt.txt")
        if prompt_file and Path(prompt_file).is_file():
            prompt = Path(prompt_file).read_text(encoding="utf-8").strip() or None
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3"),
            language=os.environ.get("LANGUAGE", "ru"),
            prompt=prompt,
        )


def format_ts(seconds: float, sep: str = ",") -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def split_fixed(audio: AudioSegment, chunk_sec: float, overlap_sec: float) -> list[dict]:
    chunk_ms = int(chunk_sec * 1000)
    overlap_ms = int(overlap_sec * 1000)
    step = max(chunk_ms - overlap_ms, 1000)
    duration = len(audio)
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        end = min(start + chunk_ms, duration)
        chunks.append({"audio": audio[start:end], "start_ms": start, "index": idx})
        idx += 1
        if end >= duration:
            break
        start += step
    return chunks


def stitch_segments(chunk_results: list[dict], overlap_sec: float) -> list[dict]:
    merged: list[dict] = []
    for i, item in enumerate(chunk_results):
        offset = item["start_ms"] / 1000.0
        segs = item.get("segments") or []
        for seg in segs:
            start = float(seg.get("start", 0)) + offset
            end = float(seg.get("end", 0)) + offset
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if i > 0 and overlap_sec > 0:
                prev_exclusive_end = chunk_results[i - 1]["start_ms"] / 1000.0 + (
                    (chunk_results[i - 1].get("chunk_duration_sec") or 0) - overlap_sec
                )
                if start < prev_exclusive_end - 0.15:
                    continue
            merged.append({"start": start, "end": end, "text": text})
    return merged


class Transcriber:
    def __init__(self, cfg: Config, concurrency: int = CONCURRENCY):
        self.client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.cfg = cfg
        self.sem = asyncio.Semaphore(concurrency)

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(8))
    async def _transcribe_file(self, path: Path) -> dict:
        kwargs = {
            "model": self.cfg.model,
            "file": path.open("rb"),
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if self.cfg.language and self.cfg.language != "auto":
            kwargs["language"] = self.cfg.language
        if self.cfg.prompt:
            kwargs["prompt"] = self.cfg.prompt[:800]
        extra = {
            "extra_body": {
                "provider": {
                    "order": ["groq", "together"],
                    "allow_fallbacks": True,
                    "options": {
                        "groq": {"prompt": self.cfg.prompt[:800]} if self.cfg.prompt else {},
                    },
                }
            }
        }
        try:
            result = await self.client.audio.transcriptions.create(**kwargs, **extra)
        finally:
            kwargs["file"].close()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return json.loads(result.model_dump_json())

    async def transcribe_chunk(self, chunk_path: Path, start_ms: int, duration_sec: float, index: int) -> dict:
        async with self.sem:
            data = await self._transcribe_file(chunk_path)
            data["start_ms"] = start_ms
            data["chunk_duration_sec"] = duration_sec
            data["index"] = index
            return data


def prepare_chunks(audio_path: Path, work_dir: Path) -> tuple[list[dict], float]:
    """Blocking: load audio, split, export chunk mp3 files. Returns (chunk infos, duration_sec)."""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    duration_sec = len(audio) / 1000.0
    chunks = split_fixed(audio, CHUNK_SEC, OVERLAP_SEC)
    infos = []
    for c in chunks:
        out = work_dir / f"chunk_{c['index']:04d}.mp3"
        c["audio"].export(out, format="mp3", bitrate="64k")
        infos.append({
            "path": out,
            "start_ms": c["start_ms"],
            "duration_sec": len(c["audio"]) / 1000.0,
            "index": c["index"],
        })
    return infos, duration_sec


async def transcribe_one(
    audio_path: Path,
    out_dir: Path,
    stem: str,
    cfg: Config,
    work_dir: Path,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> dict:
    """Transcribe one prepared mp3 file. Writes <stem>.txt/.srt/.json/.meta.json in out_dir."""
    t0 = time.perf_counter()
    log(f"чанкую {audio_path.name}")
    infos, duration_sec = await asyncio.to_thread(prepare_chunks, audio_path, work_dir)
    log(f"{len(infos)} чанков по {CHUNK_SEC:.0f} с, overlap {OVERLAP_SEC:.0f} с")

    tr = Transcriber(cfg)
    done = 0

    async def one(info: dict) -> dict:
        nonlocal done
        data = await tr.transcribe_chunk(info["path"], info["start_ms"], info["duration_sec"], info["index"])
        done += 1
        log(f"чанк {done}/{len(infos)} готов")
        if on_progress:
            await on_progress(done, len(infos))
        return data

    if on_progress:
        await on_progress(0, len(infos))
    results = await asyncio.gather(*[one(i) for i in infos])
    results = sorted(results, key=lambda x: x["index"])
    segments = stitch_segments(results, OVERLAP_SEC)
    text = "\n".join(s["text"] for s in segments if s["text"]).strip() + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.srt").write_text(to_srt(segments), encoding="utf-8")
    payload = {
        "model": cfg.model,
        "language": cfg.language,
        "segments": segments,
        "text": text,
        "chunks": len(infos),
        "chunk_sec": CHUNK_SEC,
        "overlap_sec": OVERLAP_SEC,
        "prompt_used": bool(cfg.prompt),
        "polished": False,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "success": True,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "audio_duration_sec": round(duration_sec, 2),
        "model": cfg.model,
        "chunks": len(infos),
        "segments": len(segments),
        "source_audio": str(audio_path),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
