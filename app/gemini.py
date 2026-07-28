"""Gemini backend: transcription with speaker diarization.

Unlike Whisper (a dedicated ASR at /audio/transcriptions), this sends the audio
to a multimodal LLM via /chat/completions and asks for a structured transcript
with speaker labels. Timestamps come from the model and drift — they are marked
approximate everywhere in the output.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Awaitable, Callable

import httpx

CHUNK_SEC = 1800.0      # 30 min per request: keeps the base64 payload well under the limit
OVERLAP_SEC = 60.0      # shared tail so the next chunk hears the same voices
BITRATE = "32k"         # mono 32 kbps: ~7 MB per 30 min, ~9 MB in base64

SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "description": {"type": "string"}},
                "required": ["id", "description"],
                "additionalProperties": False,
            },
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "start", "end", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["speakers", "segments"],
    "additionalProperties": False,
}

BASE_PROMPT = """Расшифруй эту аудиозапись полностью и дословно, от первой до последней секунды.
Не пересказывай, не сокращай и не пропускай куски. Сохраняй запинки и слова-паразиты.
Определи разных говорящих по голосу и обозначь их S1, S2, S3 и так далее — только такими метками,
имена в поле speaker не подставляй.
В поле speakers дай по каждому краткую характеристику: пол, роль в разговоре и имя, если оно
звучит в записи (например: "мужской голос, ведёт обсуждение, собеседники называют Романом").
Время start и end указывай в секундах от начала этой записи."""


def audio_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def make_chunk(src: Path, dst: Path, start: float, length: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length),
         "-i", str(src), "-vn", "-ac", "1", "-b:a", BITRATE, str(dst)],
        check=True, timeout=1800,
    )


def plan_chunks(duration: float) -> list[tuple[float, float]]:
    """Returns [(start_sec, length_sec)] with overlap between neighbours."""
    if duration <= CHUNK_SEC:
        return [(0.0, duration)]
    step = CHUNK_SEC - OVERLAP_SEC
    out = []
    start = 0.0
    while start < duration:
        out.append((start, min(CHUNK_SEC, duration - start)))
        if start + CHUNK_SEC >= duration:
            break
        start += step
    return out


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def drop_overlap(prev_segments: list[dict], new_segments: list[dict]) -> list[dict]:
    """Remove the part of new_segments that repeats the tail of prev_segments.

    Timestamps from the model are unreliable, so the overlap is found by matching
    words rather than time.
    """
    if not prev_segments or not new_segments:
        return new_segments
    tail = _norm_words(" ".join(s["text"] for s in prev_segments[-12:]))[-120:]
    if not tail:
        return new_segments
    cut = 0
    best = 0.0
    for i in range(min(len(new_segments), 20)):
        head = _norm_words(" ".join(s["text"] for s in new_segments[: i + 1]))[:120]
        if not head:
            continue
        ratio = difflib.SequenceMatcher(None, tail, head).ratio()
        if ratio > best:
            best, cut = ratio, i + 1
    return new_segments[cut:] if best >= 0.30 else new_segments


async def call_gemini(
    client: httpx.AsyncClient,
    cfg,
    chunk_path: Path,
    prompt: str,
    attempts: int = 4,
) -> dict:
    audio = base64.b64encode(chunk_path.read_bytes()).decode()
    body = {
        "model": cfg.gemini_model,
        "max_tokens": 60000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "transcript", "strict": True, "schema": SCHEMA},
        },
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}},
        ]}],
    }
    if "openrouter" in cfg.base_url.lower():
        # роутер может отдать запрос провайдеру, который молча игнорирует схему —
        # без неё модель ломает JSON кавычками прямой речи
        body["provider"] = {"require_parameters": True}
    last = ""
    for attempt in range(attempts):
        try:
            r = await client.post(
                cfg.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
                json=body, timeout=1800,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                raise RuntimeError(last)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"])[:300])
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            last = str(e)[:300]
            if attempt == attempts - 1:
                raise RuntimeError(f"Gemini: {last}")
            await asyncio.sleep(2 ** attempt * 3)
    raise RuntimeError(f"Gemini: {last}")


def to_text(segments: list[dict]) -> str:
    """Merge consecutive turns of the same speaker into readable blocks."""
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


async def transcribe_one(
    audio_path: Path,
    out_dir: Path,
    stem: str,
    cfg,
    work_dir: Path,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    log: Callable[[str], None] = lambda s: None,
    suffix: str = "gemini",
) -> dict:
    t0 = time.perf_counter()
    duration = await asyncio.to_thread(audio_duration, audio_path)
    plan = plan_chunks(duration)
    if len(plan) > 1:
        log(f"{len(plan)} частей по {CHUNK_SEC / 60:.0f} мин "
            f"(перехлёст {OVERLAP_SEC:.0f} с) — нумерация говорящих между частями может разойтись")
    else:
        log(f"одним запросом, {duration / 60:.1f} мин")

    segments: list[dict] = []
    speakers: dict[str, str] = {}
    if on_progress:
        await on_progress(0, len(plan))

    async with httpx.AsyncClient() as client:
        for i, (start, length) in enumerate(plan):
            chunk = work_dir / f"g_{i:03d}.mp3"
            await asyncio.to_thread(make_chunk, audio_path, chunk, start, length)
            prompt = BASE_PROMPT
            if cfg.prompt:
                prompt += f"\nТермины и имена, которые могут встретиться: {cfg.prompt}"
            if speakers:
                known = "; ".join(f"{k} — {v}" for k, v in speakers.items())
                prompt += (
                    f"\nЭто продолжение записи. Первые {OVERLAP_SEC:.0f} секунд уже были расшифрованы ранее."
                    f" Ранее в записи говорили: {known}."
                    " Используй ТЕ ЖЕ метки для тех же голосов, новые метки заводи только для новых людей."
                )
            data = await call_gemini(client, cfg, chunk, prompt)
            new = [s for s in data.get("segments", []) if (s.get("text") or "").strip()]
            for s in new:
                s["start"] = float(s.get("start") or 0) + start
                s["end"] = float(s.get("end") or 0) + start
            if i > 0:
                before = len(new)
                new = drop_overlap(segments, new)
                log(f"часть {i + 1}: убрано {before - len(new)} повторов из перехлёста")
            for sp in data.get("speakers", []):
                speakers.setdefault(sp.get("id", "?"), sp.get("description", ""))
            segments.extend(new)
            chunk.unlink(missing_ok=True)
            log(f"часть {i + 1}/{len(plan)} готова, реплик: {len(new)}")
            if on_progress:
                await on_progress(i + 1, len(plan))

    text = to_text(segments)
    out_dir = out_dir / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    payload = {
        "mode": suffix,
        "model": cfg.gemini_model,
        "language": cfg.language,
        "timestamps": "приблизительные: модель оценивает время сама, к концу записи расхождение растёт",
        "speakers": [{"id": k, "description": v} for k, v in sorted(speakers.items())],
        "chunks": len(plan),
        "segments": segments,
        "text": text,
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "success": True,
        "mode": suffix,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "audio_duration_sec": round(duration, 2),
        "model": cfg.gemini_model,
        "chunks": len(plan),
        "segments": len(segments),
        "speakers": len(speakers),
        "source_audio": str(audio_path),
    }
    (out_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": meta, "segments": segments,
            "speakers": [{"id": k, "description": v} for k, v in sorted(speakers.items())]}
