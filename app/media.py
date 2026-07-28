"""Media scanning and audio extraction (ffmpeg)."""
from __future__ import annotations

import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".ts", ".mpg", ".mpeg", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma"}


class NoAudioError(Exception):
    pass


SILENCE_DB = -60.0  # ниже этого порога в дорожке нет сигнала вообще


def scan_media(root: Path) -> list[dict]:
    """Recursively list media files under root. Paths are relative to root, posix-style.

    Our own output directory is skipped, so a folder used as both source and
    output does not feed extracted audio back in on the next run.
    """
    from .core import TRANSCRIPTS_DIR

    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if TRANSCRIPTS_DIR in p.relative_to(root).parts:
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            kind = "video"
        elif ext in AUDIO_EXTS:
            kind = "audio"
        else:
            continue
        files.append({
            "path": p.relative_to(root).as_posix(),
            "kind": kind,
            "size": p.stat().st_size,
        })
    return files


def assign_stems(rel_paths: list[str]) -> dict[str, str]:
    """Map rel path -> output stem. On stem collision within the same directory,
    fall back to the full file name (with extension) as stem."""
    by_key: dict[tuple[str, str], list[str]] = {}
    for rp in rel_paths:
        p = Path(rp)
        key = (p.parent.as_posix(), p.stem.lower())
        by_key.setdefault(key, []).append(rp)
    stems: dict[str, str] = {}
    for group in by_key.values():
        if len(group) == 1:
            stems[group[0]] = Path(group[0]).stem
        else:
            for rp in group:
                stems[rp] = Path(rp).name
    return stems


def has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode == 0 and proc.stdout.strip() != ""


def max_volume_db(path: Path) -> float:
    """Peak level of the track. Digital silence reports about -91 dB."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=1800,
    )
    for line in (proc.stderr or "").splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (IndexError, ValueError):
                break
    return 0.0


def ensure_audio(src: Path, out_dir: Path, stem: str, force: bool, log) -> Path:
    """Return an mp3 path for the pipeline. mp3 sources are used as is;
    everything else is extracted/converted to mono 64k mp3 cached in out_dir."""
    if src.suffix.lower() == ".mp3":
        _reject_if_silent(src, log)
        return src
    cached = out_dir / f"{stem}.audio.mp3"
    if cached.exists() and not force:
        log(f"звук уже извлечён: {cached.name}")
        _reject_if_silent(cached, log)
        return cached
    if not has_audio_stream(src):
        raise NoAudioError("файл не содержит аудиодорожки")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp.mp3")
    log(f"извлекаю звук ffmpeg → mono mp3 64k")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-b:a", "64k", str(tmp)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise RuntimeError(f"ffmpeg: {tail[0]}")
    tmp.replace(cached)
    _reject_if_silent(cached, log)
    return cached


def _reject_if_silent(path: Path, log) -> None:
    """Guard against paying to transcribe silence: Whisper hallucinates on it."""
    peak = max_volume_db(path)
    if peak <= SILENCE_DB:
        raise NoAudioError(f"в дорожке нет звука (пиковый уровень {peak:.0f} дБ)")
    log(f"уровень звука: пик {peak:.0f} дБ")
