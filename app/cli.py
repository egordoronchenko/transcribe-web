"""CLI mode (bonus): transcribe a folder without the web UI.

Usage inside the container:
  python -m app.cli --in /host/root/path/to/videos --out /host/root/path/to/out [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from . import media
from .core import Config, TRANSCRIPTS_DIR, transcribe_one


def log(line: str):
    print(line, flush=True)


async def run(src: Path, out: Path, force: bool) -> int:
    cfg = Config.from_env()
    if not cfg.api_key:
        print("OPENROUTER_API_KEY не задан", file=sys.stderr)
        return 2
    files = media.scan_media(src)
    if not files:
        print("медиафайлы не найдены", file=sys.stderr)
        return 2
    stems = media.assign_stems([f["path"] for f in files])
    failed = 0
    for i, f in enumerate(files):
        rel = f["path"]
        stem = stems[rel]
        out_dir = out / TRANSCRIPTS_DIR / Path(rel).parent / stem
        if all((out_dir / f"{stem}{ext}").exists() for ext in (".txt", ".srt", ".json")) and not force:
            log(f"[{i + 1}/{len(files)}] skip: {rel}")
            continue
        log(f"[{i + 1}/{len(files)}] {rel}")
        try:
            audio = media.ensure_audio(src / rel, out_dir, stem, force, log)
            with tempfile.TemporaryDirectory(prefix="chunks_") as tmp:
                meta = await transcribe_one(audio, out_dir, stem, cfg, Path(tmp), log=log)
            log(f"  готово за {meta['wall_clock_sec']:.0f} с")
        except media.NoAudioError:
            log("  нет аудиодорожки — пропускаю")
        except Exception as e:
            failed += 1
            log(f"  ошибка: {e}")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description="transcribe-web CLI")
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    sys.exit(asyncio.run(run(Path(args.src), Path(args.out), args.force)))


if __name__ == "__main__":
    main()
