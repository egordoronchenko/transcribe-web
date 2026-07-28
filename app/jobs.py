"""Single-job manager: runs one transcription job in the background,
survives page reloads, streams state and log to SSE subscribers."""
from __future__ import annotations

import asyncio
import datetime
import json
import tempfile
import time
from collections import deque
from pathlib import Path

from . import gemini, media, merge
from .core import Config, TRANSCRIPTS_DIR, transcribe_one

HOST_ROOT = Path("/host/root")

MODES = ("whisper", "gemini", "both", "merged")

# какие файлы должны быть на месте, чтобы считать файл уже обработанным (resume);
# каждый режим кладёт результат в свою подпапку
MODE_FILES = {
    "whisper": (("whisper", ".txt"), ("whisper", ".srt"), ("whisper", ".json")),
    "gemini": (("gemini", ".txt"), ("gemini", ".json")),
    "both": (("whisper", ".txt"), ("whisper", ".srt"), ("whisper", ".json"),
             ("gemini", ".txt"), ("gemini", ".json")),
    "merged": (("merged", ".txt"), ("merged", ".srt"), ("merged", ".json")),
}

MODE_LABELS = {
    "whisper": "Whisper (дословно)",
    "gemini": "Gemini (с говорящими)",
    "both": "оба режима",
    "merged": "слияние (тайминги Whisper + говорящие Gemini)",
}


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class JobManager:
    def __init__(self):
        self.state: dict | None = None
        self.task: asyncio.Task | None = None
        self.log_lines: deque[str] = deque(maxlen=500)
        self.subscribers: list[asyncio.Queue] = []
        self.cancel_soft = False
        self.cancel_hard_event: asyncio.Event | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def log(self, line: str):
        stamp = time.strftime("%H:%M:%S")
        full = f"{stamp} {line}"
        self.log_lines.append(full)
        self._broadcast({"type": "log", "line": full})

    def _broadcast(self, event: dict):
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def push_state(self):
        self._broadcast({"type": "state", "state": self.state})

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.subscribers.append(q)
        try:
            yield {"type": "state", "state": self.state}
            for line in list(self.log_lines):
                yield {"type": "log", "line": line}
            while True:
                event = await q.get()
                yield event
        finally:
            self.subscribers.remove(q)

    def start(self, source_rel: str, output_rel: str, files: list[str], force: bool,
              prompt: str | None = None, mode: str = "whisper"):
        if self.running:
            raise RuntimeError("job already running")
        stems = media.assign_stems(files)
        self.state = {
            "status": "running",
            "source": source_rel,
            "output": output_rel,
            "force": force,
            "mode": mode if mode in MODES else "whisper",
            "prompt_override": (prompt or "").strip() or None,
            "started_at": now_iso(),
            "finished_at": None,
            "current_index": None,
            "files": [
                {"path": rp, "stem": stems[rp], "status": "queued",
                 "chunks_done": 0, "chunks_total": 0,
                 "wall_clock_sec": None, "audio_duration_sec": None, "error": None}
                for rp in files
            ],
        }
        self.log_lines.clear()
        self.cancel_soft = False
        self.cancel_hard_event = asyncio.Event()
        self.task = asyncio.create_task(self._run())
        self.push_state()

    def cancel(self, hard: bool):
        if not self.running:
            return
        self.cancel_soft = True
        self.state["status"] = "cancelling"
        self.log("отмена: " + ("прерываю текущий файл" if hard else "дообработаю текущий файл"))
        if hard and self.cancel_hard_event:
            self.cancel_hard_event.set()
        self.push_state()

    async def _run(self):
        try:
            await self._run_inner()
        except Exception as e:
            self.state["status"] = "failed"
            self.state["finished_at"] = now_iso()
            self.log(f"внутренняя ошибка задания: {e!r}")
            self.push_state()

    async def _run_mode(self, mode, audio_path, out_dir, stem, cfg, work_dir, on_progress) -> dict:
        """Runs the backend(s) the mode needs and returns the meta to report."""
        if mode in ("whisper", "both", "merged"):
            if mode != "whisper":
                self.log("шаг 1: Whisper — дословный текст и точные тайминги")
            w = await transcribe_one(audio_path, out_dir, stem, cfg, work_dir,
                                     on_progress=on_progress, log=self.log)
        if mode in ("gemini", "both", "merged"):
            if mode != "gemini":
                self.log("шаг 2: Gemini — разделение говорящих")
            g = await gemini.transcribe_one(audio_path, out_dir, stem, cfg, work_dir,
                                            on_progress=on_progress, log=self.log)
        if mode == "whisper":
            return w["meta"]
        if mode == "gemini":
            return g["meta"]
        if mode == "both":
            return {**w["meta"], "mode": "both",
                    "wall_clock_sec": round(w["meta"]["wall_clock_sec"] + g["meta"]["wall_clock_sec"], 2)}
        self.log("шаг 3: свожу тайминги и говорящих по совпадению текста")
        m = merge.write_merged(out_dir, stem, w, g, cfg, log=self.log)
        return {**m["meta"],
                "wall_clock_sec": round(w["meta"]["wall_clock_sec"]
                                        + g["meta"]["wall_clock_sec"]
                                        + m["meta"]["wall_clock_sec"], 2)}

    async def _run_inner(self):
        st = self.state
        cfg = Config.from_env()
        if st.get("prompt_override") is not None and st["prompt_override"] != (cfg.prompt or ""):
            cfg.prompt = st["prompt_override"]
            self.log("подсказка распознавания изменена пользователем для этого задания")
        source = HOST_ROOT / st["source"]
        output = HOST_ROOT / st["output"]
        mode = st.get("mode", "whisper")
        result_exts = MODE_FILES[mode]
        models = {"whisper": cfg.model, "gemini": cfg.gemini_model}.get(
            mode, f"{cfg.model} + {cfg.gemini_model}")
        self.log(f"задание: {len(st['files'])} файлов, режим {MODE_LABELS[mode]}, "
                 f"модель {models}, язык {cfg.language}")

        for i, f in enumerate(st["files"]):
            if self.cancel_soft:
                f["status"] = "cancelled"
                continue
            st["current_index"] = i
            src = source / f["path"]
            rel_dir = Path(f["path"]).parent
            stem = f["stem"]
            out_dir = output / TRANSCRIPTS_DIR / rel_dir / stem
            t0 = time.perf_counter()

            existing = [out_dir / sub / f"{stem}{ext}" for sub, ext in result_exts]
            if all(p.exists() for p in existing) and not st["force"]:
                f["status"] = "skip"
                self.log(f"skip (готово): {f['path']}")
                self.push_state()
                continue

            try:
                f["status"] = "extracting"
                self.push_state()
                self.log(f"файл {i + 1}/{len(st['files'])}: {f['path']}")
                audio_path = await asyncio.to_thread(
                    media.ensure_audio, src, out_dir, stem, st["force"], self.log
                )

                f["status"] = "transcribing"
                self.push_state()

                async def on_progress(done: int, total: int):
                    f["chunks_done"] = done
                    f["chunks_total"] = total
                    self.push_state()

                with tempfile.TemporaryDirectory(prefix="chunks_") as tmp:
                    work = asyncio.create_task(self._run_mode(
                        mode, audio_path, out_dir, stem, cfg, Path(tmp), on_progress))
                    hard_cancel = asyncio.create_task(self.cancel_hard_event.wait())
                    done_set, _ = await asyncio.wait(
                        {work, hard_cancel}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if hard_cancel in done_set and work not in done_set:
                        work.cancel()
                        try:
                            await work
                        except asyncio.CancelledError:
                            pass
                        f["status"] = "cancelled"
                        self.log(f"прервано: {f['path']}")
                        continue
                    hard_cancel.cancel()
                    meta = work.result()

                f["status"] = "ok"
                f["wall_clock_sec"] = meta["wall_clock_sec"]
                f["audio_duration_sec"] = meta["audio_duration_sec"]
                self.log(f"готово: {f['path']} за {meta['wall_clock_sec']:.0f} с")
            except media.NoAudioError as e:
                f["status"] = "no_audio"
                f["error"] = str(e)
                self.log(f"нет аудио: {f['path']}")
            except Exception as e:
                f["status"] = "error"
                f["error"] = str(e)[:500]
                f["wall_clock_sec"] = round(time.perf_counter() - t0, 2)
                self.log(f"ошибка: {f['path']} — {e}")
            self.push_state()

        st["current_index"] = None
        st["finished_at"] = now_iso()
        counts = {}
        for f in st["files"]:
            counts[f["status"]] = counts.get(f["status"], 0) + 1
        if counts.get("cancelled"):
            st["status"] = "cancelled"
        elif counts.get("error") or counts.get("no_audio"):
            st["status"] = "partial" if counts.get("ok") or counts.get("skip") else "failed"
        else:
            st["status"] = "done"
        self._write_report(cfg, output / TRANSCRIPTS_DIR, counts)
        self.log(f"итог: {json.dumps(counts, ensure_ascii=False)} → статус {st['status']}")
        self.push_state()

    def _write_report(self, cfg: Config, output: Path, counts: dict):
        st = self.state
        report = {
            "started_at": st["started_at"],
            "finished_at": st["finished_at"],
            "status": st["status"],
            "source": f"/host/root/{st['source']}",
            "output": f"/host/root/{st['output']}/{TRANSCRIPTS_DIR}",
            "mode": st.get("mode", "whisper"),
            "model": cfg.gemini_model if st.get("mode") == "gemini" else cfg.model,
            "language": cfg.language,
            "force": st["force"],
            "files": [
                {k: f[k] for k in ("path", "stem", "status", "wall_clock_sec", "audio_duration_sec", "error")}
                for f in st["files"]
            ],
            "summary": {
                **counts,
                "total_audio_sec": round(sum(f["audio_duration_sec"] or 0 for f in st["files"]), 1),
                "total_wall_clock_sec": round(sum(f["wall_clock_sec"] or 0 for f in st["files"]), 1),
            },
        }
        try:
            output.mkdir(parents=True, exist_ok=True)
            (output / "job-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            self.log(f"не удалось записать job-report.json: {e}")


manager = JobManager()
