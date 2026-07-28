"""Summaries over finished transcripts.

The summary is a separate derived document: the transcript itself is never
touched or rewritten, so a model's liberties in the summary can always be
checked against the original text lying next to it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

TEMPLATES = {
    "lecture": {
        "title": "Лекция",
        "prompt": """Это расшифровка лекции. Составь конспект в формате Markdown:

## О чём лекция
Два-три предложения по сути.

## Структура
Список разделов в порядке изложения, по одной строке на тему.

## Ключевые тезисы
Главные утверждения и выводы, маркированным списком.

## Термины
Термины, определения и названия инструментов, которые встретились. Формат: **термин** — пояснение.

## На что обратить внимание
Практические советы, предостережения, типичные ошибки, о которых говорил лектор.

Пиши по-русски. Не выдумывай того, чего нет в расшифровке. Термины сохраняй в точности как в тексте.""",
    },
    "meeting": {
        "title": "Совещание",
        "prompt": """Это расшифровка рабочего совещания. Реплики помечены говорящими (S1, S2, ...).
Составь протокол в формате Markdown:

## Участники
По каждой метке — что за роль у человека, судя по разговору.

## Обсуждаемые вопросы
Список тем, которые поднимались.

## Принятые решения
Что именно решили. Указывай, кто предложил и кто согласился.

## Задачи
Таблица: задача | кто делает | срок. Только то, что реально прозвучало.

## Разногласия и открытые вопросы
Где мнения разошлись и что осталось нерешённым.

Пиши по-русски. Ничего не додумывай: если срок или ответственный не назывались, так и пиши «не назван».""",
    },
    "call": {
        "title": "Созвон с заказчиком",
        "prompt": """Это расшифровка созвона с заказчиком. Реплики помечены говорящими (S1, S2, ...).
Составь резюме в формате Markdown:

## Суть разговора
Два-три предложения.

## Требования и пожелания заказчика
Что просили, с деталями.

## Возражения, риски, ограничения
Что вызывало сомнения у сторон.

## Договорённости
О чём условились, что кому обещали.

## Следующие шаги
Что делать дальше и к какому сроку.

Пиши по-русски. Ничего не додумывай: если о чём-то не договорились явно, отметь это.""",
    },
}

DEFAULT_TEMPLATE = "meeting"
SOURCE_PRIORITY = ("merged", "gemini", "whisper")


def pick_source(file_dir: Path, stem: str) -> Path | None:
    """Prefer the richest transcript available: merged, then gemini, then whisper."""
    for mode in SOURCE_PRIORITY:
        p = file_dir / mode / f"{stem}.txt"
        if p.is_file():
            return p
    return None


def scan_transcripts(root: Path) -> list[dict]:
    """List finished transcripts under <root>/transcripts, ready to summarize."""
    from .core import TRANSCRIPTS_DIR

    base = root / TRANSCRIPTS_DIR
    if not base.is_dir():
        return []
    entries: dict[str, dict] = {}
    for mode in SOURCE_PRIORITY:
        for txt in sorted(base.rglob(f"{mode}/*.txt")):
            file_dir = txt.parent.parent
            try:
                key = file_dir.relative_to(base).as_posix()
            except ValueError:
                continue
            e = entries.setdefault(key, {"path": key, "stem": txt.stem, "modes": [], "has_summary": False})
            if mode not in e["modes"]:
                e["modes"].append(mode)
    for key, e in entries.items():
        e["has_summary"] = (base / key / "summary" / f"{e['stem']}.md").is_file()
    return [entries[k] for k in sorted(entries)]


async def summarize(text: str, prompt: str, cfg, log=lambda s: None) -> tuple[str, dict]:
    body = {
        "model": cfg.summary_model,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": "Ты аккуратный аналитик. Работаешь только с тем, что есть в тексте."},
            {"role": "user", "content": f"{prompt}\n\n---\nРАСШИФРОВКА:\n\n{text}"},
        ],
    }
    async with httpx.AsyncClient() as client:
        last = ""
        for attempt in range(4):
            try:
                r = await client.post(
                    cfg.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
                    json=body, timeout=1200,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    raise RuntimeError(last)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise RuntimeError(str(data["error"])[:300])
                return data["choices"][0]["message"]["content"], (data.get("usage") or {})
            except Exception as e:
                last = str(e)[:300]
                if attempt == 3:
                    raise RuntimeError(f"саммари: {last}")
                import asyncio
                await asyncio.sleep(2 ** attempt * 3)
    raise RuntimeError(f"саммари: {last}")


async def summarize_file(
    file_dir: Path,
    stem: str,
    template_key: str,
    prompt: str,
    cfg,
    log=lambda s: None,
) -> dict:
    src = pick_source(file_dir, stem)
    if src is None:
        raise RuntimeError("нет расшифровки для суммаризации")
    text = src.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("расшифровка пустая")
    t0 = time.perf_counter()
    log(f"саммари по {src.parent.name}: {len(text)} символов, шаблон «{template_key}»")
    md, usage = await summarize(text, prompt, cfg, log)

    out_dir = file_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (f"<!-- саммари: модель {cfg.summary_model}, шаблон {template_key}, "
              f"источник {src.parent.name}/{src.name} -->\n\n")
    (out_dir / f"{stem}.md").write_text(header + md.strip() + "\n", encoding="utf-8")
    meta = {
        "success": True,
        "model": cfg.summary_model,
        "template": template_key,
        "source": f"{src.parent.name}/{src.name}",
        "source_chars": len(text),
        "summary_chars": len(md),
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "usage": usage,
    }
    (out_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"саммари готово: {len(md)} символов за {meta['wall_clock_sec']:.0f} с")
    return meta
