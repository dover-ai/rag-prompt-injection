#!/usr/bin/env python3
"""
translate.py — перевод анонимизированных англ. документов (data/anon) на русский.

Использует бесплатный публичный эндпоинт Google Translate (ключ не нужен).
Переводит поабзацно (длинные абзацы дробит по предложениям, чтобы не упереться
в лимит запроса). Заголовок '# ...' переводится как заголовок. Вымышленные имена
транслитерируются и корректно склоняются переводчиком.
Результат — knowledge_base/<slug>.md (финальная русская база знаний).

Идемпотентно: уже переведённые файлы пропускаются.
Зависимости: только стандартная библиотека.

Запуск:
  python scripts/translate.py           # перевести все непереведённые
  python scripts/translate.py gorruk     # только один файл (для проверки, с перезаписью)
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "anon"          # анонимизированный англ. текст (после anonymize.py)
OUT = ROOT / "knowledge_base"        # финальная русская база знаний
GTX = "https://translate.googleapis.com/translate_a/single"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
MAX_CHARS = 1400  # безопасный размер одного запроса


def gt(text):
    """Перевод одного блока (< MAX_CHARS) EN->RU через публичный эндпоинт."""
    q = urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ru", "dt": "t", "q": text})
    req = urllib.request.Request(GTX + "?" + q, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            return "".join(seg[0] for seg in data[0] if seg and seg[0])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise RuntimeError("unreachable")


def translate_block(block):
    block = block.strip()
    if len(block) <= MAX_CHARS:
        return gt(block)
    groups, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", block):
        if cur and len(cur) + len(s) + 1 > MAX_CHARS:
            groups.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        groups.append(cur)
    return " ".join(gt(g) for g in groups)


def translate_doc(text):
    lines = text.split("\n")
    title = ""
    if lines and lines[0].startswith("# "):
        title = "# " + gt(lines[0][2:].strip())
        body = "\n".join(lines[1:]).strip()
    else:
        body = text
    paras = [translate_block(p) for p in body.split("\n\n") if p.strip()]
    body_ru = "\n\n".join(paras)
    return (title + "\n\n" + body_ru).strip() if title else body_ru


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    allfiles = sorted(IN.glob("*.md"))
    if only:
        files = [f for f in allfiles if f.stem == only]
        if not files:
            print(f"нет файла data/raw/{only}.md")
            sys.exit(1)
    else:
        files = [f for f in allfiles if not (OUT / f.name).exists()]

    print(f"к переводу: {len(files)}")
    done = 0
    for f in files:
        try:
            ru = translate_doc(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ERROR  {f.stem}: {type(e).__name__} {e}")
            continue
        (OUT / f.name).write_text(ru.rstrip() + "\n", encoding="utf-8")
        done += 1
        print(f"  ok  {f.stem}  ({len(ru)} симв.)")
        time.sleep(0.4)
    print(f"\nпереведено: {done} | всего в {OUT.name}: {len(list(OUT.glob('*.md')))}")


if __name__ == "__main__":
    main()
