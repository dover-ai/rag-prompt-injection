#!/usr/bin/env python3
"""
localize_terms.py — русский словарь замен terms_map.json.

Подмена в пайплайне идёт на английском (data/terms_map.en.json: англ. термин →
вымышленный токен), а итоговая база — русская. Этот скрипт переводит обе стороны
на русский (официальная локализация WoW для оригиналов, транслитерация для
вымышленных токенов) и собирает terms_map.json как

    {"русский оригинал": "русское вымышленное"}      напр. "Тралл": "Горрук"

чтобы словарь читался в одном языке с базой знаний.

Запуск:  python scripts/localize_terms.py   (после anonymize.py)
Зависимости: только стандартная библиотека.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "terms_map.en.json"
OUT = ROOT / "terms_map.json"
GTX = "https://translate.googleapis.com/translate_a/single"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def gt(text):
    q = urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ru", "dt": "t", "q": text})
    req = urllib.request.Request(GTX + "?" + q, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            return "".join(s[0] for s in data[0] if s and s[0])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise RuntimeError("unreachable")


def translate_terms(terms):
    """Переводит список терминов батчами (через \\n) с проверкой выравнивания;
    при рассинхроне батча откатывается на построчный перевод."""
    result = {}
    i = 0
    while i < len(terms):
        chunk, size = [], 0
        while i < len(terms) and size < 1200:
            chunk.append(terms[i])
            size += len(terms[i]) + 1
            i += 1
        lines = gt("\n".join(chunk)).split("\n")
        if len(lines) == len(chunk):
            for en, ru in zip(chunk, lines):
                result[en] = ru.strip()
        else:
            for en in chunk:
                result[en] = gt(en).strip()
                time.sleep(0.2)
        time.sleep(0.3)
    return result


def main():
    if not SRC.exists():
        print("нет data/terms_map.en.json — сначала запустите anonymize.py")
        sys.exit(1)
    en_map = json.loads(SRC.read_text(encoding="utf-8"))
    ru_orig = translate_terms(list(en_map.keys()))
    ru_fake = translate_terms(list(dict.fromkeys(en_map.values())))

    ru_map = {}
    for en, tok in en_map.items():
        ru_map[ru_orig.get(en, en)] = ru_fake.get(tok, tok)
    ordered = dict(sorted(ru_map.items(), key=lambda kv: (-len(kv[0]), kv[0])))
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"terms_map.json (RU): {len(ordered)} записей → {OUT}")


if __name__ == "__main__":
    main()
