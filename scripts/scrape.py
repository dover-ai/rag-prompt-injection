#!/usr/bin/env python3
"""
scrape.py — выгрузка исходных статей вики (warcraft.wiki.gg) в чистый текст.

Тянет статьи по списку ENTITIES через MediaWiki API (prop=extracts, explaintext),
оставляет интро + «лорные» секции (описание, культура, общество…), выкидывает
поэкспансионную историю, боевые механики и предложения с игровыми/мета-маркерами
(Warcraft, expansion, patch, playable race…), обрезает по объёму и пишет
data/raw/<slug>.md (один файл — одна сущность).

data/raw/ — промежуточный результат, в git не хранится (см. .gitignore).
Финальная переименованная база собирается скриптом anonymize.py в knowledge_base/.

Запуск:  python scripts/scrape.py
Зависимости: только стандартная библиотека.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

API = "https://warcraft.wiki.gg/api.php"
UA = "YandexTestTaskBot/1.0 (educational RAG project)"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
CAP = 3000  # целевой максимум символов на документ

ENTITIES = [
    # персонажи
    "Thrall", "Sylvanas Windrunner", "Arthas Menethil", "Jaina Proudmoore",
    "Illidan Stormrage", "Malfurion Stormrage", "Tyrande Whisperwind",
    "Anduin Wrynn", "Grommash Hellscream", "Gul'dan", "Tirion Fordring", "Deathwing",
    # расы
    "Orc", "Human", "Night elf", "Forsaken", "Tauren", "Darkspear tribe",
    "Blood elf", "Draenei",
    # фракции
    "Horde", "Alliance", "Scourge", "Burning Legion", "Kirin Tor", "Argent Crusade",
    # локации
    "Azeroth", "Orgrimmar", "Stormwind City", "Icecrown Citadel", "Dalaran", "Quel'Thalas",
    # артефакты / магия
    "Frostmourne", "Ashbringer", "Sunwell", "Dark Portal",
    # события
    "Third War", "War of the Ancients",
]

# «Лорные» секции — вне игровой механики и хронологии по экспансиям.
TIMELESS = ("description", "overview", "personality", "biology", "appearance",
            "characteristics", "culture", "society", "government", "geography",
            "religion", "traits", "characterization")

# Маркеры мета/игровых предложений — такие предложения выкидываем целиком.
DROP_KW = ("warcraft", "blizzard", "expansion", "patch", "retcon", "tcg", "rpg",
           "novel", "comic", "manga", "video game", "gameplay", "questline",
           "playable", "allied race", "developer", "cinematic", "voiced", "in-game",
           "trading card", "roleplaying", "short story", "audio drama", "novella",
           "the game", "this article", "pronounced", "wowpedia", "wiki")

HEADER_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def fetch(title):
    q = {"action": "query", "format": "json", "prop": "extracts",
         "explaintext": "1", "redirects": "1", "titles": title}
    url = API + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def parse(extract):
    """(intro, groups): intro — до первого заголовка; groups — верхнеуровневые
    секции с влитыми подсекциями (строки заголовков подсекций отброшены)."""
    intro, groups = [], []
    cur_title, cur_lines, started = None, [], False
    for ln in extract.split("\n"):
        m = HEADER_RE.match(ln.strip())
        if m:
            started = True
            if len(m.group(1)) == 2:
                if cur_title is not None:
                    groups.append((cur_title, "\n".join(cur_lines).strip()))
                cur_title, cur_lines = m.group(2), []
        elif not started:
            intro.append(ln)
        elif cur_title is not None:
            cur_lines.append(ln)
    if cur_title is not None:
        groups.append((cur_title, "\n".join(cur_lines).strip()))
    return "\n".join(intro).strip(), groups


def meta_filter(text):
    """Выкидывает предложения с игровыми/мета-маркерами, сохраняя абзацы."""
    out_paras = []
    for para in text.split("\n\n"):
        sents = re.split(r"(?<=[.!?])\s+", para.strip())
        kept = [s for s in sents if s and not any(k in s.lower() for k in DROP_KW)]
        if kept:
            out_paras.append(" ".join(kept))
    return "\n\n".join(out_paras)


def clean(extract):
    intro, groups = parse(extract)
    parts, total = [intro], len(intro)
    for title2, prose in groups:
        if total >= CAP:
            break
        if prose and any(k in title2.lower() for k in TIMELESS):
            parts.append(prose)
            total += len(prose)
    out = meta_filter("\n\n".join(p for p in parts if p))
    if len(out) < 700:  # добираем историей, если совсем коротко
        for title2, prose in groups:
            if prose and any(k in title2.lower() for k in ("history", "biography", "origin")):
                out += "\n\n" + meta_filter(prose)
                if len(out) >= CAP:
                    break
    out = re.sub(r"\s*\(/[^)]*\)", "", out)   # убрать IPA-транскрипции вида (/θrɔːl/ THRAWL)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if len(out) > CAP:
        cut = out.rfind("\n\n", 0, CAP)
        if cut < CAP * 0.5:
            dot = out.rfind(". ", 0, CAP)
            cut = dot + 1 if dot > 0 else CAP
        out = out[:cut].strip()
    return out


def slug(title):
    return re.sub(r"[^\w]+", "_", title.lower(), flags=re.UNICODE).strip("_")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for t in ENTITIES:
        try:
            rtitle, ext = fetch(t)
            body = clean(ext)
        except Exception as e:
            print(f"  ERROR  {t}: {type(e).__name__} {e}")
            continue
        (OUT / f"{slug(t)}.md").write_text(f"# {rtitle}\n\n{body}\n", encoding="utf-8")
        report.append((t, len(body)))
        print(f"{len(body):6d}  {t}")
        time.sleep(0.3)
    print(f"\n{len(report)} документов записано в {OUT}")
    short = [t for t, n in report if n < 400]
    if short:
        print("SHORT (<400) — проверить:", short)


if __name__ == "__main__":
    main()
