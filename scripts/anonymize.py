#!/usr/bin/env python3
"""
anonymize.py — замена ВСЕХ имён собственных WoW на вымышленные (в англ. тексте).

Двухслойная замена по data/raw:
  1) курируемое ядро (build_terms_map.MAP) — красивые согласованные имена для
     ключевых сущностей (Thrall → Gorruk, Horde → Grennok, ...);
  2) авто-хвост — любое оставшееся имя собственное детектируется эвристикой
     «слово встречается в корпусе только с заглавной буквы = имя» и заменяется
     детерминированным «опаковым» токеном (одно слово → всегда один токен).

Так гарантируется, что в базе не останется узнаваемых терминов WoW. Итоговый
полный словарь замен пишется в terms_map.json (ядро + авто). Результат замены —
data/anon/<slug>.md (англ., с вымышленными именами); имя файла из заменённого «# …».

Замена делается на английском ДО перевода: нет склонений, а Google Translate потом
сам склоняет транслитерированные вымышленные имена по-русски.

Запуск:  python scripts/anonymize.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_terms_map import MAP as CURATED  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "raw"
OUT = ROOT / "data" / "anon"

# Заглавные слова, которые НЕ являются именами мира (служебные, частые начала
# предложений, родовые понятия). Дополняет эвристику «встречается в нижнем регистре».
COMMON = set("""
The A An He She It His Her Its They Them Their This That These Those There Then Thus
After Before During While When Where As At By For From In Into On Of To With And But Or
Not Now Later However Although Though Meanwhile Instead Despite Because Since Until Among
Between Both Many Most Some All One Two Three Four Five Six Seven First Second Third Fourth
Fifth Only Even Still Yet Such Rather Almost Soon Once Upon Following Prior Together
Originally Finally Eventually Ultimately Shortly Additionally Furthermore Moreover Hence
Therefore Also Being Having Knowing Seeking Striving Claiming Taking Ruling Returning
Composed Given Known Created Named Raised Led Driven Founded Convinced Inspired Becoming
Located Studying Uniting Heeding Nourished Cradled Committed Destroyed Trained Tutored
Seducing Please Much More Few Large Major Recent Early Proud Brave True Similar Individual
Eternal Grim Dire Supreme Celestial Furious Rampant Mortally Around Inside Alongside
Floating Ordering Consisting Devouring Leeching Focused Bound Blasted Fallen Remaining
Destiny History Rise Wrath Fury Reign Purge Invasion Rebellion Resistance Campaign Hunt
Trial Echo Veil Anvil Home Rest Heart Time Ages Decades Years Hour Volume Chronicle Parts
Heroes Church Army Court Guild Senate Wizard Magus Archmage Chieftain Shaman Warlock Rogue
Rider Hunters Watchers Horsemen Guardians Forces Clans Blades Skulls Bones Church Great
Dark Light Void Night Shadow Fire Ice Storm World Sea War Wars Battle Siege King Queen
Prince Princess Lord Lady Warchief Chief High Grand Order Knights Knight Council Prophet
Highlord Champion Guardian Betrayer Jailer Keeper Warden City Citadel Keep Tower Spire
Gate Gates Portal Throne Hall Temple Chapel Isle Isles Mount Valley Vale Well Font Chasm
Plateau Plane Realm Realms Kingdom Kingdoms Empire Nation Tribe Clan God Gods Titan Titans
Dragon Dragons Aspect Aspects Ancient Ancients Old New Iron Sun Moon Star Earth Wind Water
Flame Blood Death Life Soul Bell Sword Blade Crown Helm Shield Currently General Admiral
Master Sir Church Sisterhood Court Circle Guard Vanguard Offensive Army Host Legends Den
Rebellion Skyway Slums Embassy Longhouse Underhold Underbelly Cleft Den Strand Lake Grove
Ruins Caverns Deeps Glades Barrow Sanctum Cathedral Stair Jungle Grove Nexus Pillar Anvil
Green Red Curse Corrupted Broken Exiled Exiles Refuge Born Flesh Scourging Purge Fall
Destroyer Warder Ages Scaleborn Twilight Hammers Church Crane Chi August Wall Rise Given
Known Much Even Around Please
Ones Drag Tree Spring Thorns Valor Vigil Skull Horror Glacier Dawn Violet Tempest Peninsula
Walkers Ichor Archbishop Destructor Tranquil Desolate Domination Verdict Ashen Undead Undeath
Warrior Beyond Wisdom Strength Spirits Western Campaign Guard Lords Isles Underbelly Hope Hunt
""".split())

WORD_RE = re.compile(r"[A-Za-z]{2,}")
# имя собственное: Заглавная + строчная (исключает II, PvP, аббревиатуры), плюс
# возможные апострофные части (Mal'Ganis, Al'Akir); хвост 's отсекаем как притяжательный.
PN_RE = re.compile(r"[A-Z][a-z][A-Za-z]*(?:['’][A-Za-z]+)*")

# «опаковый» генератор токенов — детерминированный по слову, единая фэнтези-фонетика
ONSET = ["v", "z", "k", "g", "d", "r", "s", "t", "n", "m", "th", "dr", "kr", "gr",
         "vr", "zh", "x", "sh", "kh", "gh", "br", "tr"]
VOWEL = ["a", "e", "i", "o", "u", "ae", "y", "or", "ar", "el", "ir", "yn", "au"]
CODA = ["n", "r", "k", "l", "th", "x", "s", "d", "rn", "ng", "kar", "dar", "thar",
        "nel", "vor", "reth", "mar", "gath", "lun", "dris"]


def gen_token(word):
    h = hashlib.md5(word.lower().encode("utf-8")).digest()
    syl = 2 + (h[0] % 2)  # 2–3 слога
    out = ""
    for i in range(syl):
        out += ONSET[h[(2 * i + 1) % 16] % len(ONSET)]
        out += VOWEL[h[(2 * i + 2) % 16] % len(VOWEL)]
    out += CODA[h[5] % len(CODA)]
    return out.capitalize()


def build_skip():
    """Слова, которые не трогаем: встречающиеся в корпусе со строчной буквы +
    COMMON + слова наших курируемых токенов (чтобы авто-слой их не перезаписал)."""
    lower = set()
    for f in IN.glob("*.md"):
        for w in WORD_RE.findall(f.read_text(encoding="utf-8")):
            if w[0].islower():
                lower.add(w.lower())
    skip = {w.lower() for w in COMMON} | lower
    for v in CURATED.values():
        for w in re.findall(r"[A-Za-z']+", v):
            skip.add(w.lower())
    return skip


# заглавное слово в СЕРЕДИНЕ предложения (после строчной буквы/цифры/запятой + пробел) —
# признак имени собственного. Слова, встречающиеся только в начале предложения
# («Unlike», «Hunting»), сюда не попадают и авто-слоем не трогаются.
MID_RE = re.compile(r"(?<=[a-z0-9,;:)’'\"]\s)([A-Z][a-z][A-Za-z'’]*)")


def build_midcap():
    seen = set()
    for f in IN.glob("*.md"):
        for m in MID_RE.finditer(f.read_text(encoding="utf-8")):
            seen.add(m.group(1).lower())
    return seen


CURATED_PATTERNS = [
    (re.compile(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])"), v)
    for k, v in sorted(CURATED.items(), key=lambda kv: -len(kv[0]))
]


def slug(title):
    return re.sub(r"[^\w]+", "_", title.lower(), flags=re.UNICODE).strip("_")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skip = build_skip()
    midcap = build_midcap()
    auto = {}      # low -> токен (дедуп + подстановка)
    auto_key = {}  # low -> оригинал в исходном регистре (для словаря)

    def repl(m):
        w = m.group(0)
        mp = re.search(r"['’]s$", w)  # притяжательное 's отделяем
        core, suf = (w[:mp.start()], w[mp.start():]) if mp else (w, "")
        low = core.lower()
        if low in skip or low not in midcap:  # не имя собственное — не трогаем
            return w
        if low not in auto:
            tok = gen_token(core)
            # избегаем коллизии токена с уже занятым
            while tok.lower() in {t.lower() for t in auto.values()} | skip:
                tok = gen_token(core + "x")
                core = core + "x"
            auto[low] = tok
            auto_key[low] = core
        return auto[low] + suf

    total_curated = 0
    for f in sorted(IN.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for pat, val in CURATED_PATTERNS:
            text, c = pat.subn(val, text)
            total_curated += c
        text = PN_RE.sub(repl, text)
        first = text.splitlines()[0] if text else ""
        title = first[2:].strip() if first.startswith("# ") else f.stem
        (OUT / f"{slug(title)}.md").write_text(text, encoding="utf-8")

    # англ. карта замен (курируемое ядро + авто-хвост) — промежуточный результат;
    # русский terms_map.json собирает localize_terms.py.
    full = dict(CURATED)
    full.update({auto_key[k]: v for k, v in auto.items()})
    ordered = dict(sorted(full.items(), key=lambda kv: (-len(kv[0]), kv[0])))
    (ROOT / "data" / "terms_map.en.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"замен ядра: {total_curated} | авто-имён: {len(auto)} | "
          f"всего замен: {len(ordered)} → data/terms_map.en.json")
    print(f"файлов: {len(list(OUT.glob('*.md')))} → {OUT}")


if __name__ == "__main__":
    main()
