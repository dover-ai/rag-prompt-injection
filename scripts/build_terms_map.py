#!/usr/bin/env python3
"""
build_terms_map.py — словарь замен «WoW-термин → вымышленный токен» (terms_map.json).

Замена выполняется на АНГЛИЙСКОМ тексте (data/raw) до перевода на русский:
в английском нет склонений, поэтому подстановка тривиальна и надёжна, а Google
Translate затем сам склоняет транслитерированные вымышленные имена по-русски.

Значения — «опаковые» токены (не английские слова), чтобы переводчик их
ТРАНСЛИТЕРИРОВАЛ (Grennok → Греннок), а не переводил по смыслу. Имена подобраны в
единой фэнтезийной фонетике; составные имена собираются из атомов
(Sylvanas + Windrunner → Veyra + Sael), фамилии единообразны для родственников.

Запуск:  python scripts/build_terms_map.py   →  пишет terms_map.json
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# --- Персонажи: имена -------------------------------------------------------
NAMES = {
    "Thrall": "Gorruk", "Sylvanas": "Veyra", "Arthas": "Kaelric",
    "Jaina": "Elyndra", "Illidan": "Xaltir", "Malfurion": "Faeron",
    "Tyrande": "Aelira", "Anduin": "Corin", "Grommash": "Gormash", "Grom": "Gorm",
    "Gul'dan": "Volgath", "Tirion": "Dorian", "Deathwing": "Direxar",
    "Garrosh": "Garruk", "Vol'jin": "Zaljin", "Kel'Thuzad": "Vextharun",
    "Kil'jaeden": "Xargathel", "Archimonde": "Arganoth", "Mannoroth": "Vharnoth",
    "Sargeras": "Zarganeth", "Medivh": "Kaervin", "Cenarius": "Sylthanos",
    "Velen": "Vaelen", "Kael'thas": "Kaelith", "Lor'themar": "Lorvaeth",
    "Bolvar": "Bolren", "Uther": "Uthric", "Darion": "Daeric",
    "Alexandros": "Alaxis", "Ner'zhul": "Vorzhul", "Terenas": "Tereth",
    "Magni": "Modrin", "Orgrim": "Orgath", "Zovaal": "Zovax",
    "Xal'atath": "Xelaroth", "M'uru": "Morath", "Akama": "Akar",
    "Korialstrasz": "Korovax", "Fyrakk": "Pyrrak", "Azshara": "Azsheia",
    "Shandris": "Shaelis", "Dath'Remar": "Daermah", "Varok": "Vorlan",
    "Aggra": "Ashra", "Durotan": "Darogan", "Draka": "Draya",
}

# --- Персонажи: фамилии (единообразно для родственников) ---------------------
SURNAMES = {
    "Windrunner": "Sael", "Menethil": "Vothmyr", "Proudmoore": "Tidan",
    "Stormrage": "Vaelthorn", "Whisperwind": "Silwen", "Wrynn": "Aldan",
    "Hellscream": "Grakmar", "Fordring": "Velmont", "Mograine": "Kravon",
    "Sunstrider": "Solthar", "Feathermoon": "Lunareth", "Theron": "Vaeloth",
    "Fordragon": "Drexen", "Bronzebeard": "Durnmar", "Saurfang": "Grimtak",
    "Doomhammer": "Vranmar", "Lightbringer": "Radan",
}

# --- Расы / народы (+ множественные и прилагательные формы) ------------------
RACES = {
    "Orc": "Grokkar", "Orcs": "Grokkar", "orcish": "grokkar", "Orcish": "Grokkar",
    "Human": "Auren", "Humans": "Auren",
    "Night elf": "Lunthari", "Night elves": "Lunthari",
    "night elf": "lunthari", "night elves": "lunthari",
    "Blood elf": "Velthari", "Blood elves": "Velthari",
    "blood elf": "velthari", "blood elves": "velthari",
    "Forsaken": "Nethari", "Tauren": "Kaeloth", "Draenei": "Aetheri",
    "Darkspear": "Duskaari", "Darkspears": "Duskaari",
    "Highborne": "Solvar", "Amani": "Vozari", "Gurubashi": "Xulgar",
    "Val'kyr": "Vaelkar", "Illidari": "Xaltiri", "Sunreavers": "Solthari",
    "Sentinels": "Sylvani", "Thalassian": "Veltharic", "Darnassian": "Lunaric",
}

# --- Фракции / организации --------------------------------------------------
FACTIONS = {
    "Horde": "Grennok", "Alliance": "Valdor", "Scourge": "Skorn",
    "Legion": "Gharn", "Kirin Tor": "Kyrnthal", "Argent": "Silvaris",
    "Scarlet": "Crimvar", "Silver Hand": "Paldrin", "Ebon Blade": "Vexhal",
    "Cenarion": "Sylthic", "Earthen Ring": "Loamthar", "Dragon Aspects": "Varnyx",
    "Aspects": "Varnyx", "Old Gods": "Zorncil", "Pantheon": "Kaelthorn",
    "Shattered Sun": "Solvaris",
}

# --- Локации ----------------------------------------------------------------
PLACES = {
    "Azeroth": "Aureth", "Kalimdor": "Marendor", "Northrend": "Norvath",
    "Lordaeron": "Veldoran", "Outland": "Xordreth", "Draenor": "Korrath",
    "Eastern Kingdoms": "Veresmark", "Stormwind": "Korveth", "Orgrimmar": "Grakmoth",
    "Icecrown": "Vexmor", "Dalaran": "Azuras", "Quel'Thalas": "Sylthara",
    "Quel'Danas": "Syldana", "Silvermoon": "Sildath", "Undercity": "Nethmere",
    "Durotar": "Dargan", "Theramore": "Veymar", "Teldrassil": "Sylthil",
    "Plaguelands": "Skornmark", "Broken Shore": "Drathmar", "Argus": "Xargus",
    "Hyjal": "Vaelor", "Well of Eternity": "Ayluneth", "Emerald Dream": "Virelith",
    "Ashenvale": "Kelvane", "Mulgore": "Marloth", "Gilneas": "Velmoor",
    "Kul Tiras": "Vardis", "Ironforge": "Durnhold", "Khaz Modan": "Durazmar",
    "Strom": "Voreth", "Deepholm": "Nadreth", "Maelstrom": "Zundmar",
    "Amirdrassil": "Sylmara", "Nordrassil": "Eldrath", "Ragefire": "Karndeep",
    "Hearthglen": "Velmarch", "Nagrand": "Narangal", "Pandaria": "Xanthil",
    "Dorn": "Vorn", "Sunwell Plateau": "Solnar Plateau",
}

# --- Артефакты / магия / концепции ------------------------------------------
ARTIFACTS = {
    "Frostmourne": "Kryndel", "Ashbringer": "Solbrand", "Sunwell": "Solnar",
    "Dark Portal": "Nexumar", "Frozen Throne": "Nyxthar", "Dragon Soul": "Draknyr",
    "Darkwell": "Nethvar", "Holy Light": "Aelmyr", "Elune": "Aluna",
    "Earth Mother": "Terramyr", "Earthmother": "Terramyr", "Lich King": "Nyxaris",
    "Fel": "Nyth", "the Light": "the Aelmyr",
}

# --- События ----------------------------------------------------------------
EVENTS = {
    "War of the Ancients": "War of the Eldar", "Ancients": "Eldar",
    "Cataclysm": "Vharnfall", "Sundering": "Vharn", "Troll Wars": "Vozari Wars",
}

# --- Прочее (доводка узнаваемых остатков) ------------------------------------
MISC = {
    "Burning Crusade": "Nythrift", "Light's Hope": "Dawnhold", "Banshee": "Nethra",
}

MAP = {}
for d in (NAMES, SURNAMES, RACES, FACTIONS, PLACES, ARTIFACTS, EVENTS, MISC):
    MAP.update(d)


def main():
    # MAP импортируется anonymize.py как «курируемое ядро» (красивые имена для
    # ключевых сущностей). Итоговый terms_map.json (ядро + автозамены хвоста)
    # собирает anonymize.py.
    print(f"курируемое ядро словаря: {len(MAP)} записей")


if __name__ == "__main__":
    main()
