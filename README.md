# YandexTestTask_Bot — RAG-бот по корпоративной базе знаний

Учебный проект: интеллектуальный бот на основе **Retrieval-Augmented Generation (RAG)**
для внутренней базы знаний вымышленной компании **QuantumForge Software**
(SaaS-платформа цифровых двойников промышленных объектов).

Бот даёт сотрудникам быстрые персонализированные ответы по разрозненной документации
(Markdown/MDX, Confluence, PDF) со ссылками на источники и выявляет пробелы в базе знаний.

Полное описание задачи, исследование и ход работы по заданиям — в
[`Project_template.md`](./Project_template.md).

## Стек

Python · LangChain · FAISS · Docker (Compose)

## Как устроена работа

- Базовая ветка — `main`; работа по заданиям — в ветке `rag`.
- Ответы по каждому заданию фиксируются в [`Project_template.md`](./Project_template.md).
- Сдача — pull request `rag → main`.

## Установка

Все команды — **из корня репозитория**. На чистой машине один раз ставим окружение
(если `.venv` уже собран — шаг можно пропустить):

```bash
cd ~/Projects/YandexTestTask_Bot   # корень репозитория (куда клонировали)

python -m venv .venv
# torch — CPU-сборкой, чтобы не тянуть CUDA (~2.5 ГБ)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
```

## Запуск бота (Задание 4)

LLM — локальная `Qwen/Qwen2.5-1.5B-Instruct` (transformers, CPU); модели
(`multilingual-e5-small` + Qwen) скачиваются при первом запуске. Индекс уже в
репозитории (`index/`). Генерация на CPU — ~30–90 с на ответ.

```bash
cd ~/Projects/YandexTestTask_Bot   # корень репозитория (куда клонировали)

.venv/Scripts/python scripts/rag_bot.py                       # диалог (REPL), выход — exit
.venv/Scripts/python scripts/rag_bot.py "Кто такой Горрук?"   # один вопрос
.venv/Scripts/python scripts/rag_bot.py --demo                # демо: 4 ответа + 2 «Я не знаю»
```

## Пересборка индекса (Задание 3, не обязательно)

```bash
.venv/Scripts/python scripts/build_index.py                   # пересоберёт index/
.venv/Scripts/python scripts/query_index.py "Кто такой Горрук?"
```

Docker-упаковка (бот + FAISS) — следующим шагом.
