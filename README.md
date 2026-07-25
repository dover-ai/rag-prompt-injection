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

## Сборка векторного индекса (Задание 3)

```bash
python -m venv .venv
# Windows: torch — CPU-сборкой, чтобы не тянуть CUDA (~2.5 ГБ)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt

# собрать индекс (модель e5-small скачается при первом запуске) → index/
.venv/Scripts/python scripts/build_index.py

# поиск по индексу
.venv/Scripts/python scripts/query_index.py "Кто такой Горрук?"
```

## Запуск бота

_Появится после сборки бота (Docker + FAISS)._
