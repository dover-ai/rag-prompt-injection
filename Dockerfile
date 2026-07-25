# Минимальный образ RAG-бота QuantumForge (бот + FAISS в одном процессе).
FROM python:3.11-slim

# libgomp1 — OpenMP для faiss-cpu / torch
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости: torch — CPU-сборкой (без CUDA), затем остальное
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Код, готовый FAISS-индекс и база знаний
COPY scripts/ ./scripts/
COPY index/ ./index/
COPY knowledge_base/ ./knowledge_base/

# Кэш скачиваемых моделей (e5-small + Qwen) — в volume (см. docker-compose.yml),
# чтобы не качать их при каждом запуске.
ENV HF_HOME=/models

# По умолчанию — интерактивный бот (REPL).
# Демо:  docker compose run --rm rag-bot python scripts/rag_bot.py --demo
# Атака: docker compose run --rm rag-bot python scripts/rag_bot.py --attack
CMD ["python", "scripts/rag_bot.py"]
