#!/usr/bin/env python3
"""
query_index.py — демо-поиск по FAISS-индексу базы знаний.

Загружает index/, прогоняет примеры запросов (или запрос из аргументов) и печатает
топ-3 найденных чанка с релевантностью и источником (заголовок + chunk_id).

Запуск:
  .venv/Scripts/python scripts/query_index.py                 # примеры запросов
  .venv/Scripts/python scripts/query_index.py "Кто такой Горрук?"
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_community.vectorstores import FAISS
from build_index import E5Embeddings, INDEX_DIR

# запросы используют ВЫМЫШЛЕННЫЕ имена (в базе только они): Горрук=Тралл,
# Криндель=Ледяная Скорбь, Греннок=Орда — проверяем, что бот отвечает по базе.
DEMO_QUERIES = [
    "Кто такой Горрук?",
    "Что такое Криндель?",
    "Кто входит в фракцию Греннок?",
]


def main():
    emb = E5Embeddings()
    store = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUERIES
    for q in queries:
        print("\n" + "=" * 72)
        print(f"ЗАПРОС: {q}")
        for doc, score in store.similarity_search_with_score(q, k=3):
            title = doc.metadata.get("title")
            cid = doc.metadata.get("chunk_id")
            snippet = " ".join(doc.page_content.split())[:200]
            print(f"  [{score:.3f}] {title}  ({cid})")
            print(f"     {snippet}…")


if __name__ == "__main__":
    main()
