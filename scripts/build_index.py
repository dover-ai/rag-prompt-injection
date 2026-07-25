#!/usr/bin/env python3
"""
build_index.py — построение векторного индекса базы знаний (FAISS).

Читает knowledge_base/*.md → режет на чанки (RecursiveCharacterTextSplitter) →
считает эмбеддинги (multilingual-e5-small) → сохраняет FAISS-индекс в index/.
У каждого чанка сохраняются метаданные: source (файл), title (заголовок), chunk_id —
чтобы бот мог ссылаться на источник при цитировании.

Запуск:  .venv/Scripts/python scripts/build_index.py
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "index"
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


class E5Embeddings(Embeddings):
    """Обёртка sentence-transformers. Модели multilingual-e5 требуют префиксы
    'passage: ' для документов и 'query: ' для запросов."""

    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vecs = self.model.encode(["passage: " + t for t in texts],
                                 normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def embed_query(self, text):
        return self.model.encode("query: " + text, normalize_embeddings=True).tolist()


def load_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    files = sorted(KB.glob("*.md"))
    chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else f.stem
        for i, part in enumerate(splitter.split_text(text)):
            chunks.append(Document(
                page_content=part,
                metadata={"source": f.name, "title": title, "chunk_id": f"{f.stem}#{i}"}))
    return files, chunks


def main():
    files, chunks = load_chunks()
    print(f"документов: {len(files)} | чанков: {len(chunks)}")
    print(f"модель: {MODEL_NAME} (загрузка при первом запуске может занять время)")

    emb = E5Embeddings()  # загрузка/скачивание модели — вне замера времени генерации
    dim = len(emb.embed_query("тест"))

    t0 = time.perf_counter()
    store = FAISS.from_documents(chunks, emb)
    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    dt = time.perf_counter() - t0

    info = {
        "model": MODEL_NAME,
        "embedding_dim": dim,
        "documents": len(files),
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "build_seconds": round(dt, 1),
    }
    (INDEX_DIR / "build_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"размерность вектора: {dim}")
    print(f"индекс сохранён: {INDEX_DIR}\\index.faiss + index.pkl")
    print(f"время генерации эмбеддингов + индекса: {dt:.1f} c")


if __name__ == "__main__":
    main()
