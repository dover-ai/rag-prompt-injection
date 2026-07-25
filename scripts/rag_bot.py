#!/usr/bin/env python3
"""
rag_bot.py — RAG-бот по базе знаний (Задание 4).

Цепочка (собрана вручную, чтобы было видно, что под капотом):
  запрос → эмбеддинг (e5-small, тот же, что при индексации) → поиск в FAISS →
  сборка промпта (few-shot + Chain-of-Thought + найденный контекст) →
  локальная LLM (transformers) → ответ со ссылкой на источники.

Если в найденных чанках нет ответа — бот честно пишет «Я не знаю»
(и по порогу расстояния, и по инструкции в System-промпте).

Запуск:
  .venv/Scripts/python scripts/rag_bot.py                        # интерактивный REPL
  .venv/Scripts/python scripts/rag_bot.py "Кто такой Горрук?"    # один вопрос
  .venv/Scripts/python scripts/rag_bot.py --demo                 # прогнать демо-вопросы
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from build_index import E5Embeddings, INDEX_DIR

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 4
# Расстояние L2² на нормализованных векторах (0 — идеально). Если ближайший чанк
# дальше порога — в базе нет релевантного, отвечаем «Я не знаю» не дёргая LLM.
# Откалибровано: внутрибазовые запросы ~0.18–0.31, посторонние ~0.41–0.48.
MAX_DISTANCE = 0.40

SYSTEM_PROMPT = (
    "Ты — ассистент корпоративной базы знаний. Отвечай на русском и ТОЛЬКО по тексту "
    "в разделе «Контекст». Правила:\n"
    "1) Сначала рассуждай пошагово и КРАТКО — 2–4 пронумерованных шага (что нужно "
    "найти и что нашлось в контексте). Не повторяйся.\n"
    "2) Затем с новой строки дай краткий итог после слова «Ответ:».\n"
    "3) Если в контексте нет ответа на вопрос — не выдумывай, напиши ровно «Я не знаю»."
)

# Few-shot примеры — из НАШЕЙ базы (реальные фрагменты про вымышленный мир),
# показывают модели формат «шаги рассуждения + Ответ:».
FEWSHOTS = [
    {
        "context": "[Греннок] Греннок — одна из двух основных политических фракций "
                   "смертных рас Аурета; её противник — Вальдоры. Ценит силу и честь.",
        "question": "Кто противостоит Гренноку?",
        "answer": "1. Нужно найти врага фракции Греннок.\n"
                  "2. В контексте сказано: противник Греннока — Вальдоры.\n"
                  "Ответ: Гренноку противостоят Вальдоры.",
    },
    {
        "context": "[Горрук] Горрук — бывший вождь Греннока и шаман, основатель нации "
                   "Дарган в Марендоре.",
        "question": "Что основал Горрук?",
        "answer": "1. Нужно найти, что основал Горрук.\n"
                  "2. В контексте: Горрук основал нацию Дарган в Марендоре.\n"
                  "Ответ: Горрук основал нацию Дарган (в Марендоре).",
    },
]

DEMO_QUESTIONS = [
    "Кто такой Горрук?",
    "Что такое Криндель и на что он способен?",
    "Кто противостоит фракции Греннок?",
    "Что такое Аурет?",
    "Кто такой Дарт Вейдер?",          # нет в базе → «Я не знаю»
    "Какая сегодня погода в Москве?",  # не по теме → «Я не знаю»
]


def format_context(docs):
    return "\n\n".join(f"[{d.metadata.get('title')}] {d.page_content.strip()}" for d in docs)


def build_messages(context, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOTS:
        messages.append({"role": "user",
                         "content": f"Контекст:\n{ex['context']}\n\nВопрос: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user",
                     "content": f"Контекст:\n{context}\n\nВопрос: {question}"})
    return messages


class RagBot:
    def __init__(self):
        self.emb = E5Embeddings()
        self.store = FAISS.load_local(str(INDEX_DIR), self.emb,
                                      allow_dangerous_deserialization=True)
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32)
        self.model.eval()

    def answer(self, question, k=TOP_K):
        hits = self.store.similarity_search_with_score(question, k=k)
        sources = [(d.metadata.get("title"), d.metadata.get("chunk_id"), float(s)) for d, s in hits]
        best = hits[0][1] if hits else 9.9
        if not hits or best > MAX_DISTANCE:
            return "Я не знаю", sources  # в базе нет релевантного контекста
        messages = build_messages(format_context([d for d, _ in hits]), question)
        print("  … генерирую ответ (модель на CPU, обычно 30–90 с) …", flush=True)
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False,
                                       repetition_penalty=1.15,
                                       pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return gen, sources


def show(question, answer, sources):
    print("\n" + "=" * 72)
    print(f"Вопрос: {question}\n")
    print(answer)
    if sources:
        print("\nИсточники (title · chunk_id · distance):")
        for title, cid, dist in sources:
            print(f"  - {title} ({cid}) d={dist:.3f}")


def main():
    print(f"Загрузка индекса и модели {LLM_MODEL} …")
    bot = RagBot()
    args = sys.argv[1:]
    if args == ["--demo"]:
        for q in DEMO_QUESTIONS:
            ans, src = bot.answer(q)
            show(q, ans, src)
        return
    if args:
        q = " ".join(args)
        ans, src = bot.answer(q)
        show(q, ans, src)
        return
    print("RAG-бот (Задание 4). Введите вопрос. 'exit' — выход.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q.lower() in ("exit", "quit", "выход"):
            break
        if not q:
            continue
        ans, src = bot.answer(q)
        show(q, ans, src)


if __name__ == "__main__":
    main()
