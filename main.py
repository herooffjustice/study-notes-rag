"""
main.py - command-line version of the Study Notes Q&A.

Usage:
  python main.py                         # sample notes, interactive
  python main.py --pdf my_notes.pdf      # your own PDF
  python main.py --demo                  # run the demo questions and save outputs
"""

import argparse
import json
from pathlib import Path

from notes_rag import NotesIndex, answer_question, chunk_pages, load_pdf

DEMO_QUESTIONS = [
    "Why do we divide by the square root of d_k in scaled dot-product attention?",
    "What is the difference between BERT and GPT?",
    "How does Byte Pair Encoding work?",
    "What problem does an LSTM solve?",
    "Who won the IPL in 2023?",
]


def show(question, out):
    print("=" * 70)
    print("Question:", question)
    print("-" * 70)
    for r in out["results"]:
        print(f"  score {r['score']:.3f} | page {r['page']} | chunk {r['chunk_id']}")
    print("-" * 70)
    print("Answer:", out["answer"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(Path("data") / "sample_notes.pdf"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    pages = load_pdf(args.pdf)
    chunks = chunk_pages(pages)
    index = NotesIndex().build(chunks)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks indexed in FAISS.")

    if args.demo:
        log = []
        for q in DEMO_QUESTIONS:
            out = answer_question(index, q)
            show(q, out)
            log.append({
                "question": q,
                "retrieved": [{"page": r["page"], "chunk_id": r["chunk_id"],
                               "score": round(r["score"], 3)} for r in out["results"]],
                "llm_called": out["llm_called"],
                "answer": out["answer"],
            })
        Path("outputs").mkdir(exist_ok=True)
        with open("outputs/demo_run.json", "w") as f:
            json.dump(log, f, indent=2)
        print("\nSaved outputs/demo_run.json")
        return

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        show(q, answer_question(index, q))


if __name__ == "__main__":
    main()
