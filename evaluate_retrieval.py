"""
evaluate_retrieval.py - checks how chunk size affects retrieval.

For each test question we know which page holds the answer.
Hit@1 = the top chunk is from the right page.
Hit@3 = one of the top 3 chunks is from the right page.
Below threshold = answerable questions whose best score is under MIN_SCORE,
                  so the app would wrongly say "not covered".
No LLM is used here, so the result is exactly the same on every run.

Run:  python evaluate_retrieval.py
"""

import json
from pathlib import Path

from notes_rag import MIN_SCORE, NotesIndex, chunk_pages, load_pdf

TEST_QUESTIONS = [
    ("Why are one-hot vectors bad at showing meaning?", 1),
    ("What is the difference between CBOW and Skip-gram?", 1),
    ("What is the vanishing gradient problem?", 2),
    ("What gates does an LSTM have?", 2),
    ("Who introduced attention for machine translation?", 3),
    ("Why do we divide by the square root of d_k?", 3),
    ("How many heads did the original Transformer use?", 3),
    ("Why does the Transformer need positional encoding?", 4),
    ("What is the time complexity of self-attention?", 4),
    ("How is BERT pre-trained?", 5),
    ("How does Byte Pair Encoding build its vocabulary?", 5),
    ("What does RAG stand for and why is it useful?", 5),
]

CHUNK_SETTINGS = [(40, 10), (60, 15), (80, 20), (120, 30)]


def main():
    pages = load_pdf(Path("data") / "sample_notes.pdf")
    index = NotesIndex()
    rows = []
    for size, overlap in CHUNK_SETTINGS:
        chunks = chunk_pages(pages, size, overlap)
        index.build(chunks)
        hit1 = hit3 = below = 0
        top_scores = []
        for question, gold_page in TEST_QUESTIONS:
            results = index.search(question, k=3)
            pages_found = [r["page"] for r in results]
            hit1 += pages_found[0] == gold_page
            hit3 += gold_page in pages_found
            top_scores.append(results[0]["score"])
            below += results[0]["score"] < MIN_SCORE
        n = len(TEST_QUESTIONS)
        rows.append({
            "chunk_size": size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "hit_at_1": round(hit1 / n, 3),
            "hit_at_3": round(hit3 / n, 3),
            "mean_top_score": round(sum(top_scores) / n, 3),
            "below_threshold": int(below),
        })

    # Out-of-scope questions: the best score should stay low.
    index.build(chunk_pages(pages))
    off_topic = {}
    for q in ["Who won the IPL in 2023?", "What is the capital of France?",
              "How does photosynthesis work?"]:
        off_topic[q] = round(index.search(q, k=1)[0]["score"], 3)

    print(f"{'size':>5} {'overlap':>8} {'chunks':>7} {'Hit@1':>6} {'Hit@3':>6} "
          f"{'meanTop':>8} {'below':>6}")
    for r in rows:
        print(f"{r['chunk_size']:>5} {r['overlap']:>8} {r['num_chunks']:>7} "
              f"{r['hit_at_1']:>6} {r['hit_at_3']:>6} {r['mean_top_score']:>8} "
              f"{r['below_threshold']:>6}")
    print("\nTop score for off-topic questions (threshold is 0.25):")
    for q, s in off_topic.items():
        print(f"  {s:.3f}  {q}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/retrieval_eval.json", "w") as f:
        json.dump({"chunk_settings": rows, "off_topic_scores": off_topic,
                   "num_questions": len(TEST_QUESTIONS)}, f, indent=2)
    print("\nSaved outputs/retrieval_eval.json")


if __name__ == "__main__":
    main()
