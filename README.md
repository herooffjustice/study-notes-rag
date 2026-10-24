# Study Notes Q&A — Vector RAG over PDF notes

Ask questions about your PDF study notes. The app splits the PDF into small chunks, turns each chunk into an embedding, stores them in a **FAISS** index, finds the chunks closest to your question, and **Google Gemini** writes the answer **only** from those chunks, with page numbers.

Course: Advanced NLP (AML23702) · Global Academy of Technology

## How it works

```
PDF ──> page text (pypdf) ──> chunks (60 words, 15 overlap)
     ──> embeddings (all-MiniLM-L6-v2) ──> FAISS IndexFlatIP (cosine)
Question ──> embedding ──> top-4 chunks ──> score check
     ──> prompt with chunks + rules ──> Gemini ──> answer with (p. N) citations
```

If the best chunk scores below **0.25**, the app does not call Gemini and says the topic is not in the notes.

## Files

| File | What it does |
|---|---|
| `notes_rag.py` | Load PDF, chunk, embed, FAISS search, prompt, answer |
| `llm.py` | Calls Gemini (key from `.env`), retries on rate limits |
| `app.py` | Streamlit web app (upload PDF, ask, see chunks and scores) |
| `main.py` | Command-line version (`--demo` saves `outputs/demo_run.json`) |
| `evaluate_retrieval.py` | Hit@1 / Hit@3 for different chunk sizes (no API needed) |
| `data/make_sample_pdf.py` | Creates the 5-page sample notes PDF |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your Gemini key into .env
```

Get a free key at <https://aistudio.google.com> → **Get API key**.

## Run

```bash
python data/make_sample_pdf.py     # only if data/sample_notes.pdf is missing
python evaluate_retrieval.py       # retrieval check, no key needed
python main.py --demo              # demo questions with Gemini
streamlit run app.py               # web app
```

## Results (from the saved runs in `outputs/`)

Retrieval check on 12 test questions with a known page (`evaluate_retrieval.py`):

| Chunk size / overlap | Chunks | Hit@1 | Hit@3 | Mean top score | Answerable but below 0.25 |
|---|---|---|---|---|---|
| 40 / 10 | 36 | 0.917 | 1.0 | 0.542 | 1 |
| **60 / 15 (used)** | 24 | 0.833 | 1.0 | 0.509 | **0** |
| 80 / 20 | 18 | 0.917 | 1.0 | 0.474 | 2 |
| 120 / 30 | 12 | 0.917 | 1.0 | 0.433 | 2 |

60-word chunks are the only setting where no answerable question falls below the 0.25 threshold.
Off-topic questions score 0.09–0.20, below the 0.25 threshold, so Gemini is not called for them.

All results from the runs are saved in `outputs/`.
