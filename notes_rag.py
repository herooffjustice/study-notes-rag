"""
notes_rag.py - the full RAG pipeline for PDF study notes.

Steps:
  1. load_pdf()        read the text of every page
  2. chunk_pages()     cut the text into small overlapping chunks
  3. NotesIndex        turn chunks into embeddings and store them in FAISS
  4. search()          find the chunks closest to the question
  5. answer_question() send the chunks + question to Gemini
"""

import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from llm import generate

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 60       # words per chunk
CHUNK_OVERLAP = 15    # words shared between two neighbouring chunks
TOP_K = 4             # chunks sent to the LLM
MIN_SCORE = 0.25      # below this cosine score we treat the notes as "no match"


# ---------- 1. Load ----------
def load_pdf(path_or_file):
    """Return a list of {"page": n, "text": "..."} (page numbers start at 1)."""
    reader = PdfReader(path_or_file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": number, "text": text})
    return pages


# ---------- 2. Chunk ----------
def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split each page into word windows. Each chunk remembers its page."""
    chunks = []
    step = chunk_size - overlap
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if len(piece) < 15 and start > 0:   # skip tiny tail pieces
                break
            chunks.append({
                "chunk_id": len(chunks),
                "page": page["page"],
                "text": " ".join(piece),
            })
            if start + chunk_size >= len(words):
                break
    return chunks


# ---------- 3. Embed + index ----------
class NotesIndex:
    """Holds the embedding model, the FAISS index and the chunk texts."""

    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        # normalize_embeddings=True makes inner product equal to cosine similarity
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    # ---------- 4. Search ----------
    def search(self, question, k=TOP_K):
        query = self.model.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(query, dtype="float32"), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)
        return results


# ---------- 5. Generate ----------
def build_context(results):
    blocks = []
    for r in results:
        blocks.append(f"[Page {r['page']} | chunk {r['chunk_id']}]\n{r['text']}")
    return "\n\n".join(blocks)


def build_prompt(question, context):
    return f"""You are a study assistant. Answer the student's question using ONLY the notes below.

Rules:
1. Use only the information in the notes. Do not use outside knowledge.
2. After each fact, cite the page like this: (p. 3).
3. If the notes do not contain the answer, reply exactly:
   "This is not covered in the uploaded notes."
4. Keep the answer short and clear (2 to 5 sentences).

Notes:
{context}

Question: {question}

Answer:"""


def answer_question(index, question, k=TOP_K, min_score=MIN_SCORE):
    """Return a dict with the retrieved chunks, the prompt and the answer."""
    results = index.search(question, k)
    best = results[0]["score"] if results else 0.0
    if best < min_score:
        return {
            "results": results,
            "context": "",
            "answer": "This is not covered in the uploaded notes. "
                      f"(Best match score {best:.2f} is below the threshold {min_score}.)",
            "llm_called": False,
        }
    context = build_context(results)
    answer = generate(build_prompt(question, context))
    return {"results": results, "context": context, "answer": answer, "llm_called": True}
