"""
app.py - Streamlit web app for the Study Notes Q&A (vector RAG).

Run:  streamlit run app.py
"""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from llm import get_model_name
from notes_rag import (CHUNK_OVERLAP, CHUNK_SIZE, MIN_SCORE, TOP_K, NotesIndex,
                       answer_question, chunk_pages, load_pdf)

st.set_page_config(page_title="Study Notes Q&A", page_icon="📘", layout="wide")

SAMPLE_PDF = Path("data") / "sample_notes.pdf"
EXAMPLES = [
    "Why do we divide by the square root of d_k in attention?",
    "What is the difference between BERT and GPT?",
    "How does Byte Pair Encoding work?",
    "Who won the IPL in 2023?",
]


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_index_object():
    return NotesIndex()


@st.cache_data(show_spinner="Reading and indexing the PDF...")
def build_index(pdf_bytes, chunk_size, overlap):
    pages = load_pdf(io.BytesIO(pdf_bytes))
    chunks = chunk_pages(pages, chunk_size, overlap)
    return pages, chunks


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Notes")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        pdf_bytes, pdf_name = uploaded.getvalue(), uploaded.name
    else:
        pdf_bytes, pdf_name = SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name
        st.caption("No file uploaded, so the sample notes are used.")

    st.header("Settings")
    chunk_size = st.slider("Chunk size (words)", 30, 150, CHUNK_SIZE, 10)
    overlap = st.slider("Overlap (words)", 0, 40, CHUNK_OVERLAP, 5)
    top_k = st.slider("Top-k chunks", 1, 8, TOP_K)
    min_score = st.slider("Minimum match score", 0.0, 0.6, MIN_SCORE, 0.05)
    st.caption(f"Embeddings: all-MiniLM-L6-v2 · Index: FAISS (cosine) · LLM: {get_model_name()}")

pages, chunks = build_index(pdf_bytes, chunk_size, overlap)
index = get_index_object()
key = hashlib.md5(pdf_bytes + f"{chunk_size}-{overlap}".encode()).hexdigest()
if st.session_state.get("index_key") != key:
    index.build(chunks)
    st.session_state["index_key"] = key

# ---------------- Main area ----------------
st.title("Study Notes Q&A")
st.write("Ask a question about your PDF notes. The app finds the most similar chunks "
         "with FAISS and Gemini answers **only** from those chunks, with page citations.")

c1, c2, c3 = st.columns(3)
c1.metric("File", pdf_name)
c2.metric("Pages", len(pages))
c3.metric("Chunks in FAISS", len(chunks))

tab_ask, tab_chunks, tab_eval = st.tabs(["Ask", "All chunks", "Retrieval check"])

with tab_ask:
    st.write("**Example questions:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state["question"] = example

    question = st.text_input("Your question", key="question")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching the notes and asking Gemini..."):
                out = answer_question(index, question, k=top_k, min_score=min_score)
            st.subheader("Answer")
            if out["llm_called"]:
                st.success(out["answer"])
            else:
                st.warning(out["answer"])
            st.subheader("Retrieved chunks")
            for rank, r in enumerate(out["results"], start=1):
                with st.expander(f"#{rank} · page {r['page']} · chunk {r['chunk_id']} · "
                                 f"score {r['score']:.3f}", expanded=(rank == 1)):
                    st.write(r["text"])

with tab_chunks:
    st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)

with tab_eval:
    eval_file = Path("outputs") / "retrieval_eval.json"
    if eval_file.exists():
        data = json.loads(eval_file.read_text())
        st.write(f"Retrieval accuracy on {data['num_questions']} test questions "
                 "for different chunk sizes (from `evaluate_retrieval.py`):")
        st.dataframe(pd.DataFrame(data["chunk_settings"]), hide_index=True)
        st.write("Top match score for off-topic questions:")
        st.json(data["off_topic_scores"])
    else:
        st.info("Run `python evaluate_retrieval.py` to create this table.")
