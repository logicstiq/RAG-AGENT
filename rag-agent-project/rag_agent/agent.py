"""The RAG agent: ingest_docs()/ingest_data() build the two indexes,
query() answers a question by routing to whichever one fits.
"""
from __future__ import annotations

import os
import pickle

from chunker import build_chunks
from embedder import TfidfEmbedder
from graph_builder import build_chunk_graph
from llm_client import generate_answer, summarize_table_result
from retriever import retrieve
from structured_query import answer_structured_query
from structured_store import StructuredStore
from vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
GRAPH_PATH = os.path.join(DATA_DIR, "graph.pkl")
EMBEDDER_PATH = os.path.join(DATA_DIR, "embedder.pkl")


class RAGAgent:
    def __init__(self):
        self.embedder = TfidfEmbedder()
        self.store = VectorStore()
        self.graph = None
        self.structured = StructuredStore()

    # ---------- text documents (RAG path) ----------

    def ingest_docs(self, docs_folder: str, chunk_size: int = 800, overlap: int = 150) -> int:
        chunks = build_chunks(docs_folder, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"No .txt/.md/.pdf files found in {docs_folder}")

        texts = [c.text for c in chunks]
        vectors = self.embedder.fit_transform(texts)

        metadata = {c.chunk_id: {"text": c.text, "source": c.source} for c in chunks}
        self.store.add([c.chunk_id for c in chunks], vectors, metadata)

        self.graph = build_chunk_graph(chunks, self.embedder)

        os.makedirs(DATA_DIR, exist_ok=True)
        self.store.save(STORE_PATH)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        with open(EMBEDDER_PATH, "wb") as f:
            pickle.dump(self.embedder, f)

        return len(chunks)

    # ---------- structured files: CSV/XLSX (exact-match path) ----------

    def ingest_data(self, data_folder: str) -> list[str]:
        loaded = self.structured.ingest_folder(data_folder)
        if not loaded:
            raise ValueError(f"No .csv/.xlsx files found in {data_folder}")
        os.makedirs(DATA_DIR, exist_ok=True)
        self.structured.save()
        return loaded

    # ---------- loading a previously built index ----------

    def load(self) -> None:
        if os.path.exists(STORE_PATH):
            self.store.load(STORE_PATH)
            with open(GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)
            with open(EMBEDDER_PATH, "rb") as f:
                self.embedder = pickle.load(f)
        self.structured.load()  # fine if empty — is_empty() handles it

        if self.store.vectors is None and self.structured.is_empty():
            raise RuntimeError(
                "No index found — run `python cli.py ingest --docs <folder>` "
                "and/or `python cli.py ingest-data --files <folder>` first."
            )

    # ---------- answering ----------

    def query(self, question: str, top_k: int = 4, graph_expand: int = 2) -> dict:
        # Try the structured (exact-match) path first if any spreadsheets were
        # ingested — a real filter result beats a fuzzy text-similarity guess
        # whenever the question is actually about row-level data.
        if not self.structured.is_empty():
            result = answer_structured_query(question, self.structured.tables)
            if result["matched"]:
                answer = summarize_table_result(question, result["text"])
                return {
                    "answer": answer,
                    "sources": [result["table_name"]],
                    "passages": [],
                    "mode": "structured",
                }

        if self.store.vectors is None:
            return {
                "answer": (
                    "That didn't match any row, column, or filter I recognize in "
                    "the ingested data file(s), and no text documents are "
                    "ingested either.\n\n" + self.structured.schema_summary()
                ),
                "sources": [],
                "passages": [],
                "mode": "none",
            }

        passages = retrieve(question, self.embedder, self.store, self.graph, top_k, graph_expand)
        answer = generate_answer(question, passages)
        sources = sorted({p["source"] for p in passages})
        return {"answer": answer, "sources": sources, "passages": passages, "mode": "text"}
