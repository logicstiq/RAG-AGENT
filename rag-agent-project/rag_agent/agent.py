"""The RAG agent: ingest_docs()/ingest_data() build the two indexes,
query() answers a question by routing to whichever one fits.
"""
from __future__ import annotations

import os
import pickle

from chunker import Chunk, build_chunks, build_chunks_for_file
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
            raise ValueError(f"No .txt/.md/.pdf/.docx files found in {docs_folder}")
        self._build_text_index(chunks)
        return len(chunks)

    def ingest_additional_doc(self, path: str) -> int:
        """Adds ONE new file (text or spreadsheet-adjacent prose, not a
        table) to the existing text index — used by the live upload
        endpoint. TF-IDF has to be refit on the whole corpus to stay
        consistent (there's no clean way to add one document to an
        already-fitted vectorizer), so this reconstructs every existing
        chunk's text from the vector store's own metadata, adds the new
        file's chunks, and rebuilds from there. Cheap at this corpus size;
        would need a different approach at a much larger scale.
        """
        existing_chunks = [
            Chunk(chunk_id=cid, text=meta["text"], source=meta["source"], position=0)
            for cid, meta in self.store.metadata.items()
        ] if self.store.vectors is not None else []

        new_chunks = build_chunks_for_file(path)
        if not new_chunks:
            raise ValueError(f"No readable text found in {os.path.basename(path)}")

        self._build_text_index(existing_chunks + new_chunks)
        return len(new_chunks)

    def _build_text_index(self, chunks: list[Chunk]) -> None:
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

    def query(self, question: str, top_k: int = 6, graph_expand: int = 2) -> dict:
        # Try the structured (exact-match) path first if any spreadsheets were
        # ingested — a real filter result beats a fuzzy text-similarity guess
        # whenever the question is actually about row-level data.
        if not self.structured.is_empty():
            result = answer_structured_query(question, self.structured.tables)
            if result["matched"]:
                answer = summarize_table_result(question, result["text"])
                chart_image = None
                if result.get("chart"):
                    from chart_builder import render_bar_chart

                    spec = result["chart"]
                    chart_image = render_bar_chart(spec["labels"], spec["values"], spec["title"])
                return {
                    "answer": answer,
                    "sources": [result["table_name"]],
                    "passages": [],
                    "mode": "structured",
                    "chart": chart_image,
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
                "chart": None,
            }

        passages = retrieve(question, self.embedder, self.store, self.graph, top_k, graph_expand)
        answer = generate_answer(question, passages)
        sources = sorted({p["source"] for p in passages})
        return {"answer": answer, "sources": sources, "passages": passages, "mode": "text", "chart": None}
