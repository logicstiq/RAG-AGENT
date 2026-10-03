"""Minimal in-memory vector store: cosine similarity over a numpy matrix.

No FAISS/Milvus/Chroma dependency — fine up to tens of thousands of chunks,
which covers almost any portfolio-scale or small-team document set.
"""
from __future__ import annotations

import pickle

import numpy as np


class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self.metadata: dict[str, dict] = {}

    def add(self, ids: list[str], vectors: np.ndarray, metadata: dict[str, dict]) -> None:
        self.ids = list(ids)
        self.vectors = vectors
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-9
        scores = (self.vectors @ query_vector) / norms
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"ids": self.ids, "vectors": self.vectors, "metadata": self.metadata}, f
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.ids = state["ids"]
        self.vectors = state["vectors"]
        self.metadata = state["metadata"]
