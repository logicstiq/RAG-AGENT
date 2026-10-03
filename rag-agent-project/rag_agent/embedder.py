"""Turns chunk text into vectors. Default backend needs no API key or download.

Swap in a neural/API embedding backend later by implementing the same
two methods (fit_transform / transform) and pointing agent.py at it.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features: int = 20_000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder has not been fit yet — run ingest first.")
        return self.vectorizer.transform(texts).toarray()

    def top_terms(self, text: str, n: int = 8) -> list[str]:
        """Top keywords for one piece of text, used by the graph builder."""
        vec = self.vectorizer.transform([text]).toarray()[0]
        if vec.sum() == 0:
            return []
        idx = np.argsort(vec)[::-1][:n]
        feature_names = self.vectorizer.get_feature_names_out()
        return [feature_names[i] for i in idx if vec[i] > 0]
