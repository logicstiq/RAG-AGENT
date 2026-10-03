"""Hybrid retrieval: top-k vector hits, then pull in their graph neighbors."""
from __future__ import annotations

import networkx as nx

from embedder import TfidfEmbedder
from graph_builder import neighbors
from vector_store import VectorStore


def retrieve(
    query: str,
    embedder: TfidfEmbedder,
    store: VectorStore,
    graph: nx.Graph,
    top_k: int = 4,
    graph_expand: int = 2,
) -> list[dict]:
    query_vec = embedder.transform([query])[0]
    hits = store.search(query_vec, top_k=top_k)

    seen = set()
    results: list[dict] = []

    for chunk_id, score in hits:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        results.append({**store.metadata[chunk_id], "chunk_id": chunk_id, "score": score, "via": "vector"})

        for neighbor_id in neighbors(graph, chunk_id, limit=graph_expand):
            if neighbor_id in seen or neighbor_id not in store.metadata:
                continue
            seen.add(neighbor_id)
            results.append(
                {**store.metadata[neighbor_id], "chunk_id": neighbor_id, "score": score * 0.75, "via": "graph"}
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
