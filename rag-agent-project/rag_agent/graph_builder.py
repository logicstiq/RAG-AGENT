"""Links chunks that share top keywords — the 'graph' in graph-lite RAG.

This is the simplified stand-in for Medical-Graph-RAG's UMLS/hierarchy
linking: instead of an ontology, we link any two chunks whose top-keyword
sets overlap enough. When a query matches one chunk, its graph neighbors
(related passages elsewhere in the corpus, possibly in a different source
file) get pulled into context too.
"""
from __future__ import annotations

import networkx as nx

from chunker import Chunk
from embedder import TfidfEmbedder


def build_chunk_graph(
    chunks: list[Chunk], embedder: TfidfEmbedder, min_shared_terms: int = 2
) -> nx.Graph:
    graph = nx.Graph()
    keyword_sets: dict[str, set[str]] = {}

    for chunk in chunks:
        graph.add_node(chunk.chunk_id, source=chunk.source)
        keyword_sets[chunk.chunk_id] = set(embedder.top_terms(chunk.text, n=10))

    ids = [c.chunk_id for c in chunks]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = keyword_sets[ids[i]] & keyword_sets[ids[j]]
            if len(shared) >= min_shared_terms:
                graph.add_edge(ids[i], ids[j], weight=len(shared), shared=sorted(shared))

    return graph


def neighbors(graph: nx.Graph, chunk_id: str, limit: int = 3) -> list[str]:
    if chunk_id not in graph:
        return []
    ranked = sorted(
        graph.adj[chunk_id].items(), key=lambda kv: kv[1]["weight"], reverse=True
    )
    return [node_id for node_id, _ in ranked[:limit]]
