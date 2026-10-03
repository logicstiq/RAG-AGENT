"""Load .txt/.pdf files from a folder and split them into overlapping chunks."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    position: int


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pip install pypdf to ingest PDF files") from e
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(folder: str) -> list[tuple[str, str]]:
    """Returns [(filename, raw_text), ...] for every .txt/.pdf file in folder."""
    docs = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith(".txt") or lower.endswith(".md"):
            docs.append((name, _read_txt(path)))
        elif lower.endswith(".pdf"):
            docs.append((name, _read_pdf(path)))
    return docs


def split_into_chunks(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    """Paragraph-aware sliding window split, roughly `chunk_size` chars each."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            # carry overlap forward from the tail of the previous chunk
            tail = buf[-overlap:] if buf else ""
            buf = f"{tail}\n{para}".strip() if tail else para
            # if a single paragraph is itself bigger than chunk_size, hard-split it
            while len(buf) > chunk_size * 1.5:
                chunks.append(buf[:chunk_size])
                buf = buf[chunk_size - overlap :]
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(folder: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename, raw_text in load_documents(folder):
        pieces = split_into_chunks(raw_text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}::{i}",
                    text=piece,
                    source=filename,
                    position=i,
                )
            )
    return chunks
