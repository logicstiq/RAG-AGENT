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


def _read_docx(path: str) -> str:
    try:
        import docx
    except ImportError as e:
        raise RuntimeError("pip install python-docx to ingest .docx files") from e
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    # Tables are common in business docs (contract terms, pricing tables)
    # and would otherwise be silently dropped.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_document(path: str) -> str:
    """Reads one file by extension. Shared by folder ingestion and the
    live single-file upload endpoint, so both paths support exactly the
    same formats."""
    lower = path.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        return _read_txt(path)
    if lower.endswith(".pdf"):
        return _read_pdf(path)
    if lower.endswith(".docx"):
        return _read_docx(path)
    raise ValueError(f"Unsupported document type: {os.path.basename(path)}")


def load_documents(folder: str) -> list[tuple[str, str]]:
    """Returns [(filename, raw_text), ...] for every .txt/.md/.pdf/.docx file in folder."""
    docs = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if name.lower().endswith((".txt", ".md", ".pdf", ".docx")):
            docs.append((name, read_document(path)))
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
        chunks.extend(_chunks_for_one_file(filename, raw_text, chunk_size, overlap))
    return chunks


def build_chunks_for_file(path: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Same as build_chunks, but for one specific file rather than a whole
    folder — used by the live upload endpoint, which adds one new document
    to an already-ingested corpus rather than re-scanning a directory."""
    filename = os.path.basename(path)
    raw_text = read_document(path)
    return _chunks_for_one_file(filename, raw_text, chunk_size, overlap)


def _chunks_for_one_file(filename: str, raw_text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    pieces = split_into_chunks(raw_text, chunk_size, overlap)
    return [
        Chunk(chunk_id=f"{filename}::{i}", text=piece, source=filename, position=i)
        for i, piece in enumerate(pieces)
    ]
