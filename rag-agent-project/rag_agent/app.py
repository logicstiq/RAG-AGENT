"""Minimal web chat UI for the RAG agent, plus a live file-upload endpoint
so a NEW spreadsheet can be ingested through the website itself — no
terminal, no redeploy. Run with: python app.py
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from agent import RAGAgent

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
DATA_EXTENSIONS = (".csv", ".xlsx", ".xls")
DOC_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")

app = Flask(__name__)
agent = RAGAgent()
try:
    agent.load()  # whatever was pre-ingested via the CLI, if anything
except RuntimeError:
    # Nothing pre-ingested yet — fine. The person can upload a file through
    # the website itself (see /api/upload-data below) and start from there.
    pass


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = agent.query(question)
    return jsonify(
        {
            "answer": result["answer"],
            "sources": result["sources"],
            "chart": result.get("chart"),
        }
    )


@app.route("/api/upload-data", methods=["POST"])
def upload_data():
    """Lets someone drop a new CSV/XLSX straight into the live chat —
    no CLI, no redeploy.

    This REPLACES whatever structured tables were already loaded (the demo
    data included), rather than adding alongside them. That's deliberate:
    once someone uploads their own file, the reasonable assumption is that
    questions are about THAT file. Merging it with leftover demo tables
    would mean a question can silently get answered from the wrong table
    whenever both happen to have a similarly-named column — confusing in
    exactly the moment (a live demo) where it matters most to get right.
    Text documents (sample_docs) are untouched either way.
    """
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(DATA_EXTENSIONS):
        return jsonify({"error": "Only .csv and .xlsx/.xls files are supported"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    agent.structured.tables = {}  # replace, don't merge — see docstring above
    try:
        added_tables = agent.structured.load_file(path)
    except Exception as e:
        return jsonify({"error": f"Could not read that file: {e}"}), 400

    if not added_tables:
        return jsonify({"error": "That file didn't have any readable rows/columns in it."}), 400

    try:
        agent.structured.save()  # best-effort — the upload is already usable even if this fails
    except Exception:
        pass

    tables_summary = [
        {
            "name": name,
            "rows": len(agent.structured.tables[name]),
            "columns": list(agent.structured.tables[name].columns),
        }
        for name in added_tables
    ]
    return jsonify({"tables": tables_summary})


@app.route("/api/upload-docs", methods=["POST"])
def upload_docs():
    """Lets someone drop a new .txt/.md/.pdf/.docx straight into the live
    chat. Unlike the spreadsheet upload above, this ADDS to the existing
    text corpus rather than replacing it — a new policy doc or report is
    normally meant to supplement a knowledge base, not wipe it out, and
    there's no "wrong table" collision risk the way there is with
    spreadsheets (a text question can legitimately draw on several
    documents at once; that's the whole point of the graph-linking)."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(DOC_EXTENSIONS):
        return jsonify({"error": "Only .txt, .md, .pdf, and .docx files are supported"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        n_chunks = agent.ingest_additional_doc(path)
    except Exception as e:
        return jsonify({"error": f"Could not read that file: {e}"}), 400

    return jsonify({"filename": filename, "chunks": n_chunks})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
