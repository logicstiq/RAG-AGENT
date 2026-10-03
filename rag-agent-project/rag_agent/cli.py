"""Terminal interface:
  python cli.py ingest-docs --docs sample_docs       (text: .txt/.md/.pdf)
  python cli.py ingest-data --files sample_data       (tables: .csv/.xlsx)
  python cli.py ask "your question"
  python cli.py chat
"""
from __future__ import annotations

import argparse

from agent import RAGAgent


def main():
    parser = argparse.ArgumentParser(description="Graph-lite RAG agent")
    sub = parser.add_subparsers(dest="command", required=True)

    docs_p = sub.add_parser("ingest-docs", help="Build the text index from a folder of documents")
    docs_p.add_argument("--docs", required=True, help="Folder containing .txt/.md/.pdf files")
    docs_p.add_argument("--chunk-size", type=int, default=800)
    docs_p.add_argument("--overlap", type=int, default=150)

    data_p = sub.add_parser("ingest-data", help="Load a folder of CSV/XLSX files for exact-match lookups")
    data_p.add_argument("--files", required=True, help="Folder containing .csv/.xlsx files")

    ask_p = sub.add_parser("ask", help="Ask a single question against whatever is ingested")
    ask_p.add_argument("question")
    ask_p.add_argument("--top-k", type=int, default=4)

    sub.add_parser("chat", help="Interactive question loop")

    args = parser.parse_args()
    agent = RAGAgent()

    if args.command == "ingest-docs":
        n = agent.ingest_docs(args.docs, args.chunk_size, args.overlap)
        print(f"Ingested {n} text chunks from '{args.docs}'. Index saved to ./data/")
        return

    if args.command == "ingest-data":
        loaded = agent.ingest_data(args.files)
        print(f"Loaded {len(loaded)} table(s) from '{args.files}': {', '.join(loaded)}")
        return

    agent.load()

    if args.command == "ask":
        result = agent.query(args.question, top_k=args.top_k)
        _print_result(result)
    elif args.command == "chat":
        print("Graph-lite RAG agent — type a question, or 'quit' to exit.\n")
        while True:
            question = input("> ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue
            result = agent.query(question)
            _print_result(result)
            print()


def _print_result(result: dict) -> None:
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
