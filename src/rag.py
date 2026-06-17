from __future__ import annotations

import argparse

from .utils import INDEX_DIR, read_json
from .vectorize import TfidfIndex


def load_index(name: str) -> TfidfIndex:
    return TfidfIndex.from_dict(read_json(INDEX_DIR / f"{name}_index.json"))


def retrieve(query: str, top_k: int = 4) -> dict:
    query = query.strip()
    if not query:
        return {"query": query, "topics": [], "msg100": [], "chunks": []}
    return {
        "query": query,
        "topics": load_index("topic").search(query, top_k=top_k),
        "msg100": load_index("msg100").search(query, top_k=top_k),
        "chunks": load_index("chunk").search(query, top_k=top_k),
    }


def format_rag_answer(query: str, results: dict | None = None) -> str:
    results = results or retrieve(query)
    if not any(results[key] for key in ("topics", "msg100", "chunks")):
        return "I could not find strong matching evidence in the indexed conversations."

    lines = [f"Answer based on retrieved conversation evidence for: {query}", ""]
    if results["topics"]:
        lines.append("Relevant topic checkpoints:")
        for item in results["topics"][:3]:
            lines.append(f"- Topic {item['id']} messages {item['start_id']}-{item['end_id']} (score {item['score']}): {item['summary']}")
    if results["msg100"]:
        lines.append("\nRelevant 100-message checkpoints:")
        for item in results["msg100"][:2]:
            lines.append(f"- Checkpoint {item['id']} messages {item['start_id']}-{item['end_id']} (score {item['score']}): {item['summary']}")
    if results["chunks"]:
        lines.append("\nRelevant raw message chunks:")
        for item in results["chunks"][:2]:
            preview = item["text"].replace("\n", " ")[:500]
            lines.append(f"- Chunk {item['id']} messages {item['start_id']}-{item['end_id']} (score {item['score']}): {preview}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local RAG indexes.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    print(format_rag_answer(args.query, retrieve(args.query, args.top_k)))


if __name__ == "__main__":
    main()
