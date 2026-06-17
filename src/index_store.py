from __future__ import annotations

import argparse
import json

from .summarize import describe_segment
from .utils import DATA_DIR, INDEX_DIR, ensure_dirs, read_json, read_jsonl, write_json
from .vectorize import TfidfIndex


def build_message_chunks(messages: list[dict], chunk_size: int = 40, stride: int = 20) -> list[dict]:
    chunks = []
    for start in range(0, len(messages), stride):
        chunk = messages[start : start + chunk_size]
        if not chunk:
            continue
        details = describe_segment(chunk)
        text = "\n".join(f"{m['speaker']}: {m['text']}" for m in chunk)
        chunks.append(
            {
                "chunk_id": len(chunks),
                "start_id": chunk[0]["id"],
                "end_id": chunk[-1]["id"],
                "text": text,
                "summary": details["summary"],
                "keywords": details["keywords"],
            }
        )
        if start + chunk_size >= len(messages):
            break
    return chunks


def build_indexes() -> dict:
    ensure_dirs()
    messages = list(read_jsonl(DATA_DIR / "all_messages.jsonl"))
    topics = read_json(DATA_DIR / "topic_checkpoints.json")
    msg100 = read_json(DATA_DIR / "msg100_checkpoints.json")
    chunks = build_message_chunks(messages)
    write_json(DATA_DIR / "message_chunks.json", chunks)

    topic_docs = [
        {
            "kind": "topic",
            "id": topic["topic_id"],
            "start_id": topic["start_id"],
            "end_id": topic["end_id"],
            "text": f"{topic['summary']} {' '.join(topic.get('keywords', []))}",
            "summary": topic["summary"],
        }
        for topic in topics
    ]
    fixed_docs = [
        {
            "kind": "msg100",
            "id": checkpoint["checkpoint_id"],
            "start_id": checkpoint["start_id"],
            "end_id": checkpoint["end_id"],
            "text": f"{checkpoint['summary']} {' '.join(checkpoint.get('keywords', []))}",
            "summary": checkpoint["summary"],
        }
        for checkpoint in msg100
    ]
    chunk_docs = [
        {
            "kind": "chunk",
            "id": chunk["chunk_id"],
            "start_id": chunk["start_id"],
            "end_id": chunk["end_id"],
            "text": chunk["text"],
            "summary": chunk["summary"],
        }
        for chunk in chunks
    ]

    write_json(INDEX_DIR / "topic_index.json", TfidfIndex().fit(topic_docs).to_dict())
    write_json(INDEX_DIR / "msg100_index.json", TfidfIndex().fit(fixed_docs).to_dict())
    write_json(INDEX_DIR / "chunk_index.json", TfidfIndex().fit(chunk_docs).to_dict())

    stats = {"topic_docs": len(topic_docs), "msg100_docs": len(fixed_docs), "chunk_docs": len(chunk_docs)}
    write_json(INDEX_DIR / "index_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local TF-IDF indexes for RAG retrieval.")
    parser.parse_args()
    print(json.dumps(build_indexes(), indent=2))


if __name__ == "__main__":
    main()
