from __future__ import annotations

import argparse
import json

from .utils import DATA_DIR, INDEX_DIR, read_json, read_jsonl, write_json


def validate_outputs() -> dict:
    messages = list(read_jsonl(DATA_DIR / "all_messages.jsonl"))
    topics = read_json(DATA_DIR / "topic_checkpoints.json")
    msg100 = read_json(DATA_DIR / "msg100_checkpoints.json")
    persona = read_json(DATA_DIR / "persona.json")
    index_stats = read_json(INDEX_DIR / "index_stats.json")

    topic_ranges_ok = all(
        topic["start_id"] <= topic["end_id"]
        and (idx == 0 or topics[idx - 1]["end_id"] < topic["start_id"])
        for idx, topic in enumerate(topics)
    )
    fixed_ranges_ok = all(
        checkpoint["end_id"] - checkpoint["start_id"] + 1 == 100 or checkpoint.get("partial")
        for checkpoint in msg100
    )
    persona_evidence_ok = all(
        persona[user]["message_count"] > 0
        and persona[user]["communication_style"]["avg_words_per_message"] > 0
        for user in ("User 1", "User 2")
    )

    report = {
        "messages": len(messages),
        "first_message_id": messages[0]["id"] if messages else None,
        "last_message_id": messages[-1]["id"] if messages else None,
        "topic_checkpoints": len(topics),
        "topic_ranges_ok": topic_ranges_ok,
        "msg100_checkpoints": len(msg100),
        "fixed_ranges_ok": fixed_ranges_ok,
        "persona_users": list(persona.keys()),
        "persona_evidence_ok": persona_evidence_ok,
        "index_stats": index_stats,
        "passed": topic_ranges_ok and fixed_ranges_ok and persona_evidence_ok and len(messages) == 191592,
    }
    write_json(DATA_DIR / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated project artifacts.")
    parser.parse_args()
    print(json.dumps(validate_outputs(), indent=2))


if __name__ == "__main__":
    main()
