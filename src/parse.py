from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .utils import DATA_DIR, ensure_dirs, write_json


MESSAGE_RE = re.compile(r"^(User [12]):\s*(.*)$")


def parse_csv(input_path: Path, output_path: Path = DATA_DIR / "all_messages.jsonl") -> dict:
    ensure_dirs()
    stats = {"rows": 0, "messages": 0, "User 1": 0, "User 2": 0, "skipped_lines": 0}
    global_id = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_handle, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        reader = csv.reader(csv_handle)
        for day_idx, row in enumerate(reader):
            if not row:
                continue
            stats["rows"] += 1
            turn_in_day = 0
            for raw_line in row[0].splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                match = MESSAGE_RE.match(line)
                if not match:
                    stats["skipped_lines"] += 1
                    continue
                speaker, text = match.groups()
                message = {
                    "id": global_id,
                    "day": day_idx,
                    "turn_in_day": turn_in_day,
                    "speaker": speaker,
                    "text": text.strip(),
                }
                out.write(json.dumps(message, ensure_ascii=False) + "\n")
                stats["messages"] += 1
                stats[speaker] += 1
                global_id += 1
                turn_in_day += 1

    write_json(DATA_DIR / "parse_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse single-column conversation CSV into JSONL messages.")
    parser.add_argument("--input", required=True, help="Path to conversations.csv")
    parser.add_argument("--output", default=str(DATA_DIR / "all_messages.jsonl"))
    args = parser.parse_args()
    stats = parse_csv(Path(args.input), Path(args.output))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
