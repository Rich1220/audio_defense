#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Summarize judge labels and unsafe rate.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label-key", default="llamaguard_label")
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    counts = Counter(str(row.get(args.label_key, "missing")) for row in rows)
    total = len(rows)
    unsafe = counts.get("unsafe", 0)
    unsafe_rate = unsafe / total if total else 0.0

    lines = [
        "# Safety Judge Summary",
        "",
        f"- input: `{args.input}`",
        f"- label key: `{args.label_key}`",
        f"- rows: {total}",
        f"- unsafe: {unsafe}",
        f"- unsafe rate: {unsafe_rate:.4f}",
        "",
        "| label | count | rate |",
        "|---|---:|---:|",
    ]
    for label, count in sorted(counts.items()):
        rate = count / total if total else 0.0
        lines.append(f"| {label} | {count} | {rate:.4f} |")

    text = "\n".join(lines) + "\n"
    print(text)

    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print("[OK] wrote:", out_path)


if __name__ == "__main__":
    main()
