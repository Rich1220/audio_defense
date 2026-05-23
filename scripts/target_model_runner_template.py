#!/usr/bin/env python3
"""
Template for running a new target model before hidden-state extraction.

Fill in `load_model_bundle` and `generate_one`. The output JSONL should preserve
the benchmark fields and add one response field, for example `my_model_response`.
Then run `judge_with_llamaguard.py` on that response field.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_model_bundle(model_name_or_path, device):
    raise NotImplementedError("Load your model, tokenizer, processor, etc.")


def generate_one(row, bundle, device):
    raise NotImplementedError("Return the target model response for one benchmark row.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--response-key", default="my_model_response")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        out_path.unlink(missing_ok=True)

    bundle = load_model_bundle(args.model, args.device)
    for row in tqdm(rows):
        out = dict(row)
        try:
            out[args.response_key] = generate_one(row, bundle, args.device)
            out["eval_error"] = None
        except Exception as exc:
            out[args.response_key] = ""
            out["eval_error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(out_path, out)


if __name__ == "__main__":
    main()
