#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import traceback

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_guard_output(text):
    text = (text or "").strip().lower()
    if text.startswith("unsafe"):
        return "unsafe"
    if text.startswith("safe"):
        return "safe"
    if re.search(r"\bunsafe\b", text[:100]):
        return "unsafe"
    if re.search(r"\bsafe\b", text[:100]):
        return "safe"
    return "unknown"


def row_key(row):
    return row.get("id") or row.get("sample_id") or row.get("index")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Judge model responses with Llama Guard.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--response-key", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--model", default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists() and not args.overwrite:
        for row in load_jsonl(out_path):
            done.add(row_key(row))

    rows = [row for row in load_jsonl(args.input) if row_key(row) not in done]
    print("[INFO] rows to judge:", len(rows))
    print("[INFO] loading judge:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    model.eval()

    counts = Counter()
    mode = "w" if args.overwrite else "a"
    with out_path.open(mode, encoding="utf-8") as fout:
        for row in tqdm(rows):
            prompt = str(row.get(args.prompt_key, ""))
            response = str(row.get(args.response_key, ""))

            if not response.strip():
                row["llamaguard_label"] = "empty_response"
                row["llamaguard_raw"] = ""
                row["llamaguard_error"] = None
                counts["empty_response"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                )
                if isinstance(encoded, torch.Tensor):
                    model_inputs = {"input_ids": encoded.to(args.device)}
                else:
                    model_inputs = {
                        key: (value.to(args.device) if hasattr(value, "to") else value)
                        for key, value in encoded.items()
                    }
                with torch.inference_mode():
                    output = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0][model_inputs["input_ids"].shape[-1] :]
                raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
                row["llamaguard_label"] = parse_guard_output(raw)
                row["llamaguard_raw"] = raw
                row["llamaguard_error"] = None
            except Exception as exc:
                row["llamaguard_label"] = "judge_error"
                row["llamaguard_raw"] = ""
                row["llamaguard_error"] = f"{type(exc).__name__}: {exc!r}"
                row["llamaguard_traceback"] = traceback.format_exc()

            counts[row["llamaguard_label"]] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    print("[COUNTS]", dict(counts))
    print("[OK] wrote:", out_path)


if __name__ == "__main__":
    main()
