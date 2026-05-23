#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DATASET = "MBZUAI/AudioJailbreak"


def main():
    parser = argparse.ArgumentParser(description="Build an AudioJailbreak JSONL manifest.")
    parser.add_argument("--config", default="Origin")
    parser.add_argument("--split", default="origin")
    parser.add_argument("--limit", type=int, default=50, help="0 means no limit.")
    parser.add_argument(
        "--target-model-filter",
        default="",
        help="Keep rows whose target_model contains this text. Empty keeps all rows.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--download-audio", action="store_true")
    parser.add_argument("--audio-dir", default="outputs/audiojailbreak_audio")
    args = parser.parse_args()

    ds = load_dataset(DATASET, args.config, split=args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    kept = 0
    target_model_counts = Counter(str(row.get("target_model", "")) for row in ds)
    for i, row in enumerate(tqdm(ds)):
        if args.target_model_filter:
            target_model = str(row.get("target_model", "")).lower()
            if args.target_model_filter.lower() not in target_model:
                continue
        if args.limit > 0 and kept >= args.limit:
            break
        kept += 1

        speech_path = row.get("speech_path")
        local_audio = None
        if args.download_audio and speech_path:
            repo_path = speech_path[2:] if speech_path.startswith("./") else speech_path
            local_audio = hf_hub_download(
                repo_id=DATASET,
                repo_type="dataset",
                filename=repo_path,
                local_dir=args.audio_dir,
                local_dir_use_symlinks=False,
            )

        rows.append(
            {
                "id": f"audiojailbreak/{args.config}/{args.split}/{i}",
                "source_dataset": DATASET,
                "config": args.config,
                "split": args.split,
                "index": row.get("index", i),
                "goal": row.get("goal"),
                "category": row.get("category"),
                "prompt": row.get("prompt"),
                "reference_response": row.get("response"),
                "speech_path": speech_path,
                "local_audio": local_audio,
                "attack_type": row.get("attack_type"),
                "target_model": row.get("target_model"),
                "platform": row.get("platform"),
                "source": row.get("source"),
                "behavior": row.get("behavior"),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[OK] wrote:", out_path)
    print("[INFO] rows:", len(rows))
    if not rows:
        print("[WARN] no rows were kept")
        if args.target_model_filter:
            print("[WARN] target model filter:", args.target_model_filter)
            print("[WARN] available target_model values:")
            for value, count in target_model_counts.most_common(20):
                print(f"  {value!r}: {count}")
            print("[HINT] rerun without --target-model-filter, or set it to an available value.")


if __name__ == "__main__":
    main()
