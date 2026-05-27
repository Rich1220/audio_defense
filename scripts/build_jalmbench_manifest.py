#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DATASET = "AnonymousUser000/JALMBench"
PROMPT_KEYS = ("text", "original_text", "prompt", "query", "instruction")
SUBSET_FILES = {
    "ADiv": "HarmfulQuery/ADiv.parquet",
    "SSJ": "Audio_Originated_Jailbreak/SSJ.parquet",
}


def safe_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def prompt_from_row(row):
    for key in PROMPT_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
            if key not in {"array", "bytes"}
        }
    return str(value)


def materialize_audio(audio_obj, audio_dir, subset, idx, sample_id):
    if not audio_obj:
        return None

    if isinstance(audio_obj, str):
        return audio_obj

    if not isinstance(audio_obj, dict):
        return None

    path = audio_obj.get("path")
    if path and Path(path).exists():
        return str(path)

    array = audio_obj.get("array")
    sampling_rate = audio_obj.get("sampling_rate")
    if array is None or sampling_rate is None:
        return path

    out_dir = Path(audio_dir) / safe_name(subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{idx:06d}_{safe_name(sample_id)}.wav"
    sf.write(out_path, array, int(sampling_rate))
    return str(out_path)


def parse_subsets(value):
    if value.lower() in {"all", "both"}:
        return list(SUBSET_FILES)
    aliases = {name.lower(): name for name in SUBSET_FILES}
    subsets = []
    for part in value.split(","):
        key = part.strip()
        if not key:
            continue
        canonical = aliases.get(key.lower())
        if not canonical:
            available = ", ".join(sorted(SUBSET_FILES))
            raise ValueError(f"Unknown subset {key!r}. Known subsets: {available}, all.")
        subsets.append(canonical)
    if not subsets:
        raise ValueError("No subsets selected.")
    return subsets


def load_subset_rows(args, subset, parquet_file=""):
    parquet_name = parquet_file or SUBSET_FILES.get(subset)
    if not parquet_name:
        available = ", ".join(sorted(SUBSET_FILES))
        raise ValueError(f"Unknown subset {subset!r}. Known subsets: {available}.")
    parquet_path = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        filename=parquet_name,
        local_files_only=args.local_files_only,
    )
    table = pq.read_table(parquet_path)
    rows_in = table.to_pylist()

    rows = []
    kept = 0
    for idx, row in enumerate(tqdm(rows_in, desc=subset)):
        if args.limit > 0 and kept >= args.limit:
            break
        kept += 1

        sample_id = row.get("id", idx)
        audio_obj = row.get("audio")
        if args.save_audio:
            local_audio = materialize_audio(audio_obj, args.audio_dir, subset, idx, sample_id)
        elif isinstance(audio_obj, dict):
            local_audio = audio_obj.get("path")
        else:
            local_audio = audio_obj

        output = {
            key: jsonable(value)
            for key, value in row.items()
            if key != "audio"
        }
        output.update(
            {
                "id": f"jalmbench/{subset}/{args.split}/{sample_id}",
                "source_dataset": DATASET,
                "config": subset,
                "subset": subset,
                "split": args.split,
                "index": idx,
                "prompt": prompt_from_row(row),
                "original_prompt": row.get("original_text"),
                "jalmbench_text": row.get("text"),
                "local_audio": local_audio,
                "audio": local_audio,
                "attack_type": subset,
                "category": row.get("category") or subset,
                "source": row.get("source") or "JALMBench",
                "parquet_file": parquet_name,
            }
        )
        rows.append(output)
    return parquet_name, rows


def main():
    parser = argparse.ArgumentParser(description="Build a JALMBench JSONL manifest.")
    parser.add_argument(
        "--subset",
        required=True,
        help="JALMBench subset/config: ADiv, SSJ, all, or comma-separated values such as ADiv,SSJ.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audio-dir", default="outputs/jalmbench_audio")
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Materialize parquet audio arrays to wav files.",
    )
    parser.add_argument("--parquet-file", default="", help="Override the dataset parquet file path inside the HF repo.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)

    subsets = parse_subsets(args.subset)
    if args.parquet_file and len(subsets) != 1:
        raise ValueError("--parquet-file can only be used with a single --subset.")

    rows = []
    parquet_files = {}
    subset_counts = {}
    for subset in subsets:
        parquet_name, subset_rows = load_subset_rows(args, subset, args.parquet_file)
        parquet_files[subset] = parquet_name
        subset_counts[subset] = len(subset_rows)
        rows.extend(subset_rows)
    for global_index, row in enumerate(rows):
        row["global_index"] = global_index

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[OK] wrote:", out_path)
    print("[INFO] subsets:", subsets)
    print("[INFO] parquet files:", parquet_files)
    print("[INFO] split:", args.split)
    print("[INFO] rows:", len(rows))
    print("[INFO] subset rows:", subset_counts)
    if rows and not any(row.get("local_audio") for row in rows):
        print("[WARN] no local audio path was found. Rerun with --save-audio.")
    if rows and not any(row.get("prompt") for row in rows):
        print("[WARN] no prompt text was found. Llama Guard will judge with an empty prompt.")


if __name__ == "__main__":
    main()
