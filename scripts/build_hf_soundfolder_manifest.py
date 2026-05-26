#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
METADATA_NAMES = {"metadata.csv", "metadata.jsonl", "metadata.json"}
PROMPT_KEYS = (
    "prompt",
    "question",
    "instruction",
    "text",
    "query",
    "harmful_prompt",
    "goal",
)


def norm_path(path):
    return str(path).replace("\\", "/").lstrip("./")


def rel_to_base(path, base):
    path = norm_path(path)
    base = norm_path(base).rstrip("/")
    if base and path.startswith(base + "/"):
        return path[len(base) + 1 :]
    return path


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON metadata structure: {path}")


def load_metadata(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    return []


def metadata_audio_key(row):
    for key in ("file_name", "filename", "audio", "audio_path", "path", "file", "wav"):
        value = row.get(key)
        if value:
            return norm_path(value)
    return None


def prompt_from_metadata(row):
    for key in PROMPT_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Build a manifest from a Hugging Face soundfolder dataset path.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. tsinghua-ee/SACRED-Bench")
    parser.add_argument("--repo-subdir", default="", help="Folder inside the dataset repo, e.g. Multi-speaker_Dialogue/test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audio-dir", default="outputs/hf_soundfolder_audio")
    parser.add_argument("--download-audio", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--metadata", default="", help="Optional local metadata CSV/JSONL/JSON")
    parser.add_argument("--id-prefix", default="", help="Optional id prefix. Defaults to repo/subdir.")
    parser.add_argument("--prompt", default="", help="Fallback prompt used when metadata has no prompt-like field.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    repo_subdir = norm_path(args.repo_subdir).rstrip("/")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision="main",
    )
    if repo_subdir:
        files = [path for path in files if norm_path(path).startswith(repo_subdir + "/")]

    metadata_files = [
        path for path in files
        if Path(path).name in METADATA_NAMES
    ]
    audio_files = [
        path for path in files
        if Path(path).suffix.lower() in AUDIO_EXTENSIONS
    ]
    audio_files = sorted(audio_files)
    if args.limit > 0:
        audio_files = audio_files[: args.limit]

    metadata_rows = []
    if args.metadata:
        metadata_rows = load_metadata(args.metadata)
    elif metadata_files:
        meta_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=metadata_files[0],
            local_dir=args.audio_dir,
            local_files_only=args.local_files_only,
        )
        metadata_rows = load_metadata(meta_path)
        print("[INFO] loaded metadata:", metadata_files[0], "rows:", len(metadata_rows))

    meta_by_audio = {}
    for row in metadata_rows:
        audio_key = metadata_audio_key(row)
        if not audio_key:
            continue
        meta_by_audio[audio_key] = row
        meta_by_audio[norm_path(Path(audio_key).name)] = row

    id_prefix = args.id_prefix or norm_path(f"{args.repo_id}/{repo_subdir or 'root'}")
    rows = []
    for idx, repo_audio in enumerate(tqdm(audio_files)):
        rel_audio = rel_to_base(repo_audio, repo_subdir)
        meta = (
            meta_by_audio.get(norm_path(repo_audio))
            or meta_by_audio.get(norm_path(rel_audio))
            or meta_by_audio.get(norm_path(Path(repo_audio).name))
            or {}
        )

        local_audio = None
        if args.download_audio:
            local_audio = hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=repo_audio,
                local_dir=args.audio_dir,
                local_files_only=args.local_files_only,
            )

        prompt = prompt_from_metadata(meta) or args.prompt
        row = dict(meta)
        row.update(
            {
                "id": f"{id_prefix}/{idx}",
                "source_dataset": args.repo_id,
                "repo_subdir": repo_subdir,
                "repo_audio": repo_audio,
                "local_audio": local_audio,
                "prompt": prompt,
                "category": row.get("category") or row.get("scenario") or "unknown",
                "source": row.get("source") or "SACRED-Bench",
                "attack_type": row.get("attack_type") or row.get("attack") or "soundfolder",
            }
        )
        rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[OK] wrote:", out_path)
    print("[INFO] rows:", len(rows))
    print("[INFO] audio files:", len(audio_files))
    print("[INFO] metadata files:", metadata_files[:5])
    if rows and not any(row.get("prompt") for row in rows):
        print("[WARN] no prompt text was found. Llama Guard will judge with an empty prompt unless you provide --prompt or metadata.")


if __name__ == "__main__":
    main()
