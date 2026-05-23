# 我自己跑實驗的流程

## 0. 簡單流程說明:
1. 建立環境
2. 下載/建立 AudioJailbreak benchmark manifest
3. 用Qwen2-Audio跑 response
4. 用Llama Guard judge unsafe rate
5. 抽 Qwen2-Audio hidden states
6. 跑hidden-state router defense
7. 看結果

## Step1-建立環境

### 1. 虛擬環境建立
```bash
cd ./defense/method

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```
### 2. 安裝套件
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
python -m pip install -r requirements-experiment.txt
```
### 3. 確認環境
```bash
python - <<'PY'
python - <<'PY'
import torch, transformers, datasets, librosa, sklearn, numpy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
PY
```

## Steps2 - 建立Manifest (從AudioJailbreak)
## 1. 建立
```bash
python scripts/build_audiojailbreak_manifest.py \
    --config Origin \
    --split origin \
    --limit 20 \
    --download-audio \
    --audio-dir outputs/audiojailbreak_audio \
    --out outputs/audiojailbreak_origin_smoke20_manifest.jsonl
```
## 2. 檢查
```bash
wc -l outputs/audiojailbreak_origin_smoke20_manifest.jsonl
head -n 1 outputs/audiojailbreak_origin_smoke20_manifest.jsonl
```

## Steps3 - 跑 Resopnse
```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
    --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
    --out outputs/audiojailbreak_origin_smoke20_qwen2audio_responses.jsonl \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --device cuda:0 \
    --prompt-mode safety \
    --overwrite
```
## Steps4 - Llama Guard Judge
 ```bash
 python scripts/judge_with_llamaguard.py \
    --input outputs/audiojailbreak_origin_smoke20_qwen2audio_responses.jsonl \
    --out outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
    --response-key qwen2_audio_response \
    --prompt-key prompt \
    --model meta-llama/Llama-Guard-3-8B \
    --device cuda:0 \
    --overwrite
```
### 統計 labels
```bash
python scripts/summarize_judge_labels.py \
    --input outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
    --out-md outputs/audiojailbreak_origin_smoke20_qwen2audio_safety_summary.md
```

## Steps5 - 抽 Qwen2-Audio hidden states (假設用8的倍數)
```bash
python scripts/extract_qwen2_audio_hidden.py \
    --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
    --responses outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
    --out-npz outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
    --out-meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --device cuda:0 \
    --layers 0,8,16,24,32
```
### 驗證 Hidden Feature Schema
```bash
python scripts/validate_features.py \
    --features outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
    --meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl
```

## Steps6 - 跑Defense Pipeline
```bash
 FEATURES=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
  META=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
  OUT_DIR=outputs/audiojailbreak_origin_smoke20_qwen2audio_defense \
  PYTHON=python \
  SPLIT_MODE=random \
  OBJECTIVE=f1 \
  bash run_hidden_router_pipeline.sh
```
## Steps7 - 看結果 
```bash
 cat outputs/audiojailbreak_origin_smoke20_qwen2audio_defense/single_layer_probe/probe_summary.md

  cat outputs/audiojailbreak_origin_smoke20_qwen2audio_defense/single_router_defense/defense_simulation_summary.md

  cat outputs/audiojailbreak_origin_smoke20_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
```