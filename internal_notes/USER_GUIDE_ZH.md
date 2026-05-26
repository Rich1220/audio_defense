# Hidden-State Router Defense 使用指南

這個資料夾是一套完整的 audio jailbreak defense 實驗流程。目標是用 audio-language model 的 hidden states 訓練一個輕量 router，判斷模型輸出是否可能 unsafe，並在離線實驗中模擬把高風險樣本 route 到更安全的路徑。

核心流程：

```text
建立 benchmark manifest
  -> 跑 target audio-language model 產生 response
  -> 用 safety judge 標註 safe / unsafe
  -> 抽 target model hidden states
  -> 訓練 hidden-state router
  -> 看 defense simulation 結果
```

LlamaGuard 只用於離線產生訓練與評估 label。真正的 router 輸入是 target model 的 hidden states，不是 LlamaGuard。

## 1. 檔案架構

```text
defense_method/
├── README.md
├── METHOD_DETAILS.md
├── EXPERIMENT_PIPELINE.md
├── ADAPTER_GUIDE.md
├── USER_GUIDE_ZH.md
├── MY_README.md
├── requirements.txt
├── requirements-experiment.txt
├── run_hidden_router_pipeline.sh
├── scripts/
└── outputs/
```

### 主要文件

| 檔案 | 用途 |
|---|---|
| `README.md` | 英文總覽，說明必要輸入格式、主要 scripts、輸出結果要看哪裡。 |
| `METHOD_DETAILS.md` | 方法細節，包含 hidden pooling、logistic regression、threshold、AUROC/AUPRC、defense metrics。 |
| `EXPERIMENT_PIPELINE.md` | 從零開始跑完整實驗的英文 command 流程。 |
| `ADAPTER_GUIDE.md` | 如果要換新模型或新 benchmark，要如何產生相同格式的 hidden features。 |
| `MY_README.md` | 原本的個人實驗筆記。 |
| `USER_GUIDE_ZH.md` | 中文使用指南，也就是本文件。 |

### scripts 目錄

| 檔案 | 用途 |
|---|---|
| `build_audiojailbreak_manifest.py` | 建立 AudioJailbreak manifest。 |
| `build_hf_soundfolder_manifest.py` | 建立 Hugging Face soundfolder-style dataset manifest，例如 SACRED-Bench。 |
| `build_jalmbench_manifest.py` | 建立 JALMBench manifest，必要時把音訊存成 wav。 |
| `run_qwen2_audio_audiojailbreak.py` | 用 Qwen2-Audio 讀音訊並產生 response。雖然檔名有 audiojailbreak，但也可用於 JALMBench/SACRED 這種同格式 manifest。 |
| `judge_with_llamaguard.py` | 用 LlamaGuard 判斷 response 是 safe 或 unsafe。 |
| `summarize_judge_labels.py` | 統計 judge label 分布與 unsafe rate。 |
| `extract_qwen2_audio_hidden.py` | 抽 Qwen2-Audio hidden states，輸出 `.npz` 和 `.jsonl`。 |
| `validate_features.py` | 檢查 hidden feature 檔案格式是否正確。 |
| `train_hidden_probes.py` | 掃每個 layer/pooling，訓練 single-layer linear probe。 |
| `simulate_hidden_router_defense.py` | 用單一路由器做 defense simulation。 |
| `train_auto_layer_router.py` | 自動選 layer 並 ensemble，多數正式結果建議看這個。 |
| `train_category_transfer_router.py` | 訓練在一個 harmful category，測試能否 transfer 到其他 category。 |
| `target_model_runner_template.py` | 新 target model 的 response runner 模板。 |
| `extractor_template.py` | 新 target model 的 hidden extractor 模板。 |
| `layer_utils.py` | 處理 real layer id 和 sparse layer mapping。 |

### outputs 目錄

`outputs/` 會存所有中間檔和結果。常見命名如下：

```text
{benchmark}_manifest.jsonl
{benchmark}_qwen2audio_responses.jsonl
{benchmark}_qwen2audio_llamaguard.jsonl
{benchmark}_qwen2audio_safety_summary.md
{benchmark}_qwen2audio_hidden_features.npz
{benchmark}_qwen2audio_hidden_meta.jsonl
{benchmark}_qwen2audio_defense/
```

例如：

```text
outputs/jalmbench_adiv_smoke20_manifest.jsonl
outputs/jalmbench_adiv_smoke20_qwen2audio_responses.jsonl
outputs/jalmbench_adiv_smoke20_qwen2audio_llamaguard.jsonl
outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_features.npz
outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_meta.jsonl
outputs/jalmbench_adiv_smoke20_qwen2audio_defense/
```

## 2. 環境設定

進入資料夾：

```bash
cd /path/to/audio_emotion_jailbreak/defense_method
```

建立並啟用 virtual environment：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

安裝套件：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -r requirements-experiment.txt
```

確認環境：

```bash
python - <<'PY'
import torch, transformers, datasets, librosa, sklearn, numpy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
PY
```

如果需要指定 GPU，請依照自己的機器環境設定。例如有些人會用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

也有人會直接在 script 參數裡指定：

```bash
--device cuda:0
```

重點是：`--device` 要對應到目前 process 看得到的 GPU。若設定了 `CUDA_VISIBLE_DEVICES`，程式內的 `cuda:0` 代表「可見 GPU 裡的第 0 張」，不一定是實體編號 0。

## 3. 一次完整實驗會產生什麼

一個完整 benchmark 會有以下階段。

### Step 1: 建立 manifest

manifest 是統一格式的 JSONL，每一行是一筆音訊樣本，通常包含：

```text
id
prompt
local_audio
category
source
attack_type
```

### Step 2: 跑 target model response

目前內建 target model 是 Qwen2-Audio：

```text
Qwen/Qwen2-Audio-7B-Instruct
```

輸出會新增：

```text
qwen2_audio_response
eval_error
```

### Step 3: LlamaGuard judge

用 response 和 prompt 判斷 safe/unsafe，輸出會新增：

```text
llamaguard_label
llamaguard_raw
llamaguard_error
```

router label 規則：

```text
llamaguard_label == "unsafe" -> label 1
其他                         -> label 0
```

### Step 4: 抽 hidden states

輸出兩個核心檔案：

```text
hidden_features.npz
hidden_meta.jsonl
```

`hidden_features.npz` 主要包含：

```text
hidden_last          [N, L, D]
hidden_mean_context  [N, L, D]
hidden_audio_tokens  [N, L, D]
scores               [N, K]
labels               [N]
ids                  [N]
```

三種 pooling：

| pooling | 意義 |
|---|---|
| `last` | 最後一個有效 context token 的 hidden vector。 |
| `mean_context` | 所有有效 context tokens 的平均。 |
| `audio_tokens` | audio token 位置的平均；若找不到 audio token，會 fallback 到 context。 |

### Step 5: 跑 defense

`run_hidden_router_pipeline.sh` 會依序跑：

```text
validate_features.py
train_hidden_probes.py
simulate_hidden_router_defense.py
train_auto_layer_router.py
```

主要結果在：

```text
{OUT_DIR}/single_layer_probe/probe_summary.md
{OUT_DIR}/single_router_defense/defense_simulation_summary.md
{OUT_DIR}/auto_layer_router/auto_layer_router_summary.md
```

通常最重要的是：

```text
auto_layer_router/auto_layer_router_summary.md
```

## 4. 跑 JALMBench

JALMBench 目前支援兩個 subset：

```text
ADiv
SSJ
```

建議第一次先跑 smoke test，例如 20 筆。

### 4.1 建立 manifest

```bash
python scripts/build_jalmbench_manifest.py \
  --subset ADiv \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_adiv_smoke20_manifest.jsonl

python scripts/build_jalmbench_manifest.py \
  --subset SSJ \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_ssj_smoke20_manifest.jsonl
```

### 4.2 跑 Qwen2-Audio response

```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/jalmbench_adiv_smoke20_manifest.jsonl \
  --out outputs/jalmbench_adiv_smoke20_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite

python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/jalmbench_ssj_smoke20_manifest.jsonl \
  --out outputs/jalmbench_ssj_smoke20_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite
```

### 4.3 LlamaGuard judge

```bash
python scripts/judge_with_llamaguard.py \
  --input outputs/jalmbench_adiv_smoke20_qwen2audio_responses.jsonl \
  --out outputs/jalmbench_adiv_smoke20_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite

python scripts/judge_with_llamaguard.py \
  --input outputs/jalmbench_ssj_smoke20_qwen2audio_responses.jsonl \
  --out outputs/jalmbench_ssj_smoke20_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite
```

統計 labels：

```bash
python scripts/summarize_judge_labels.py \
  --input outputs/jalmbench_adiv_smoke20_qwen2audio_llamaguard.jsonl \
  --out-md outputs/jalmbench_adiv_smoke20_qwen2audio_safety_summary.md

python scripts/summarize_judge_labels.py \
  --input outputs/jalmbench_ssj_smoke20_qwen2audio_llamaguard.jsonl \
  --out-md outputs/jalmbench_ssj_smoke20_qwen2audio_safety_summary.md
```

如果 label 全部都是 safe，router 不能訓練。這時請提高 `--limit`，例如 100 或 full run。

### 4.4 抽 hidden states

全 layer：

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/jalmbench_adiv_smoke20_manifest.jsonl \
  --responses outputs/jalmbench_adiv_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_features.npz \
  --out-meta outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0

python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/jalmbench_ssj_smoke20_manifest.jsonl \
  --responses outputs/jalmbench_ssj_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/jalmbench_ssj_smoke20_qwen2audio_hidden_features.npz \
  --out-meta outputs/jalmbench_ssj_smoke20_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0
```

如果想省空間和時間，可以只抽部分 layers：

```bash
--layers 0,8,16,24,32
```

### 4.5 驗證 hidden feature

```bash
python scripts/validate_features.py \
  --features outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_features.npz \
  --meta outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_meta.jsonl
```

確認輸出裡 labels 至少有兩類：

```text
labels: {0: ..., 1: ...}
```

如果只有：

```text
labels: {0: ...}
```

代表全部 safe，不能訓練 binary router。

### 4.6 跑 defense

ADiv：

```bash
FEATURES=outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_features.npz \
META=outputs/jalmbench_adiv_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/jalmbench_adiv_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=high_recall \
bash run_hidden_router_pipeline.sh
```

SSJ：

```bash
FEATURES=outputs/jalmbench_ssj_smoke20_qwen2audio_hidden_features.npz \
META=outputs/jalmbench_ssj_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/jalmbench_ssj_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=high_recall \
bash run_hidden_router_pipeline.sh
```

看結果：

```bash
cat outputs/jalmbench_adiv_smoke20_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
cat outputs/jalmbench_ssj_smoke20_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
```

## 5. 跑 AudioJailbreak

建立 manifest：

```bash
python scripts/build_audiojailbreak_manifest.py \
  --config Origin \
  --split origin \
  --limit 20 \
  --download-audio \
  --audio-dir outputs/audiojailbreak_audio \
  --out outputs/audiojailbreak_origin_smoke20_manifest.jsonl
```

後面的 response、judge、hidden extraction、defense 流程與 JALMBench 相同，只要把檔名換成 `audiojailbreak_origin_smoke20`。

## 6. 跑 SACRED-Bench

建立 manifest：

```bash
python scripts/build_hf_soundfolder_manifest.py \
  --repo-id tsinghua-ee/SACRED-Bench \
  --repo-subdir Multi-speaker_Dialogue/test \
  --limit 20 \
  --download-audio \
  --audio-dir outputs/sacred_multispeaker_audio \
  --out outputs/sacred_multispeaker_smoke20_manifest.jsonl
```

後續流程同上。

## 7. 結果怎麼讀

### safety summary

```text
outputs/{name}_qwen2audio_safety_summary.md
```

看：

```text
rows
unsafe
unsafe rate
```

如果 unsafe 是 0，defense 不能訓練。

### single layer probe

```text
outputs/{name}_qwen2audio_defense/single_layer_probe/probe_summary.md
```

看哪一個 pooling/layer 最能分 safe vs unsafe：

```text
pooling
layer
AUROC
AUPRC
```

### single router defense

```text
outputs/{name}_qwen2audio_defense/single_router_defense/defense_simulation_summary.md
```

這是單一 feature/router 的 defense simulation。

### auto-layer router

```text
outputs/{name}_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
```

正式報告通常優先看這個。重要欄位：

| 欄位 | 意義 |
|---|---|
| `AUROC` | router 對 unsafe risk 的排序能力。 |
| `AUPRC` | unsafe 類別的 precision-recall 表現。unsafe 少時很重要。 |
| `Unsafe Before` | route 前 unsafe rate。 |
| `Unsafe After` | simulation 中 route 後剩下的 unsafe rate。 |
| `Reduction` | 抓到多少比例 unsafe。 |
| `Route Rate` | 有多少樣本被送去安全路徑。 |
| `False Route` | safe 樣本被誤 route 的比例。 |
| `Recall` | unsafe 被抓到的比例。 |
| `Threshold` | validation set 選出的 route threshold。 |

## 8. 常見問題

### 問題 1: `labels: {0: 20}` 然後 LogisticRegression 報錯

原因：所有樣本都被 judge 成 safe，沒有 unsafe class。

解法：提高樣本數，例如：

```bash
--limit 100
```

或直接跑 full：

```bash
--limit 0
```

### 問題 2: CUDA out of memory

可以嘗試：

```bash
--layers 0,8,16,24,32
```

或降低 batch/一次只跑一個 subset。這份 pipeline 主要是一筆一筆處理，所以常見瓶頸是模型本身大小，而不是 batch size。

### 問題 3: Hugging Face gated model 不能下載

先登入：

```bash
hf auth login
```

LlamaGuard 和部分模型可能需要權限。

### 問題 4: 已經跑到一半，不想重跑全部

多數 response/judge script 都是 append JSONL。若沒有加 `--overwrite`，通常會跳過已完成 rows。若想重跑，才加：

```bash
--overwrite
```

### 問題 5: 不知道現在要看哪個結果

最簡單看這三個：

```bash
cat outputs/{name}_qwen2audio_safety_summary.md
cat outputs/{name}_qwen2audio_defense/single_layer_probe/probe_summary.md
cat outputs/{name}_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
```

## 9. 建議命名規則

建議用：

```text
{benchmark}_{subset}_{size}_{model}_{stage}
```

例子：

```text
jalmbench_adiv_smoke20_qwen2audio_responses.jsonl
jalmbench_adiv_p100_qwen2audio_llamaguard.jsonl
jalmbench_adiv_full_qwen2audio_hidden_features.npz
jalmbench_adiv_full_qwen2audio_defense/
```

這樣可以清楚知道：

```text
benchmark = jalmbench
subset    = adiv
size      = smoke20 / p100 / full
model     = qwen2audio
stage     = responses / llamaguard / hidden / defense
```

## 10. 目前的 defense claim

目前完成：

```text
Hidden-state router 能從 target audio-language model 的內部表示中讀出 unsafe risk signal，
並在 offline defense simulation 中降低 unsafe rate。
```

還沒完成：

```text
還不是 production-ready defense。
router 還沒辦法阻止所有 jailbreak。
```

因為目前 simulation 假設 routed samples 會被安全處理，真正部署還需要接實際的 fallback safety path。
