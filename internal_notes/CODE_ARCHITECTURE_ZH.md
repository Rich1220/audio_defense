# 程式碼架構整理建議

這份文件說明 `defense_method/` 目前的程式碼責任分工，以及建議如何把不同功能拆開，讓後續使用、維護、換模型、換 benchmark 更清楚。

## 1. 現況

目前架構是 script-oriented：

```text
scripts/
├── build_*_manifest.py
├── run_qwen2_audio_audiojailbreak.py
├── judge_with_llamaguard.py
├── extract_qwen2_audio_hidden.py
├── validate_features.py
├── train_hidden_probes.py
├── simulate_hidden_router_defense.py
├── train_auto_layer_router.py
├── train_category_transfer_router.py
├── target_model_runner_template.py
├── extractor_template.py
└── layer_utils.py
```

優點：

- 每個 script 可以單獨執行。
- 從 command line 跑實驗很直覺。
- 初期開發快，容易 debug 某一步。

缺點：

- `train_hidden_probes.py`、`simulate_hidden_router_defense.py`、`train_auto_layer_router.py`、`train_category_transfer_router.py` 裡面重複了很多功能，例如：
  - `load_jsonl`
  - `auroc`
  - `auprc`
  - stratified split
  - heldout split
  - logistic regression training
  - threshold metrics
- CLI 邏輯、資料處理、模型訓練、繪圖、summary 輸出混在同一個 script。
- 新人比較難一眼看出「核心方法」在哪裡。
- 要換模型或 benchmark 時，容易不知道該改哪個 script。

## 2. 建議目標架構

建議保留 `scripts/` 作為 command line entrypoints，但把核心邏輯拆到一個 package，例如：

```text
defense_method/
├── hidden_router/
│   ├── __init__.py
│   ├── io.py
│   ├── schema.py
│   ├── metrics.py
│   ├── splits.py
│   ├── layers.py
│   ├── probes.py
│   ├── thresholds.py
│   ├── defense.py
│   ├── plotting.py
│   ├── summaries.py
│   ├── manifests/
│   │   ├── __init__.py
│   │   ├── audiojailbreak.py
│   │   ├── jalmbench.py
│   │   └── hf_soundfolder.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── qwen2_audio.py
│   └── extraction/
│       ├── __init__.py
│       ├── pooling.py
│       └── qwen2_audio_hidden.py
├── scripts/
│   ├── build_audiojailbreak_manifest.py
│   ├── build_jalmbench_manifest.py
│   ├── run_qwen2_audio_audiojailbreak.py
│   ├── extract_qwen2_audio_hidden.py
│   ├── train_hidden_probes.py
│   ├── simulate_hidden_router_defense.py
│   └── train_auto_layer_router.py
└── run_hidden_router_pipeline.sh
```

核心原則：

```text
scripts/ 只負責 argparse + 呼叫 library function
hidden_router/ 才放真正邏輯
```

## 3. 各模組責任

### `hidden_router/io.py`

負責所有檔案讀寫。

應放：

```python
load_jsonl(path)
write_jsonl(path, rows)
append_jsonl(path, row)
load_features(path)
write_features(path, arrays)
```

好處：

- 不用每個 script 都自己寫一次 JSONL。
- 未來如果要支援 parquet/csv，也集中改這裡。

### `hidden_router/schema.py`

負責 hidden feature schema 驗證。

應放：

```python
REQUIRED_ARRAYS
validate_feature_arrays(data, meta)
valid_indices(meta)
label_counts(labels)
```

目前 `validate_features.py` 裡的核心邏輯可以搬到這裡。

### `hidden_router/metrics.py`

負責共用 metrics。

應放：

```python
auroc(y, scores)
auprc(y, scores)
confusion_counts(y, pred)
classification_metrics(y, pred)
```

現在 `auroc` 和 `auprc` 在多個 scripts 重複，應該只保留一份。

### `hidden_router/splits.py`

負責資料切分。

應放：

```python
stratified_random_split(y, train_frac, seed)
train_val_split(train_idx, y, fit_frac, seed)
heldout_splits(meta, key, y, min_test_pos, min_train_pos)
category_transfer_splits(meta, y, ...)
```

好處：

- random/source/category split 規則統一。
- 不同實驗不會不小心用到不同 split 邏輯。

### `hidden_router/layers.py`

負責 layer id 與 layer region。

目前已有：

```text
scripts/layer_utils.py
```

建議搬成：

```text
hidden_router/layers.py
```

應放：

```python
layer_indices_for(data, pool_key)
layer_positions(data, pool_key)
selected_position_layers(data, pool_key)
layer_regions(data, pool_key, regions)
```

### `hidden_router/probes.py`

負責 logistic regression probe。

應放：

```python
train_logistic_probe(x_train, y_train, seed)
predict_probe(model, x)
fit_predict_probe(x_train, y_train, x_test, seed)
build_feature_candidates(data, valid_idx)
sweep_layer_probes(data, y, train_idx, test_idx)
```

這是核心 router model，不應該散在多個 script 裡。

### `hidden_router/thresholds.py`

負責 threshold selection。

應放：

```python
threshold_metrics(y, scores, threshold)
choose_threshold(y_val, p_val, objective)
threshold_for_route_rate(scores, route_rate)
```

支援 objectives：

```text
f1
high_recall
low_route
```

### `hidden_router/defense.py`

負責 defense simulation。

應放：

```python
evaluate_defense(y_test, p_test, threshold)
run_single_router_defense(...)
run_auto_layer_router(...)
run_category_transfer(...)
```

重點是把「router 怎麼 route」和「route 後 metrics 怎麼算」集中管理。

### `hidden_router/plotting.py`

負責所有圖表。

應放：

```python
plot_layerwise_auroc(...)
plot_route_tradeoff(...)
plot_before_after_unsafe_rate(...)
plot_selected_layers(...)
plot_heatmap(...)
```

這樣訓練邏輯不會被 matplotlib code 混雜。

### `hidden_router/summaries.py`

負責 markdown summary。

應放：

```python
write_probe_summary(...)
write_defense_summary(...)
write_auto_layer_summary(...)
write_category_transfer_summary(...)
```

好處：

- result JSON 和 markdown report 的格式統一。
- 要改報告欄位時不用翻好幾個 script。

## 4. Benchmark 相關模組

建議放在：

```text
hidden_router/manifests/
```

### `manifests/audiojailbreak.py`

負責 AudioJailbreak manifest。

對應目前：

```text
scripts/build_audiojailbreak_manifest.py
```

### `manifests/jalmbench.py`

負責 JALMBench manifest。

對應目前：

```text
scripts/build_jalmbench_manifest.py
```

### `manifests/hf_soundfolder.py`

負責 Hugging Face soundfolder dataset。

對應目前：

```text
scripts/build_hf_soundfolder_manifest.py
```

## 5. Model 相關模組

建議放在：

```text
hidden_router/models/
```

### `models/qwen2_audio.py`

負責 Qwen2-Audio 共用邏輯：

```python
load_qwen2_audio(model_name, device, dtype, local_files_only)
build_qwen2_audio_inputs(processor, audio_path, prompt, device)
generate_qwen2_audio_response(...)
```

目前 `run_qwen2_audio_audiojailbreak.py` 和 `extract_qwen2_audio_hidden.py` 都有重複的 model loading / input building，可以抽到這裡。

## 6. Hidden Extraction 相關模組

建議放在：

```text
hidden_router/extraction/
```

### `extraction/pooling.py`

負責 hidden state pooling：

```python
pool_last(hidden_states, attention_mask)
pool_mean_context(hidden_states, attention_mask)
pool_audio_tokens(hidden_states, input_ids, tokenizer)
```

### `extraction/qwen2_audio_hidden.py`

負責 Qwen2-Audio hidden extraction：

```python
extract_qwen2_hidden_one(row, model_bundle, options)
extract_qwen2_hidden_dataset(...)
```

## 7. CLI script 應該長什麼樣子

重構後，script 應該很薄。例如：

```python
#!/usr/bin/env python3
import argparse

from hidden_router.manifests.jalmbench import build_jalmbench_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audio-dir", default="outputs/jalmbench_audio")
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()

    build_jalmbench_manifest(args)


if __name__ == "__main__":
    main()
```

也就是：

```text
CLI script = 解析參數 + 呼叫 hidden_router 裡的 function
```

## 8. 建議重構順序

不要一次大改。建議照這個順序，風險最低。

### Phase 1: 抽共用工具

先新增：

```text
hidden_router/io.py
hidden_router/metrics.py
hidden_router/splits.py
hidden_router/layers.py
hidden_router/thresholds.py
```

然後把重複 function 搬進去。

優先搬：

```text
load_jsonl
auroc
auprc
stratified split
heldout split
threshold_metrics
choose_threshold
layer_utils
```

### Phase 2: 抽 probe 和 defense 核心

新增：

```text
hidden_router/probes.py
hidden_router/defense.py
```

把 logistic regression training、predict、candidate building、defense metrics 搬進去。

### Phase 3: 抽 plotting 和 summaries

新增：

```text
hidden_router/plotting.py
hidden_router/summaries.py
```

把 matplotlib 和 markdown report 從訓練 script 裡移出。

### Phase 4: 抽 benchmark/model adapter

新增：

```text
hidden_router/manifests/
hidden_router/models/
hidden_router/extraction/
```

把 benchmark 建立、Qwen2-Audio runner、hidden extractor 拆乾淨。

## 9. 最終使用者介面

重構後，使用者仍然可以用原本 scripts：

```bash
python scripts/build_jalmbench_manifest.py ...
python scripts/run_qwen2_audio_audiojailbreak.py ...
python scripts/judge_with_llamaguard.py ...
python scripts/extract_qwen2_audio_hidden.py ...
bash run_hidden_router_pipeline.sh
```

也可以提供更高階的一鍵 CLI，例如：

```bash
python -m hidden_router.cli.run_experiment \
  --benchmark jalmbench \
  --subset ADiv \
  --limit 100 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --out-prefix outputs/jalmbench_adiv_p100_qwen2audio
```

但建議先不要急著做一鍵 CLI。先把核心邏輯拆乾淨，比較重要。

## 10. 清楚分層後的責任邊界

整理後，每一層責任會變成：

```text
scripts/
  使用者入口，只處理 argparse

hidden_router/manifests/
  不同 benchmark -> 統一 manifest

hidden_router/models/
  target model loading / response generation

hidden_router/extraction/
  hidden state extraction / pooling

hidden_router/schema.py
  feature schema 驗證

hidden_router/probes.py
  router/probe 訓練

hidden_router/defense.py
  defense simulation

hidden_router/metrics.py
  AUROC/AUPRC/confusion/route metrics

hidden_router/plotting.py
  圖表

hidden_router/summaries.py
  markdown reports
```

這樣新使用者要找東西會比較直覺：

- 想換 benchmark：看 `manifests/`
- 想換模型：看 `models/` 和 `extraction/`
- 想改 router：看 `probes.py` 和 `defense.py`
- 想改 metrics：看 `metrics.py`
- 想改輸出報告：看 `summaries.py`

## 11. 是否需要真的重構

如果只是要完成目前實驗，現有 script 架構可以繼續用。

如果要交給別人長期使用或放進論文 artifact，建議至少做 Phase 1 和 Phase 2：

```text
Phase 1: 共用工具抽出
Phase 2: probe / defense 核心抽出
```

這兩步會讓程式碼清楚很多，也能避免不同 scripts 之間 metrics 或 split 邏輯慢慢分歧。
