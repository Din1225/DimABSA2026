# DimABSA2026
Semeval 2026 比賽: https://github.com/DimABSA/DimABSA2026


## 執行方式

### 訓練 BERT（筆電）
```bash
python -m src.train_task3 \
  --train-path data/train/zho_laptop_train_alltasks.jsonl \
  --dev-path data/dev/zho_laptop_dev_task3.jsonl \
  --output-dir outputs/Erlangshen-DeBERTa-v2-320M-Chinese/laptop\
  --model-name IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
  --num-epochs 5 \
  --train-batch-size 8 \
  --learning-rate 1e-5
```
### 訓練 BERT（餐廳）
```bash
python -m src.train_task3 \
  --train-path data/train/zho_restaurant_train_alltasks.jsonl \
  --dev-path data/dev/zho_restaurant_dev_task3.jsonl \
  --output-dir outputs/Erlangshen-DeBERTa-v2-320M-Chinese/restaurant \
  --model-name IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
  --num-epochs 5 \
  --train-batch-size 8 \
  --learning-rate 1e-5
```

### VA LLM 微調（筆電）
```bash
python -m src.finetune_va \
  --train-path data/train/zho_laptop_train_alltasks.jsonl \
  --eval-path data/dev/zho_laptop_dev_task3.jsonl \
  --output-dir outputs/Llama-3.1-8B-Instruct/laptop_va_Llama-3.1-8B-Instruct \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --cache-dir /workplace/Share/LLM_model \
  --num-epochs 5 \
  --per-device-train-batch-size 1 \
 --gradient-accumulation-steps 8
```

### VA LLM 微調（餐廳）
```bash
python -m src.finetune_va \
  --train-path data/train/zho_restaurant_train_alltasks.jsonl \
  --eval-path data/dev/zho_restaurant_dev_task3.jsonl \
  --output-dir outputs/Llama-3.1-8B-Instruct/restaurant_va_Llama-3.1-8B-Instruct \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --cache-dir /workplace/Share/LLM_model \
  --num-epochs 5 \
  --per-device-train-batch-size 1 \
 --gradient-accumulation-steps 8
```

產生的 LoRA 權重與 tokenizer 將儲存在 `--output-dir`，推論時請以 `--va-model-name` 指定該路徑。若 GPU 記憶體不足，可調整批次或加上 `--no-4bit/--load-in-8bit` 控制量化模式。

> LLM 回覆需為單行 `Valence#Arousal`（小數點一位，如 `6.5#7.5`），推論腳本會限制輸出長度並自動補零。

### 推論（筆電）
```bash
python -m src.predict_task3 \
  --model-root outputs/Erlangshen-DeBERTa-v2-320M-Chinese/laptop \
  --input-path data/dev/zho_laptop_dev_task3.jsonl \
  --output-path outputs/laptop_dev_pred.jsonl \
  --va-model-name outputs/Llama-3.1-8B-Instruct/laptop_va_Llama-3.1-8B-Instruct \
  --va-cache-dir /workplace/Share/LLM_model \
  --va-load-in-4bit  # LoRA 預設以 4bit 訓練，可加此參數
```

### 推論（餐廳）
```bash
python -m src.predict_task3 \
  --model-root outputs/Erlangshen-DeBERTa-v2-320M-Chinese/restaurant \
  --input-path data/dev/zho_restaurant_dev_task3.jsonl \
  --output-path outputs/restaurant_dev_pred.jsonl \
  --va-model-name outputs/Llama-3.1-8B-Instruct/restaurant_va_Llama-3.1-8B-Instruct \
  --va-cache-dir /workplace/Share/LLM_model \
  --va-load-in-4bit  # LoRA 預設以 4bit 訓練，可加此參數
```

`src.predict_task3` 會順序完成 aspect/opinion 抽取、aspect category 分類與 LLM-based VA 估計。若要啟用 LLM，請提供本地或快取好的模型路徑，並確保與 `transformers` 相容。

### 後處理 VA 的數值 (筆電為例)
```bash
python -m src.postprocess_va \
  --input-path outputs/laptop_dev_pred.jsonl \
  --output-path outputs/laptop_dev_pred_2dec.jsonl
```
### 後處理 VA 的數值 (餐廳)
```bash
python -m src.postprocess_va \
  --input-path outputs/restaurant_dev_pred.jsonl \
  --output-path outputs/restaurant_dev_pred_2dec.jsonl
```