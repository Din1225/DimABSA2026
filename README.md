# DimABSA2026
Semeval 2026 比賽: https://github.com/DimABSA/DimABSA2026

## Pipeline 實作細節

### Aspect-Opinion Extraction
#### Dataset 與 BIO 標註
- `constants.py` 定義 Aspect/Opinion BIO 標籤與所有合法的 `ENTITY#ATTRIBUTE` 組合。
- `data_utils.extract_spans_from_quads` 從金標 quadruplet 內自動尋找每個 Aspect / Opinion 的文字與 `(start, end)` 位置。
- `_encode_token_labels` 根據 tokenizer 的 `offset_mapping` 將 span 轉成 B-/I- 標籤，搭配 `TokenTaggingDataset` 打包成可餵進 HuggingFace `Trainer` 的資料。
先對資料進行 BIO 標記，將 span 的形式變成 BIO 標籤的形式。

#### 負樣本與配對資料
- `build_relation_dataset.py` 使用 K-fold cross-validation 先訓練簡易 BIO 模型，再在各驗證fold上進行推論，得到多個 span。
- `build_pairs` 會組合：
  1. 金標 aspect-opinion + 正確類別 (正樣本)。
  2. 預測-預測、預測+金標、金標+預測 的無標籤組合 (負樣本)，並標記為 `INVALID`。
用 K-fold cross-validation 的方式先訓練 BIO 模型，然後用訓練好的模型在驗證的fold上進行推論，會抓取到多個aspect和opinion。然後進行對多個aspect和opinion進行組合，只要不是正確配對，就標記成INVALID。每句可保留最多 10 筆負樣本，輸出成 JSONL 給 relation classifier 訓練使用。


#### 訓練流程
- `train_task3.py` 依序訓練三個子模型：
  1. Aspect BIO tagger (`TokenTaggingDataset`，`AutoModelForTokenClassification`)。
  2. Opinion BIO tagger（同上）。
  3. Relation classifier (`AspectOpinionPairDataset`，`AutoModelForSequenceClassification`，含 `INVALID` 負樣本)。
  4. Category classifier (AspectCategoryDataset，AutoModelForSequenceClassification)：針對每筆 quadruplet，把 Aspect、原句 Text，將目標 Category（合法 ENTITY#ATTRIBUTE）當作分類標籤；藉由這樣的輸入，模型學會「僅依靠 Aspect 及其上下文」預測類別，作為 relation 模型失敗時的備援。
  5. LLM：透過 finetune_va.py 以 QLoRA SFT 方式微調 meta-llama/Llama-3.1-8B-Instruct（可改 base model）。collect_examples 從訓練 JSONL 擷取所有 (Text, Aspect, Opinion, VA)，利用 build_instruct_prompt 組成統一指令，再把真實 V#A 直接接在 prompt 後，形成單一文字序列。然後將其tokenize。

四個模型 Aspect BIO tagger, Opinion BIO tagger, Relation classifier(會使用到負樣本去做訓練), Category classifier，
所有模型訓練參數: `TrainingArguments`：`lr=1e-5`、`warmup_ratio=0.1`、`max_length=256`、`AdamW優化器`，使用 Cross-Entropy 當 loss function(因為是多類別，所以用softmax把logits轉換成各類別的機率分布)。

LLM：以 QLoRA SFT 方式微調。從訓練 JSONL 擷取所有 (Text, Aspect, Opinion, VA)，利用 build_instruct_prompt 組成統一指令，再把真實 V#A 直接接在 prompt 後，形成單一文字序列。然後將其tokenize。訓練參數num_epochs=5、per_device_train_batch_size=1、gradient_accumulation_steps=8、learning_rate=1e-4、warmup_ratio=0.05，最後儲存訓練好的 adapter。


#### 推論 (`predict_task3.py`)
1. 載入四個子模型（從 `metadata.json` 解析存放路徑）。
2. 針對輸入文本：
   - 以 aspect/opinion tokenizer 取得 `offset_mapping`，送入模型並透過 `decode_bio` 轉回 span。
   - 若只偵測到 aspect/opinion 其中一個，另一種缺失的以整句補上。
3. Relation 判斷：
Relation classifier 主要針對「某個 (aspect span, opinion span) 組合屬於哪個 ENTITY#ATTRIBUTE 類別？」進行判斷；它同時要辨識合法類別與 INVALID。
   - 先使用 relation classifier：在原句插入 `[ASP]…[/ASP]`、`[OPN]…[/OPN]`，並提供 `aspect SEP opinion` 第二序列，過濾掉預測 `INVALID` 的結果。
   - 若 relation 模型輸出皆 `INVALID`，呼叫 category classifier：對每個 aspect 套用 category classifier，再用 `assign_opinions_to_aspects` 依距離就近配對。
4. VA 預測 (LLM):
使用 VAPredictorConfig 給定 LLM 路徑、裝置、生成長度 (max_new_tokens≤7)、temperature、top_p 以及 4-bit 量化的設定。
推論時呼叫 build_instruct_prompt 將 Text / Aspect / Opinion 填入統一的指令模板，再把 prompt tokenized 後送入 LLM generate，擷取新生成的字串。
parse_output 以正則尋找 V#A，若解析失敗則回傳 5.0#5.0




## 執行說明

### 建立 Relation 訓練資料（筆電）
```bash
python -m src.build_relation_dataset \
  --input-path data/train/zho_laptop_train_alltasks.jsonl \
  --output-path data/train/ernie-3.0-xbase-zh_zho_laptop_train_task3_pairs.jsonl \
  --model-name nghuyong/ernie-3.0-xbase-zh \
  --num-folds 5
```
> 依論文方法使用 k-fold 建立正負樣本。餐廳資料請替換相對應路徑與輸出名稱。
### 建立 Relation 訓練資料（餐廳）
```bash
python -m src.build_relation_dataset \
  --input-path data/train/zho_restaurant_train_alltasks.jsonl \
  --output-path data/train/ernie-3.0-xbase-zh_zho_restaurant_train_task3_pairs.jsonl \
  --model-name nghuyong/ernie-3.0-xbase-zh \
  --num-folds 5
```

### 訓練 BERT（筆電）
```bash
python -m src.train_task3 \
  --train-path data/train/zho_laptop_train_alltasks.jsonl \
  --dev-path data/dev/zho_laptop_dev_task3.jsonl \
  --train-pair-path data/train/ernie-3.0-xbase-zh_zho_laptop_train_task3_pairs.jsonl \
  --output-dir outputs/ernie-3.0-xbase-zh_with_negative_sample/laptop\
  --model-name nghuyong/ernie-3.0-xbase-zh \
  --num-epochs 5 \
  --train-batch-size 8 \
  --learning-rate 1e-5
```
### 訓練 BERT（餐廳）
```bash
python -m src.train_task3 \
  --train-path data/train/zho_restaurant_train_alltasks.jsonl \
  --dev-path data/dev/zho_restaurant_dev_task3.jsonl \
  --train-pair-path data/train/ernie-3.0-xbase-zh_zho_restaurant_train_task3_pairs.jsonl \
  --output-dir outputs/ernie-3.0-xbase-zh_with_negative_sample/restaurant \
  --model-name nghuyong/ernie-3.0-xbase-zh \
  --num-epochs 5 \
  --train-batch-size 8 \
  --learning-rate 1e-5
```

### VA LLM 微調（筆電）
```bash
python -m src.finetune_va \
  --train-path data/train/zho_laptop_train_alltasks.jsonl \
  --eval-path data/dev/zho_laptop_dev_task3.jsonl \
  --output-dir outputs/Qwen3-14B/laptop_va_Qwen3-14B \
  --base-model Qwen/Qwen3-14B \
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
  --output-dir outputs/Qwen3-14B/restaurant_va_Qwen3-14B \
  --base-model Qwen/Qwen3-14B \
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
  --model-root outputs/ernie-3.0-xbase-zh_with_negative_sample/laptop \
  --input-path data/dev/zho_laptop_dev_task3.jsonl \
  --output-path outputs/laptop_dev_pred.jsonl \
  --va-model-name outputs/Qwen3-14B/laptop_va_Qwen3-14B \
  --va-cache-dir /workplace/Share/LLM_model \
  --va-load-in-4bit  # LoRA 預設以 4bit 訓練，可加此參數
```

### 推論（餐廳）
```bash
python -m src.predict_task3 \
  --model-root outputs/ernie-3.0-xbase-zh_with_negative_sample/restaurant \
  --input-path data/dev/zho_restaurant_dev_task3.jsonl \
  --output-path outputs/restaurant_dev_pred.jsonl \
  --va-model-name outputs/Qwen3-14B/restaurant_va_Qwen3-14B \
  --va-cache-dir /workplace/Share/LLM_model \
  --va-load-in-4bit  # LoRA 預設以 4bit 訓練，可加此參數
```

`src.predict_task3` 會順序完成 aspect/opinion 抽取、relation 分類（同時決定配對與類別）與 LLM-based VA 估計。若 relation 模型預測不到有效配對，系統會退回舊版類別分類流程。若要啟用 LLM，請提供本地或快取好的模型路徑，並確保與 `transformers` 相容。

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
