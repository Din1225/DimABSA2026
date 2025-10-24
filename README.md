# DimABSA2026
Semeval 2026 比賽: https://github.com/DimABSA/DimABSA2026


## 執行方式
訓練（以筆電資料為例）：
python -m src.train_task3 \
  --train-path data/train/zho_laptop_train_alltasks.jsonl \
  --output-dir outputs/laptop \
  --model-name bert-base-chinese
推論：
python -m src.predict_task3 --model-root outputs/laptop --input-path data/dev/zho_laptop_dev_task3.jsonl --output-path outputs/laptop_dev_pred.jsonl