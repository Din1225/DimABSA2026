# DimABSA2026
Semeval 2026 比賽: https://github.com/DimABSA/DimABSA2026


## 執行方式
- 參考以下指令依序微調三個模型（可視資源調整參數與輸出路徑）：
```bash
python train_aspect_opinion_llm.py --train_path /workplace/dxlin/Homework/NLP/DimABSA2026/trial_data/_train.jsonl --output_dir checkpoints/aspect_opinion
```
```bash
python train_category_classifier.py --train_path /workplace/dxlin/Homework/NLP/DimABSA2026/trial_data/_train.jsonl --output_dir checkpoints/category_classifier
```
```bash
python train_va_llm.py --train_path /workplace/dxlin/Homework/NLP/DimABSA2026/trial_data/_train.jsonl --output_dir checkpoints/va_llm
```

- 完成微調後，以 pipeline 腳本產出四元組：
```bash
python run_dim_asqp_pipeline.py \
  --input_path /workplace/dxlin/Homework/NLP/DimABSA2026/trial_data/_valid.jsonl \
  --output_path predictions.jsonl \
  --ao_adapter /workplace/dxlin/Homework/NLP/DimABSA2026/train/checkpoints/aspect_opinion/adapter \
  --category_model_dir /workplace/dxlin/Homework/NLP/DimABSA2026/train/checkpoints/category_classifier \
  --va_adapter /workplace/dxlin/Homework/NLP/DimABSA2026/train/checkpoints/va_llm/adapter
```