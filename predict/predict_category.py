import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_label_map(model_dir: str) -> Dict[str, int]:
    path = os.path.join(model_dir, "label2id.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_pairs(pairs: List[Dict[str, str]], text: str, tokenizer, max_length: int):
    inputs = [
        f"[ASPECT] {pair['Aspect']} [OPINION] {pair['Opinion']} [TEXT] {text}"
        for pair in pairs
    ]
    encoded = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded


def predict_single(text: str, pairs: List[Dict[str, str]], model, tokenizer, max_length: int, id2label: Dict[int, str]):
    if not pairs:
        return []
    encoded = tokenize_pairs(pairs, text, tokenizer, max_length)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        preds = outputs.logits.argmax(dim=-1).tolist()
    return [
        {
            "Aspect": pair["Aspect"],
            "Opinion": pair["Opinion"],
            "Category": id2label[idx],
        }
        for pair, idx in zip(pairs, preds)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="JSONL，需包含 Text 與 AspectOpinion")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_dir", required=True, help="已訓練 BERT 分類器目錄")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    label2id = load_label_map(args.model_dir)
    id2label = {int(idx): label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    results: List[Dict[str, object]] = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record["Text"]
            pairs = record.get("AspectOpinion", [])
            predictions = predict_single(text, pairs, model, tokenizer, args.max_length, id2label)
            results.append({"ID": record["ID"], "Text": text, "Predicted": predictions})

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
