import argparse
import json
from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from predict_aspect_opinion_llm import build_model as build_aspect_model, extract_pairs
from predict_category import load_label_map, predict_single as classify_categories
from predict_va_llm import load_model as build_va_model, predict_va


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="輸入 JSONL，需包含 ID 與 Text")
    parser.add_argument("--output_path", required=True, help="輸出 JSONL")
    parser.add_argument("--ao_model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--ao_adapter", default="", help="aspect/opinion LORA 目錄")
    parser.add_argument("--category_model_dir", required=True, help="中文BERT分類器目錄")
    parser.add_argument("--va_model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--va_adapter", default="", help="VA LORA 目錄")
    parser.add_argument("--cache_dir", default="/workplace/Share/LLM_model")
    parser.add_argument("--category_max_length", type=int, default=256)
    parser.add_argument("--ao_max_new_tokens", type=int, default=256)
    parser.add_argument("--va_max_new_tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading aspect/opinion model...")
    ao_model, ao_tokenizer = build_aspect_model(args.ao_model_id, args.ao_adapter or None, args.cache_dir)
    print("[INFO] Aspect/opinion model ready.")

    label2id = load_label_map(args.category_model_dir)
    id2label = {int(idx): label for label, idx in label2id.items()}
    print("[INFO] Loading category classifier...")
    cat_tokenizer = AutoTokenizer.from_pretrained(args.category_model_dir)
    cat_model = AutoModelForSequenceClassification.from_pretrained(args.category_model_dir)
    cat_model.eval()
    print("[INFO] Category classifier ready.")

    print("[INFO] Loading VA model...")
    va_model, va_tokenizer = build_va_model(args.va_model_id, args.va_adapter or None, args.cache_dir)
    print("[INFO] VA model ready.")

    outputs: List[Dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    with open(args.input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            record = json.loads(line)
            text = record["Text"]
            ao_pairs, ao_raw = extract_pairs(
                text,
                ao_model,
                ao_tokenizer,
                device,
                args.ao_max_new_tokens,
            )
            cat_results = classify_categories(text, ao_pairs, cat_model, cat_tokenizer, args.category_max_length, id2label)
            va_results = []
            for item in cat_results:
                va_value, va_raw = predict_va(
                    text,
                    item["Aspect"],
                    item["Opinion"],
                    va_model,
                    va_tokenizer,
                    args.va_max_new_tokens,
                )
                va_results.append(
                    {
                        "Aspect": item["Aspect"],
                        "Opinion": item["Opinion"],
                        "Category": item["Category"],
                        "VA": va_value,
                        "raw_generation": va_raw,
                    }
                )
            outputs.append(
                {
                    "ID": record["ID"],
                    "AspectOpinion": {
                        "raw_generation": ao_raw,
                        "pairs": ao_pairs,
                    },
                    "CategoryResults": cat_results,
                    "VAResults": va_results,
                    "Quadruplet": [
                        {
                            "Aspect": item["Aspect"],
                            "Category": item["Category"],
                            "Opinion": item["Opinion"],
                            "VA": item["VA"],
                        }
                        for item in va_results
                    ],
                }
            )
            if idx % 20 == 0:
                print(f"[INFO] Processed {idx} samples...")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Finished. Wrote {len(outputs)} predictions to {args.output_path}.")

if __name__ == "__main__":
    main()
