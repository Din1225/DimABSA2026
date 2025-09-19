import argparse
import json
import os
from typing import Dict, List

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_samples(path: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record["Text"]
            for quad in record.get("Quadruplet", []):
                samples.append(
                    {
                        "text": text,
                        "aspect": quad["Aspect"],
                        "opinion": quad["Opinion"],
                        "category": quad["Category"],
                    }
                )
    return samples


def build_dataset(samples: List[Dict[str, str]], label2id: Dict[str, int], tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch):
        text_inputs = [
            f"[ASPECT] {a} [OPINION] {o} [TEXT] {t}"
            for a, o, t in zip(batch["aspect"], batch["opinion"], batch["text"])
        ]
        tokenized = tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [label2id[label] for label in batch["category"]]
        return tokenized

    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=42)
    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="trial_data/_train.jsonl")
    parser.add_argument("--valid_path", default="", help="可選的驗證 JSONL")
    parser.add_argument("--model_id", default="bert-base-chinese")
    parser.add_argument("--output_dir", default="./checkpoints/category_classifier")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_samples = load_samples(args.train_path)
    categories = sorted({sample["category"] for sample in train_samples})
    label2id = {label: idx for idx, label in enumerate(categories)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    train_dataset = build_dataset(train_samples, label2id, tokenizer, args.max_length)

    eval_dataset = None
    if args.valid_path:
        valid_samples = load_samples(args.valid_path)
        eval_dataset = build_dataset(valid_samples, label2id, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        # evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
