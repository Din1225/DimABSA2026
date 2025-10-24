import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .task3_data import (
    NULL_TEXT,
    NULL_TOKEN,
    SentenceInstance,
    collect_categories,
    get_valence_arousal_bins,
    load_jsonl,
    train_valid_split,
)
from .task3_datasets import IntensityDataset, RelationDataset, SequenceTaggingDataset
from .task3_models import build_intensity_model, build_relation_classifier, build_sequence_tagger, build_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_relation_features(instances: Sequence[SentenceInstance]) -> List[Tuple[str, str, str, str]]:
    features: List[Tuple[str, str, str, str]] = []
    for inst in instances:
        positives: Dict[Tuple[str, str], str] = {}
        aspects = set()
        opinions = set()
        for quad in inst.quadruplets:
            aspect_text = quad.aspect if quad.aspect != NULL_TEXT else NULL_TEXT
            opinion_text = quad.opinion if quad.opinion != NULL_TEXT else NULL_TEXT
            pair = (aspect_text, opinion_text)
            positives[pair] = quad.category
            aspects.add(aspect_text)
            opinions.add(opinion_text)
        for (aspect, opinion), category in positives.items():
            features.append((inst.text, aspect, opinion, category))
        negative_candidates = []
        pairs = [(a, o) for a in aspects for o in opinions if (a, o) not in positives]
        random.shuffle(pairs)
        max_negatives = max(1, len(positives))
        for aspect, opinion in pairs[:max_negatives]:
            negative_candidates.append((inst.text, aspect, opinion, "INVALID"))
        features.extend(negative_candidates)
    return features


def build_intensity_features(instances: Sequence[SentenceInstance]) -> List[Tuple[str, str, str, float, float]]:
    features: List[Tuple[str, str, str, float, float]] = []
    for inst in instances:
        for quad in inst.quadruplets:
            aspect_text = quad.aspect if quad.aspect != NULL_TEXT else NULL_TEXT
            opinion_text = quad.opinion if quad.opinion != NULL_TEXT else NULL_TEXT
            features.append(
                (
                    inst.text,
                    aspect_text,
                    opinion_text,
                    quad.valence,
                    quad.arousal,
                )
            )
    return features


def train_sequence_tagger(
    model_name: str,
    tokenizer,
    train_data: Sequence[SentenceInstance],
    valid_data: Sequence[SentenceInstance],
    output_dir: Path,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    train_dataset = SequenceTaggingDataset(tokenizer, train_data, max_length=max_length)
    valid_dataset = SequenceTaggingDataset(tokenizer, valid_data, max_length=max_length)
    model = build_sequence_tagger(model_name, num_labels=len(SequenceTaggingDataset.label2id))
    model.resize_token_embeddings(len(tokenizer))
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        seed=42,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "tag_labels.json").open("w", encoding="utf-8") as fh:
        json.dump(SequenceTaggingDataset.id2label, fh, ensure_ascii=False, indent=2)


def train_relation_classifier_model(
    model_name: str,
    tokenizer,
    categories: Sequence[str],
    train_data: Sequence[SentenceInstance],
    valid_data: Sequence[SentenceInstance],
    output_dir: Path,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    label2id = {"INVALID": 0}
    for idx, cat in enumerate(categories, start=1):
        label2id[cat] = idx
    id2label = {idx: label for label, idx in label2id.items()}
    train_features = build_relation_features(train_data)
    valid_features = build_relation_features(valid_data)
    train_dataset = RelationDataset(tokenizer, train_features, label2id, max_length=max_length)
    valid_dataset = RelationDataset(tokenizer, valid_features, label2id, max_length=max_length)
    model = build_relation_classifier(model_name, num_labels=len(label2id))
    model.resize_token_embeddings(len(tokenizer))
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "relation_labels.json").open("w", encoding="utf-8") as fh:
        json.dump(id2label, fh, ensure_ascii=False, indent=2)


def train_intensity_model(
    model_name: str,
    tokenizer,
    train_data: Sequence[SentenceInstance],
    valid_data: Sequence[SentenceInstance],
    output_dir: Path,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    bin_step: float = 0.25,
) -> None:
    train_features = build_intensity_features(train_data)
    valid_features = build_intensity_features(valid_data)
    train_dataset = IntensityDataset(tokenizer, train_features, max_length=max_length, bin_step=bin_step)
    valid_dataset = IntensityDataset(tokenizer, valid_features, max_length=max_length, bin_step=bin_step)
    num_bins = get_valence_arousal_bins(9.0, step=bin_step) + 1
    model = build_intensity_model(model_name, num_bins=num_bins)
    model.resize_token_embeddings(len(tokenizer))
    model.config.architectures = ["ValenceArousalModel"]
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.0,
        logging_steps=50,
        seed=42,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "intensity_bins.json").open("w", encoding="utf-8") as fh:
        json.dump({"step": bin_step, "num_bins": num_bins}, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DimASQP Subtask3 pipeline.")
    parser.add_argument("--train-path", type=Path, required=True, help="Path to training jsonl file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for saving models.")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese", help="Base transformer model.")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for tagging.")
    parser.add_argument("--relation-max-length", type=int, default=192, help="Max length for relation model.")
    parser.add_argument("--intensity-max-length", type=int, default=192, help="Max length for intensity model.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for tagging and relation models.")
    parser.add_argument("--intensity-epochs", type=int, default=6, help="Epochs for intensity model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for all models.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate for tagging/relation.")
    parser.add_argument("--intensity-learning-rate", type=float, default=2e-5, help="Learning rate for intensity model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    instances = load_jsonl(args.train_path)
    train_data, valid_data = train_valid_split(instances, valid_ratio=args.val_ratio, seed=args.seed)
    categories = collect_categories(instances)
    tokenizer, added_tokens = build_tokenizer(args.model_name, NULL_TOKEN)

    tagging_dir = args.output_dir / "tagger"
    relation_dir = args.output_dir / "relation"
    intensity_dir = args.output_dir / "intensity"
    ensure_dir(tagging_dir)
    ensure_dir(relation_dir)
    ensure_dir(intensity_dir)

    train_sequence_tagger(
        args.model_name,
        tokenizer,
        train_data,
        valid_data,
        tagging_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    train_relation_classifier_model(
        args.model_name,
        tokenizer,
        categories,
        train_data,
        valid_data,
        relation_dir,
        max_length=args.relation_max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    train_intensity_model(
        args.model_name,
        tokenizer,
        train_data,
        valid_data,
        intensity_dir,
        max_length=args.intensity_max_length,
        epochs=args.intensity_epochs,
        batch_size=args.batch_size,
        learning_rate=args.intensity_learning_rate,
    )

    with (args.output_dir / "categories.json").open("w", encoding="utf-8") as fh:
        json.dump(categories, fh, ensure_ascii=False, indent=2)

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": args.model_name,
                "val_ratio": args.val_ratio,
                "added_special_tokens": added_tokens,
                "tagger_max_length": args.max_length,
                "relation_max_length": args.relation_max_length,
                "intensity_max_length": args.intensity_max_length,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
