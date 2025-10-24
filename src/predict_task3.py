import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from .task3_data import NULL_TEXT, NULL_TOKEN, get_bin_center, load_jsonl
from .task3_models import ValenceArousalModel


def load_label_mapping(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def decode_entities(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    offsets: torch.Tensor,
    tokenizer,
    text: str,
    label_map: Dict[int, str],
    prefix_len: int,
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    null_id = tokenizer.convert_tokens_to_ids(NULL_TOKEN)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    preds = logits.argmax(dim=-1).tolist()
    token_ids = input_ids.tolist()
    offset_list = offsets.tolist()

    def collect(target: str) -> List[Dict[str, int]]:
        entities: List[Dict[str, int]] = []
        current = None
        for idx, label_id in enumerate(preds):
            label = label_map.get(label_id, "O")
            token_id = token_ids[idx]
            start, end = offset_list[idx]
            if token_id in (cls_id, sep_id, pad_id):
                if current:
                    entities.append(current)
                    current = None
                continue
            if token_id == null_id:
                if label in (f"B-{target}", f"I-{target}"):
                    candidate = {"text": NULL_TEXT, "start": -1, "end": -1}
                    if candidate not in entities:
                        entities.append(candidate)
                if current:
                    entities.append(current)
                    current = None
                continue
            if end <= start:
                if current:
                    entities.append(current)
                    current = None
                continue
            if label == f"B-{target}":
                if current:
                    entities.append(current)
                actual_start = max(0, start - prefix_len)
                actual_end = max(actual_start, end - prefix_len)
                current = {"text": "", "start": actual_start, "end": actual_end}
            elif label == f"I-{target}" and current:
                actual_end = max(current["end"], end - prefix_len)
                current["end"] = actual_end
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        for entity in entities:
            if entity["start"] >= 0:
                entity["text"] = text[entity["start"]:entity["end"]]
            else:
                entity["text"] = NULL_TEXT
        unique: List[Dict[str, int]] = []
        seen = set()
        for entity in entities:
            key = (entity["text"], entity["start"], entity["end"])
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique

    aspects = collect("ASP")
    opinions = collect("OPN")
    return aspects, opinions


def classify_pairs(
    model: AutoModelForSequenceClassification,
    tokenizer,
    text: str,
    aspects: List[Dict[str, int]],
    opinions: List[Dict[str, int]],
    label_map: Dict[int, str],
    max_length: int,
    device: torch.device,
) -> List[Dict]:
    if not aspects or not opinions:
        return []
    results: List[Dict] = []
    with torch.no_grad():
        for aspect in aspects:
            for opinion in opinions:
                aspect_text = aspect["text"]
                opinion_text = opinion["text"]
                encoded = tokenizer(
                    text,
                    f"{aspect_text} {tokenizer.sep_token} {opinion_text}",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**batch).logits.squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                pred_id = int(torch.argmax(probs).cpu().item())
                label = label_map[pred_id]
                if label == "INVALID":
                    continue
                confidence = float(probs[pred_id].cpu().item())
                results.append(
                    {
                        "aspect": aspect,
                        "opinion": opinion,
                        "category": label,
                        "confidence": confidence,
                    }
                )
    return results


def predict_intensity(
    model: ValenceArousalModel,
    tokenizer,
    text: str,
    candidates: List[Dict],
    step: float,
    max_length: int,
    device: torch.device,
) -> List[Dict]:
    outputs: List[Dict] = []
    if not candidates:
        return outputs
    with torch.no_grad():
        for candidate in candidates:
            aspect_text = candidate["aspect"]["text"]
            opinion_text = candidate["opinion"]["text"]
            encoded = tokenizer(
                text,
                f"{aspect_text} {tokenizer.sep_token} {opinion_text}",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in encoded.items()}
            forward = model(**batch)
            valence_logits = forward["valence_logits"].squeeze(0)
            arousal_logits = forward["arousal_logits"].squeeze(0)
            valence_score = forward["valence_score"].squeeze(0).cpu().item()
            arousal_score = forward["arousal_score"].squeeze(0).cpu().item()
            valence_bin = int(torch.argmax(torch.softmax(valence_logits, dim=-1)).cpu().item())
            arousal_bin = int(torch.argmax(torch.softmax(arousal_logits, dim=-1)).cpu().item())
            valence_cls = get_bin_center(valence_bin, step=step)
            arousal_cls = get_bin_center(arousal_bin, step=step)
            valence = max(1.0, min(9.0, (valence_score + valence_cls) / 2.0))
            arousal = max(1.0, min(9.0, (arousal_score + arousal_cls) / 2.0))
            outputs.append(
                {
                    "aspect": candidate["aspect"],
                    "opinion": candidate["opinion"],
                    "category": candidate["category"],
                    "valence": round(valence, 2),
                    "arousal": round(arousal, 2),
                }
            )
    return outputs


def format_output(predictions: List[Dict]) -> List[Dict]:
    formatted: List[Dict] = []
    for pred in predictions:
        aspect_text = pred["aspect"]["text"] if pred["aspect"]["start"] >= 0 else NULL_TEXT
        opinion_text = pred["opinion"]["text"] if pred["opinion"]["start"] >= 0 else NULL_TEXT
        formatted.append(
            {
                "Aspect": aspect_text,
                "Category": pred["category"],
                "Opinion": opinion_text,
                "VA": f"{pred['valence']:.2f}#{pred['arousal']:.2f}",
            }
        )
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DimASQP Subtask3 inference.")
    parser.add_argument("--model-root", type=Path, required=True, help="Directory with trained models.")
    parser.add_argument("--input-path", type=Path, required=True, help="Input jsonl file.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output jsonl file.")
    parser.add_argument("--relation-max-length", type=int, default=192, help="Max length for relation classifier.")
    parser.add_argument("--intensity-max-length", type=int, default=192, help="Max length for intensity predictor.")
    parser.add_argument("--tagger-max-length", type=int, default=256, help="Max length for sequence tagger.")
    return parser.parse_args()


def load_metadata(root: Path) -> Dict:
    meta_path = root / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.model_root)
    tagger_dir = args.model_root / "tagger"
    relation_dir = args.model_root / "relation"
    intensity_dir = args.model_root / "intensity"

    tokenizer = AutoTokenizer.from_pretrained(tagger_dir)
    tagger = AutoModelForTokenClassification.from_pretrained(tagger_dir)
    relation_model = AutoModelForSequenceClassification.from_pretrained(relation_dir)
    intensity_model = ValenceArousalModel.from_pretrained(intensity_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tagger.to(device).eval()
    relation_model.to(device).eval()
    intensity_model.to(device).eval()

    tagger_labels = load_label_mapping(tagger_dir / "tag_labels.json")
    relation_labels = load_label_mapping(relation_dir / "relation_labels.json")

    intensity_config = intensity_dir / "intensity_bins.json"
    if intensity_config.exists():
        with intensity_config.open("r", encoding="utf-8") as fh:
            config_data = json.load(fh)
        bin_step = float(config_data.get("step", 0.25))
    else:
        bin_step = 0.25

    tagger_max_length = int(metadata.get("tagger_max_length", args.tagger_max_length))
    relation_max_length = int(metadata.get("relation_max_length", args.relation_max_length))
    intensity_max_length = int(metadata.get("intensity_max_length", args.intensity_max_length))

    inputs = load_jsonl(args.input_path)
    prefix_len = len(f"{NULL_TOKEN} ")

    with args.output_path.open("w", encoding="utf-8") as out_fh:
        for item in inputs:
            augmented = f"{NULL_TOKEN} {item.text}"
            encoded = tokenizer(
                augmented,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=tagger_max_length,
            )
            offsets = encoded.pop("offset_mapping")[0]
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = tagger(**encoded).logits.squeeze(0)
            aspects, opinions = decode_entities(
                logits,
                encoded["input_ids"].squeeze(0).cpu(),
                offsets.cpu(),
                tokenizer,
                item.text,
                tagger_labels,
                prefix_len,
            )
            relation_candidates = classify_pairs(
                relation_model,
                tokenizer,
                item.text,
                aspects,
                opinions,
                relation_labels,
                relation_max_length,
                device,
            )
            intensity_predictions = predict_intensity(
                intensity_model,
                tokenizer,
                item.text,
                relation_candidates,
                step=bin_step,
                max_length=intensity_max_length,
                device=device,
            )
            quadruplets = format_output(intensity_predictions)
            out_record = {
                "ID": item.idx,
                "Quadruplet": quadruplets,
            }
            out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
