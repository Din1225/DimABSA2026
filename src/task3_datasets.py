from typing import Dict, List, Sequence, Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .task3_data import NULL_TEXT, NULL_TOKEN, SentenceInstance, get_valence_arousal_bins, sentence_to_char_labels


class SequenceTaggingDataset(Dataset):
    """Dataset for aspect/opinion sequence tagging."""

    label2id = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPN": 3, "I-OPN": 4}
    id2label = {idx: tag for tag, idx in label2id.items()}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        samples: Sequence[SentenceInstance],
        max_length: int = 256,
        include_null_token: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.samples = list(samples)
        self.max_length = max_length
        self.prefix = f"{NULL_TOKEN} " if include_null_token else ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        instance = self.samples[idx]
        augmented_text = f"{self.prefix}{instance.text}"
        encoding = self.tokenizer(
            augmented_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoding.pop("offset_mapping")
        aspect_chars, opinion_chars = sentence_to_char_labels(instance.text, instance.quadruplets, self.prefix)

        labels: List[int] = []
        for start, end in offsets:
            if end <= start or end > len(aspect_chars):
                labels.append(-100)
                continue
            aspect_tag = aspect_chars[start]
            opinion_tag = opinion_chars[start]
            if aspect_tag == 0 and opinion_tag == 0:
                labels.append(self.label2id["O"])
            elif aspect_tag in (1, 2):
                labels.append(self.label2id["B-ASP"] if aspect_tag == 1 else self.label2id["I-ASP"])
            elif opinion_tag in (1, 2):
                labels.append(self.label2id["B-OPN"] if opinion_tag == 1 else self.label2id["I-OPN"])
            else:
                labels.append(self.label2id["O"])

        encoding["labels"] = labels
        # return plain lists; Trainer collator will convert/pad
        return {key: value for key, value in encoding.items()}


class RelationDataset(Dataset):
    """Dataset for relation classification between aspect/opinion."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        features: Sequence[Tuple[str, str, str, str]],
        label2id: Dict[str, int],
        max_length: int = 192,
    ) -> None:
        self.tokenizer = tokenizer
        self.features = list(features)
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        text, aspect, opinion, category = self.features[idx]
        pair_text = f"{aspect} {self.tokenizer.sep_token} {opinion}"
        batch_encoding = self.tokenizer(
            text,
            pair_text,
            truncation=True,
            max_length=self.max_length,
        )
        encoding = {key: value for key, value in batch_encoding.items()}
        encoding["labels"] = self.label2id[category]
        return encoding


class IntensityDataset(Dataset):
    """Dataset for valence/arousal prediction given aspect-opinion pair."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        features: Sequence[Tuple[str, str, str, float, float]],
        max_length: int = 192,
        bin_step: float = 0.25,
    ) -> None:
        self.tokenizer = tokenizer
        self.features = list(features)
        self.max_length = max_length
        self.bin_step = bin_step

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[float]]:
        text, aspect, opinion, valence, arousal = self.features[idx]
        pair_text = f"{aspect} {self.tokenizer.sep_token} {opinion}"
        batch_encoding = self.tokenizer(
            text,
            pair_text,
            truncation=True,
            max_length=self.max_length,
        )
        encoding = {key: value for key, value in batch_encoding.items()}
        valence_bin = get_valence_arousal_bins(valence, step=self.bin_step)
        arousal_bin = get_valence_arousal_bins(arousal, step=self.bin_step)
        encoding["labels"] = [valence, arousal, float(valence_bin), float(arousal_bin)]
        return encoding
