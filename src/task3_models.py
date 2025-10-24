from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    PretrainedConfig,
)


class ValenceArousalModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, num_bins: int, dropout_prob: float = 0.0) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.valence_classifier = nn.Linear(hidden_size, num_bins)
        self.arousal_classifier = nn.Linear(hidden_size, num_bins)
        self.valence_regressor = nn.Linear(hidden_size, 1)
        self.arousal_regressor = nn.Linear(hidden_size, 1)
        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        valence_logits = self.valence_classifier(pooled)
        arousal_logits = self.arousal_classifier(pooled)
        valence_score = self.valence_regressor(pooled).squeeze(-1)
        arousal_score = self.arousal_regressor(pooled).squeeze(-1)
        result: Dict[str, torch.Tensor] = {
            "valence_logits": valence_logits,
            "arousal_logits": arousal_logits,
            "valence_score": valence_score,
            "arousal_score": arousal_score,
        }
        if labels is not None:
            valence_target = labels[:, 0]
            arousal_target = labels[:, 1]
            valence_bins = labels[:, 2].long()
            arousal_bins = labels[:, 3].long()
            loss = (
                self.loss_mse(valence_score, valence_target)
                + self.loss_mse(arousal_score, arousal_target)
                + self.loss_ce(valence_logits, valence_bins)
                + self.loss_ce(arousal_logits, arousal_bins)
            ) / 4.0
            result["loss"] = loss
        return result


def build_tokenizer(model_name: str, null_token: str = "[NULL]") -> Tuple[AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    to_add = []
    if null_token not in tokenizer.get_vocab():
        to_add.append(null_token)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer, len(to_add)


def build_sequence_tagger(model_name: str, num_labels: int) -> AutoModelForTokenClassification:
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def build_relation_classifier(model_name: str, num_labels: int) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )


def build_intensity_model(model_name: str, num_bins: int) -> ValenceArousalModel:
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    return ValenceArousalModel.from_pretrained(
        model_name,
        config=config,
        num_bins=num_bins,
        dropout_prob=0.0,
    )
