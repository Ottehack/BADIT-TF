"""Frozen P2 generation-evaluation helpers without training dependencies."""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nltk.stem import porter

from badit_tf.core import iter_ability_layers


def load_record(record: dict[str, Any], cache: dict[str, dict]) -> dict[str, Any]:
    source = record["source_file"]
    if source not in cache:
        cache[source] = json.loads(Path(source).read_text(encoding="utf-8"))
    item = dict(record)
    item["instance"] = cache[source]["Instances"][int(record["instance_index"])]
    return item


def set_route_masks(model: torch.nn.Module, mask: torch.Tensor) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_attention_mask(mask)


def clear_route_overrides(model: torch.nn.Module) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_route_code_override(None)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(no_punctuation.split())


class RougeTokenizer:
    """Exact frozen Google ROUGE default tokenization used by upstream."""

    _non_alphanumeric = re.compile(r"[^a-z0-9]+")
    _spaces = re.compile(r"\s+")
    _valid = re.compile(r"^[a-z0-9]+$")

    def __init__(self) -> None:
        self._stemmer = porter.PorterStemmer()

    def tokenize(self, text: str) -> list[str]:
        normalized = self._non_alphanumeric.sub(" ", text.lower())
        tokens = self._spaces.split(normalized)
        tokens = [
            self._stemmer.stem(token) if len(token) > 3 else token for token in tokens
        ]
        return [token for token in tokens if self._valid.match(token)]


def rouge_fmeasure(
    target: str, prediction: str, metric: str, tokenizer: RougeTokenizer
) -> float:
    target_tokens = tokenizer.tokenize(target)
    prediction_tokens = tokenizer.tokenize(prediction)
    if not target_tokens or not prediction_tokens:
        return 0.0
    if metric == "rouge1":
        overlap = sum((Counter(target_tokens) & Counter(prediction_tokens)).values())
    elif metric == "rougeL":
        previous = [0] * (len(prediction_tokens) + 1)
        for target_token in target_tokens:
            current = [0]
            for index, prediction_token in enumerate(prediction_tokens, start=1):
                if target_token == prediction_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        overlap = previous[-1]
    else:
        raise ValueError(metric)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = RougeTokenizer()
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        candidates = []
        for reference in row["references"]:
            candidates.append(
                {
                    "exact_match": float(
                        normalize_answer(row["prediction"])
                        == normalize_answer(reference)
                    ),
                    "rouge1": rouge_fmeasure(
                        reference, row["prediction"], "rouge1", tokenizer
                    ),
                    "rougeL": rouge_fmeasure(
                        reference, row["prediction"], "rougeL", tokenizer
                    ),
                }
            )
        by_task[row["task"]].append(
            {
                key: max(item[key] for item in candidates)
                for key in ("exact_match", "rouge1", "rougeL")
            }
        )
    per_task = {}
    for task, values in sorted(by_task.items()):
        per_task[task] = {
            key: 100.0 * float(np.mean([item[key] for item in values]))
            for key in ("exact_match", "rouge1", "rougeL")
        }
        per_task[task]["n"] = len(values)
    macro = {
        key: float(np.mean([item[key] for item in per_task.values()]))
        for key in ("exact_match", "rouge1", "rougeL")
    }
    micro = {
        key: 100.0
        * float(
            np.mean(
                [item[key] for values in by_task.values() for item in values]
            )
        )
        for key in ("exact_match", "rouge1", "rougeL")
    }
    return {"macro": macro, "micro": micro, "per_task": per_task}
