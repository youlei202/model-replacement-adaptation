from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def helmert_matrix(option_count: int) -> np.ndarray:
    if option_count < 2:
        raise ValueError("option_count must be at least two")
    matrix = np.zeros((option_count - 1, option_count), dtype=np.float64)
    for row in range(option_count - 1):
        width = row + 1
        scale = np.sqrt(width * (width + 1))
        matrix[row, :width] = 1.0 / scale
        matrix[row, width] = -width / scale
    return matrix


def scores_to_coordinates(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values @ helmert_matrix(values.shape[1]).T


def coordinates_to_probabilities(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    option_count = values.shape[1] + 1
    logits = values @ helmert_matrix(option_count)
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.sum(probs, axis=1, keepdims=True)


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values -= np.max(values, axis=1, keepdims=True)
    probs = np.exp(values)
    return probs / np.sum(probs, axis=1, keepdims=True)


def total_variation(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    p = np.asarray(actual, dtype=np.float64)
    q = np.asarray(predicted, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("probability arrays must have equal shape")
    return 0.5 * np.sum(np.abs(p - q), axis=1)


def inverse_permutation(values: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    result = np.empty_like(values)
    result[:, np.asarray(permutation, dtype=int)] = values
    return result


def canonical_permutation(item_id: str, option_count: int, salt: str = "canonical") -> List[int]:
    ranks = []
    for option in range(option_count):
        text = "%s|20260722|%s|%d" % (item_id, salt, option)
        ranks.append((hashlib.sha256(text.encode("utf-8")).hexdigest(), option))
    return [option for _, option in sorted(ranks)]


PROMPTS: Dict[str, Tuple[str, str]] = {
    "mmlu_pro": (
        "Choose the single best option. Return only its numeric answer code.",
        "Question:\n{question}\n\nOptions:\n{options}\n\nReturn only one code from 0 to 9.",
    ),
    "hellaswag": (
        "Choose the single best option. Return only its numeric answer code.",
        "Context:\n{context}\n\nPossible continuations:\n{options}\n\nReturn only one code from 0 to 3.",
    ),
    "rewardbench": (
        "Choose the single best option. Return only its numeric answer code.",
        "Conversation:\n{context}\n\nCandidate responses:\n{options}\n\nWhich response should a careful, helpful, and safe assistant prefer?\nReturn only 0 or 1.",
    ),
}


@dataclass(frozen=True)
class SemanticItem:
    task_id: str
    item_id: str
    context: str
    options: Tuple[str, ...]
    gold: int
    domain: str


def render_prompt(item: SemanticItem, permutation: Sequence[int]) -> Tuple[str, str]:
    system, template = PROMPTS[item.task_id]
    ordered = [item.options[index] for index in permutation]
    option_text = "\n".join("%d. %s" % (index, value) for index, value in enumerate(ordered))
    if item.task_id == "mmlu_pro":
        user = template.format(question=item.context, options=option_text)
    else:
        user = template.format(context=item.context, options=option_text)
    return system, user
