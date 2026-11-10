from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


PROTOCOL_VERSION = "audit_v29"
TRANSFORMERS_COMMIT = "c7ad72879438c7d83955394312cea9f437e9f43b"
GLOBAL_SEED = 20260722
MAX_CONTEXT_TOKENS = 4096
MIN_GOLD_STRATUM = 4
MIN_DOMAIN_STRATUM = 4
PREFLIGHT_ITEMS = 256
PREFLIGHT_GENERATIONS = 32
PREFLIGHT_LOGIT_TOLERANCE = 0.02
PREFLIGHT_PROBABILITY_TOLERANCE = 0.002
PREFLIGHT_DETERMINISM_TOLERANCE = 1e-7
PREFLIGHT_NORMALIZATION_TOLERANCE = 1e-6
PREFLIGHT_VRAM_LIMIT_GIB = 29.0
MODEL_JOB_MINIMUM_FREE_GIB = 29.0
MISTRAL_TOKENIZER_REGEX_FIX = True


@dataclass(frozen=True)
class ModelContract:
    model_id: str
    role: str
    thinking: bool = False


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    dataset_id: str
    dataset_config: str
    source_split: str
    option_count: int
    total: int
    fit: int
    validation: int
    iid_test: int
    ood_test: int
    domain_field: str


PEERS: Tuple[ModelContract, ...] = (
    ModelContract("Qwen/Qwen3-8B", "peer"),
    ModelContract("allenai/Olmo-3-7B-Instruct", "peer"),
    ModelContract("mistralai/Ministral-3-8B-Instruct-2512-BF16", "peer"),
)

REPLICATION_PEERS: Tuple[ModelContract, ...] = (
    ModelContract("mistralai/Mistral-7B-Instruct-v0.3", "replication_peer"),
    ModelContract("tiiuae/Falcon3-7B-Instruct", "replication_peer"),
    ModelContract("Qwen/Qwen2.5-7B-Instruct", "replication_peer"),
)

TARGETS: Tuple[ModelContract, ...] = (
    ModelContract("Qwen/Qwen3.5-9B", "target"),
    ModelContract("google/gemma-4-12B-it", "target"),
    ModelContract("ibm-granite/granite-3.3-8b-instruct", "target"),
)

ROLE_ROTATION_TARGETS: Tuple[ModelContract, ...] = tuple(
    ModelContract(model.model_id, "secondary_target") for model in REPLICATION_PEERS
)

CROSS_ROLE_PEERS: Tuple[ModelContract, ...] = tuple(
    ModelContract(model.model_id, "cross_role_peer") for model in TARGETS
)

RESPONSE_MODELS: Tuple[ModelContract, ...] = PEERS + TARGETS + REPLICATION_PEERS

TASKS: Tuple[TaskContract, ...] = (
    TaskContract(
        "mmlu_pro",
        "TIGER-Lab/MMLU-Pro",
        "default",
        "test",
        10,
        4608,
        2048,
        512,
        1024,
        1024,
        "category",
    ),
    TaskContract(
        "hellaswag",
        "Rowan/hellaswag",
        "default",
        "validation",
        4,
        4608,
        2048,
        512,
        1024,
        1024,
        "activity_label",
    ),
    TaskContract(
        "rewardbench",
        "allenai/reward-bench",
        "default",
        "filtered",
        2,
        2048,
        1024,
        256,
        512,
        256,
        "subset",
    ),
)

LAMBDA_GRID: Tuple[float, ...] = tuple(float(value) for value in np.logspace(-8, 2, 17))
RFF_DIMENSIONS: Tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
SPECTRAL_RANKS: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
QUANTIZED_BITS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 32)
NATIVE_CODECS: Tuple[str, ...] = ("native_fp16", "native_fp32", "native_fp64")
BIT_CAPS: Tuple[int, ...] = (
    0,
    2 ** 5,
    2 ** 7,
    2 ** 9,
    2 ** 10,
    2 ** 11,
    2 ** 13,
    2 ** 14,
    2 ** 15,
    2 ** 17,
    2 ** 18,
    2 ** 19,
    2 ** 20,
    2 ** 21,
)
MATCHED_BUDGETS: Tuple[int, ...] = (2 ** 10, 2 ** 14, 2 ** 18)
ROSTER_REPLICATION_ADAPTERS: Tuple[str, ...] = ("convex", "rff", "krr")
REFERENCE_ROSTER_ID = "reference_instruction_pool"
REPLICATION_ROSTER_ID = "independent_instruction_pool"
CROSS_ROLE_ROSTER_ID = "cross_role_instruction_pool"
FIT_SIZES: Tuple[int, ...] = (128, 256, 512, 1024)
MLP_WIDTHS: Tuple[int, ...] = (32, 64, 128)
MLP_LAMBDA_INDICES: Tuple[int, ...] = (4, 8, 12)
ADAPTERS: Tuple[str, ...] = ("convex", "linear", "rff", "krr", "mlp")
MAIN_ADAPTERS: Tuple[str, ...] = ("mean_all",) + ADAPTERS
PEER_COUNTS: Tuple[int, ...] = (1, 2, 3)


def protocol_dict() -> Dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "global_seed": GLOBAL_SEED,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "minimum_gold_stratum": MIN_GOLD_STRATUM,
        "minimum_domain_stratum": MIN_DOMAIN_STRATUM,
        "preflight_items": PREFLIGHT_ITEMS,
        "preflight_generation_cases": PREFLIGHT_GENERATIONS,
        "preflight_logit_tolerance": PREFLIGHT_LOGIT_TOLERANCE,
        "preflight_probability_tolerance": PREFLIGHT_PROBABILITY_TOLERANCE,
        "preflight_determinism_tolerance": PREFLIGHT_DETERMINISM_TOLERANCE,
        "preflight_normalization_tolerance": PREFLIGHT_NORMALIZATION_TOLERANCE,
        "preflight_vram_limit_gib": PREFLIGHT_VRAM_LIMIT_GIB,
        "model_job_minimum_free_gib": MODEL_JOB_MINIMUM_FREE_GIB,
        "mistral_tokenizer_regex_fix": MISTRAL_TOKENIZER_REGEX_FIX,
        "peers": [asdict(model) for model in PEERS],
        "replication_peers": [asdict(model) for model in REPLICATION_PEERS],
        "targets": [asdict(model) for model in TARGETS],
        "secondary_targets": [asdict(model) for model in ROLE_ROTATION_TARGETS],
        "role_rotation_rosters": [
            {
                "roster_id": REFERENCE_ROSTER_ID,
                "models": [asdict(model) for model in PEERS],
            },
            {
                "roster_id": CROSS_ROLE_ROSTER_ID,
                "models": [asdict(model) for model in CROSS_ROLE_PEERS],
            },
        ],
        "tasks": [asdict(task) for task in TASKS],
        "lambda_grid": list(LAMBDA_GRID),
        "rff_dimensions": list(RFF_DIMENSIONS),
        "spectral_ranks": list(SPECTRAL_RANKS),
        "quantized_bits": list(QUANTIZED_BITS),
        "native_codecs": list(NATIVE_CODECS),
        "bit_caps": list(BIT_CAPS),
        "matched_budgets": list(MATCHED_BUDGETS),
        "fit_sizes": list(FIT_SIZES),
        "mlp_widths": list(MLP_WIDTHS),
        "mlp_lambda_indices": list(MLP_LAMBDA_INDICES),
        "split_order": [
            "tokenizer_intersection",
            "whole_domain_ood",
            "cluster_disjoint_iid",
        ],
        "test_unlocks": 1,
    }


def validate_model_revision(
    model_id: str, revision: str, protocol: Mapping[str, object]
) -> None:
    revisions = protocol.get("revisions")
    if not isinstance(revisions, Mapping):
        raise ValueError("protocol lock has no revision map")
    models = revisions.get("models")
    if not isinstance(models, Mapping) or model_id not in models:
        raise ValueError("model is not part of the frozen revision lock")
    expected = str(models[model_id])
    if revision != expected:
        raise ValueError(
            "revision mismatch for %s: expected %s, got %s"
            % (model_id, expected, revision)
        )
