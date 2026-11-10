from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import (
    MAX_CONTEXT_TOKENS,
    PREFLIGHT_VRAM_LIMIT_GIB,
    PREFLIGHT_DETERMINISM_TOLERANCE,
    PREFLIGHT_ITEMS,
    PREFLIGHT_LOGIT_TOLERANCE,
    PREFLIGHT_NORMALIZATION_TOLERANCE,
    PREFLIGHT_PROBABILITY_TOLERANCE,
    validate_model_revision,
)
from kernel_isqed.io import (
    CANONICAL_DIR,
    read_json,
    write_json,
    write_table,
)
from kernel_isqed.llm import SemanticOptionScorer
from kernel_isqed.frozen import source_tree_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 256-item Target-Imprint LLM preflight"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--items", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_DIR / "preflight")
    args = parser.parse_args()
    validate_model_revision(
        args.model_id, args.revision, read_json(CANONICAL_DIR / "protocol_lock.json")
    )
    if args.items != PREFLIGHT_ITEMS:
        raise ValueError("formal preflight is fixed at %d items" % PREFLIGHT_ITEMS)
    if args.device != "cuda":
        raise ValueError("formal LLM preflight requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to run formal preflight on CPU"
        )
    prompts = pd.read_parquet(CANONICAL_DIR / "prompt_manifest.parquet")
    expected = pd.read_parquet(CANONICAL_DIR / "expected_run_keys.parquet")
    registered = expected[
        (expected["key_family"] == "PREFLIGHT")
        & (expected["check"] == "stratified_item")
        & (expected["model_id"] == args.model_id)
    ][["task_id", "item_id", "length_bin", "tail_anchor"]]
    if registered.duplicated(["task_id", "item_id"]).any():
        raise RuntimeError("preflight expected-key ledger contains duplicate items")
    if prompts.duplicated(["task_id", "item_id"]).any():
        raise RuntimeError("prompt manifest contains duplicate task/item keys")
    sample = registered.merge(
        prompts, on=["task_id", "item_id"], how="left", validate="one_to_one"
    )
    if len(sample) != args.items:
        raise RuntimeError(
            "preflight requested %d items but only %d are available"
            % (args.items, len(sample))
        )
    sanity = expected[
        (expected["key_family"] == "PREFLIGHT")
        & (expected["check"] == "sanity_generation")
        & (expected["model_id"] == args.model_id)
    ]
    generation_ids = set(
        map(tuple, sanity[["task_id", "item_id"]].astype(str).to_numpy())
    )
    if len(generation_ids) != 32:
        raise RuntimeError(
            "preflight expected-key ledger does not contain 32 sanity items"
        )
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    scorer = SemanticOptionScorer(args.model_id, args.revision, args.device)
    load_seconds = time.perf_counter() - load_started
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    vram_limit_bytes = int(PREFLIGHT_VRAM_LIMIT_GIB * (1024 ** 3))
    rows = []
    inventory = None
    codebooks = {}
    for _, row in sample.iterrows():
        option_count = {"mmlu_pro": 10, "hellaswag": 4, "rewardbench": 2}[
            row["task_id"]
        ]
        codes = [str(index) for index in range(option_count)]
        if row["task_id"] not in codebooks:
            task_inventory = scorer.inventory(
                row["system_prompt"], row["user_prompt"], codes
            )
            codebooks[row["task_id"]] = {
                key: task_inventory[key]
                for key in (
                    "final_answer_prefix_ids",
                    "final_answer_prefix_hash",
                    "scored_position",
                    "code_token_ids",
                    "code_token_lengths",
                    "code_tokenization",
                    "codebook_mode",
                )
            }
            if inventory is None:
                inventory = task_inventory
        started = time.perf_counter()
        first = scorer.score(row["system_prompt"], row["user_prompt"], codes)
        second = scorer.score(row["system_prompt"], row["user_prompt"], codes)
        equivalence = scorer.compare_official_logits(
            row["system_prompt"], row["user_prompt"], codes
        )
        generation = {
            "generated_text": "",
            "generation_code": "",
            "generation_valid": np.nan,
        }
        if (str(row["task_id"]), str(row["item_id"])) in generation_ids:
            generation = scorer.generation_sanity(
                row["system_prompt"], row["user_prompt"], codes
            )
        elapsed = time.perf_counter() - started
        prompt_length = int(
            scorer.render(row["system_prompt"], row["user_prompt"]).shape[1]
        )
        peak_allocated = int(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        probability_sum_error = abs(float(first.probabilities.sum()) - 1.0)
        deterministic_difference = float(
            np.max(np.abs(first.probabilities - second.probabilities))
        )
        codebook_valid = all(
            length > 0 for length in first.code_token_lengths
        ) and (
            args.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
            or all(length == 1 for length in first.code_token_lengths)
        )
        rows.append(
            {
                "model_id": args.model_id,
                "task_id": row["task_id"],
                "item_id": row["item_id"],
                "length_bin": int(row["length_bin"]),
                "tail_anchor": bool(row["tail_anchor"]),
                "prompt_tokens": prompt_length,
                "probability_sum": float(first.probabilities.sum()),
                "probability_sum_error": probability_sum_error,
                "deterministic_max_difference": deterministic_difference,
                "centered_logit_max_difference": equivalence[
                    "centered_logit_max_difference"
                ],
                "official_probability_max_difference": equivalence[
                    "probability_max_difference"
                ],
                "seconds_per_view": elapsed,
                "views_per_second": 1.0 / elapsed,
                "input_tokens_per_second": prompt_length / elapsed,
                "candidate_passes_per_view": 1,
                "batch_token_cap": MAX_CONTEXT_TOKENS,
                "peak_allocated_bytes": peak_allocated,
                "vram_limit_bytes": vram_limit_bytes,
                "code_token_lengths": "|".join(
                    str(value) for value in first.code_token_lengths
                ),
                "generated_text": generation["generated_text"],
                "generation_code": generation["generation_code"],
                "generation_valid": generation["generation_valid"],
                "status": "completed"
                if (
                    prompt_length <= MAX_CONTEXT_TOKENS
                    and np.isfinite(first.probabilities).all()
                    and probability_sum_error <= PREFLIGHT_NORMALIZATION_TOLERANCE
                    and deterministic_difference <= PREFLIGHT_DETERMINISM_TOLERANCE
                    and equivalence["centered_logit_max_difference"]
                    <= PREFLIGHT_LOGIT_TOLERANCE
                    and equivalence["probability_max_difference"]
                    <= PREFLIGHT_PROBABILITY_TOLERANCE
                    and peak_allocated < vram_limit_bytes
                    and codebook_valid
                )
                else "wrapper_failure",
            }
        )
    inventory["task_codebooks"] = codebooks
    inventory["source_tree_hash"] = source_tree_hash()
    inventory["official_equivalence_items"] = len(rows)
    inventory["generation_sanity_items"] = int(
        pd.DataFrame(rows)["generation_valid"].notna().sum()
    )
    inventory[
        "preflight_selection"
    ] = "shared_task_by_four_length_bins_with_per_model_p95_p99_anchors"
    inventory["model_load_seconds"] = load_seconds
    inventory["vram_limit_bytes"] = vram_limit_bytes
    inventory["gpu_name"] = device_properties.name
    inventory["gpu_total_memory_bytes"] = device_properties.total_memory
    safe_name = args.model_id.replace("/", "--")
    write_table(pd.DataFrame(rows), args.output_dir / (safe_name + ".parquet"))
    write_json(args.output_dir / (safe_name + "_inventory.json"), inventory)
    frame = pd.DataFrame(rows)
    print(
        frame
        .groupby("task_id")
        .agg(
            items=("item_id", "size"),
            max_tokens=("prompt_tokens", "max"),
            seconds=("seconds_per_view", "mean"),
        )
    )
    failed = frame[frame["status"] != "completed"]
    if not failed.empty:
        diagnostics = {
            "normalization": int(
                (failed["probability_sum_error"] > PREFLIGHT_NORMALIZATION_TOLERANCE).sum()
            ),
            "determinism": int(
                (failed["deterministic_max_difference"] > PREFLIGHT_DETERMINISM_TOLERANCE).sum()
            ),
            "official_logits": int(
                (failed["centered_logit_max_difference"] > PREFLIGHT_LOGIT_TOLERANCE).sum()
            ),
            "official_probabilities": int(
                (failed["official_probability_max_difference"] > PREFLIGHT_PROBABILITY_TOLERANCE).sum()
            ),
            "vram": int(
                (failed["peak_allocated_bytes"] >= failed["vram_limit_bytes"]).sum()
            ),
            "codebook": int(
                failed["code_token_lengths"].map(
                    lambda value: any(
                        int(length) <= 0 for length in str(value).split("|")
                    )
                    or (
                        args.model_id != "mistralai/Mistral-7B-Instruct-v0.3"
                        and any(
                            int(length) != 1
                            for length in str(value).split("|")
                        )
                    )
                ).sum()
            ),
            "generation": int(
                (
                    failed["generation_valid"].notna()
                    & ~failed["generation_valid"].astype(bool)
                ).sum()
            ),
        }
        raise RuntimeError("preflight contract failed: %s" % diagnostics)


if __name__ == "__main__":
    main()
