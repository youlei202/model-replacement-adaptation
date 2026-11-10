from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import RESPONSE_MODELS, validate_model_revision
from kernel_isqed.io import CANONICAL_DIR, read_json, sha256_file, write_table
from kernel_isqed.llm import SemanticOptionScorer
from kernel_isqed.frozen import source_tree_hash, verify_frozen_inputs
from kernel_isqed.response import scores_to_coordinates


def _validated_preflight(model_id: str, revision: str) -> str:
    safe_name = model_id.replace("/", "--")
    result_path = CANONICAL_DIR / "preflight" / (safe_name + ".parquet")
    inventory_path = CANONICAL_DIR / "preflight" / (safe_name + "_inventory.json")
    if not result_path.is_file() or not inventory_path.is_file():
        raise RuntimeError(
            "response caching requires a completed preflight for this model"
        )
    results = pd.read_parquet(result_path)
    inventory = read_json(inventory_path)
    if (
        len(results) != 256
        or set(results["status"].astype(str)) != {"completed"}
        or inventory.get("requested_revision") != revision
        or inventory.get("official_equivalence_items") != 256
        or inventory.get("source_tree_hash") != source_tree_hash()
    ):
        raise RuntimeError(
            "preflight is incomplete or belongs to another model revision"
        )
    return sha256_file(inventory_path)


def _write_checkpoint(existing: pd.DataFrame, rows, path: Path) -> pd.DataFrame:
    incoming = pd.DataFrame(rows)
    output = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    output = output.drop_duplicates(
        ["model_id", "task_id", "item_id", "split"], keep="last"
    )
    temporary = path.with_suffix(".partial.parquet")
    write_table(output, temporary)
    temporary.replace(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache one LLM endpoint on the frozen semantic response contract"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("fit_validation", "test"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=CANONICAL_DIR / "response_cache"
    )
    args = parser.parse_args()
    validate_model_revision(
        args.model_id, args.revision, read_json(CANONICAL_DIR / "protocol_lock.json")
    )
    state_freeze_digest = ""
    if args.stage == "test":
        freeze_path = CANONICAL_DIR / "state_freeze.json"
        unlock_path = CANONICAL_DIR / "test_unlock.json"
        if not freeze_path.is_file() or not unlock_path.is_file():
            raise RuntimeError(
                "test responses stay sealed until the registered one-time unlock"
            )
        freeze = read_json(freeze_path)
        unlock = read_json(unlock_path)
        state_freeze_digest = sha256_file(freeze_path)
        if (
            unlock.get("unlock_count") != 1
            or unlock.get("state_freeze_sha256") != state_freeze_digest
        ):
            raise RuntimeError(
                "test unlock marker does not match the frozen state release"
            )
        if source_tree_hash() != freeze.get("source_tree_hash") or unlock.get(
            "source_tree_hash"
        ) != freeze.get("source_tree_hash"):
            raise RuntimeError("source code changed after the one-time test unlock")
        verify_frozen_inputs(freeze)
        checks = {
            "serialized_states_sha256": CANONICAL_DIR / "serialized_states.parquet",
            "expected_run_keys_sha256": CANONICAL_DIR / "expected_run_keys.parquet",
        }
        for field, path in checks.items():
            if not path.is_file() or sha256_file(path) != freeze.get(field):
                raise RuntimeError("frozen test-unlock hash mismatch: %s" % path)
    roles = {model.model_id: model.role for model in RESPONSE_MODELS}
    if args.model_id not in roles:
        raise ValueError("model is not part of the frozen pool")
    inventory_sha256 = _validated_preflight(args.model_id, args.revision)
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    prompts = pd.read_parquet(CANONICAL_DIR / "prompt_manifest.parquet")
    split_columns = ["task_id", "item_id", "split", "item_cluster", "domain", "gold"]
    frame = prompts.merge(split[split_columns], on=["task_id", "item_id"], how="inner")
    allowed = (
        ("fit", "validation")
        if args.stage == "fit_validation"
        else ("iid_test", "ood_test")
    )
    frame = frame[frame["split"].isin(allowed)].sort_values(["task_id", "item_id"])
    if args.limit > 0:
        frame = frame.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (args.model_id.replace("/", "--") + ".parquet")
    existing = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    if not existing.empty:
        same_preflight = (
            existing["preflight_inventory_sha256"].astype(str) == inventory_sha256
            if "preflight_inventory_sha256" in existing
            else pd.Series(False, index=existing.index)
        )
        same_freeze = (
            existing["state_freeze_sha256"].astype(str) == state_freeze_digest
            if "state_freeze_sha256" in existing
            else pd.Series(False, index=existing.index)
        )
        stage_release = (
            ~existing["split"].isin(("iid_test", "ood_test")) | same_freeze
            if args.stage == "test"
            else pd.Series(True, index=existing.index)
        )
        existing = existing[
            (existing["model_id"].astype(str) == args.model_id)
            & (existing["revision"].astype(str) == args.revision)
            & same_preflight
            & stage_release
        ].copy()
        completed = set(
            map(
                tuple,
                existing[["task_id", "item_id", "split"]].drop_duplicates().to_numpy(),
            )
        )
        pending = frame.apply(
            lambda row: (row["task_id"], row["item_id"], row["split"]) not in completed,
            axis=1,
        )
        frame = frame[pending]
    if frame.empty:
        print("cache already contains every requested item:", path)
        return
    scorer = SemanticOptionScorer(args.model_id, args.revision, args.device)
    rows = []
    processed = 0
    for _, row in frame.iterrows():
        option_count = {"mmlu_pro": 10, "hellaswag": 4, "rewardbench": 2}[
            row["task_id"]
        ]
        result = scorer.score(
            row["system_prompt"],
            row["user_prompt"],
            [str(value) for value in range(option_count)],
        )
        permutation = np.asarray(json.loads(row["permutation"]), dtype=int)
        semantic_scores = np.empty(option_count, dtype=np.float64)
        semantic_scores[permutation] = result.log_scores
        coordinates = scores_to_coordinates(semantic_scores)[0]
        output = {
            "model_id": args.model_id,
            "role": roles[args.model_id],
            "revision": args.revision,
            "task_id": row["task_id"],
            "item_id": row["item_id"],
            "split": row["split"],
            "item_cluster": row["item_cluster"],
            "domain": row["domain"],
            "gold": int(row["gold"]),
            "prompt_text": row["user_prompt"],
            "prompt_hash": row["prompt_hash"],
            "preflight_inventory_sha256": inventory_sha256,
            "state_freeze_sha256": state_freeze_digest,
            "code_token_lengths": "|".join(
                str(value) for value in result.code_token_lengths
            ),
        }
        for coordinate_index in range(9):
            output["coord_%d" % coordinate_index] = (
                float(coordinates[coordinate_index])
                if coordinate_index < len(coordinates)
                else np.nan
            )
        rows.append(output)
        processed += 1
        if len(rows) >= 100:
            print("cached", processed, "/", len(frame))
            existing = _write_checkpoint(existing, rows, path)
            rows = []
    _write_checkpoint(existing, rows, path)
    print("saved:", path)


if __name__ == "__main__":
    main()
