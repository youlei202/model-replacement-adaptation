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

from kernel_isqed.contracts import PEERS, TARGETS, TASKS, validate_model_revision
from kernel_isqed.io import (
    CANONICAL_DIR,
    read_json,
    sha256_file,
    stable_hash,
    write_table,
)
from kernel_isqed.llm import SemanticOptionScorer
from kernel_isqed.frozen import source_tree_hash, verify_frozen_inputs
from kernel_isqed.response import (
    SemanticItem,
    canonical_permutation,
    render_prompt,
    scores_to_coordinates,
)


def _validated_preflight(model_id: str, revision: str) -> str:
    safe_name = model_id.replace("/", "--")
    result_path = CANONICAL_DIR / "preflight" / (safe_name + ".parquet")
    inventory_path = CANONICAL_DIR / "preflight" / (safe_name + "_inventory.json")
    if not result_path.is_file() or not inventory_path.is_file():
        raise RuntimeError("robustness caching requires a completed preflight")
    results = pd.read_parquet(result_path)
    inventory = read_json(inventory_path)
    if len(results) != 256 or set(results["status"].astype(str)) != {"completed"}:
        raise RuntimeError(
            "robustness caching requires a successful 256-item preflight"
        )
    if (
        inventory.get("requested_revision") != revision
        or inventory.get("source_tree_hash") != source_tree_hash()
    ):
        raise RuntimeError("preflight inventory belongs to another revision")
    return sha256_file(inventory_path)


def _write_checkpoint(existing: pd.DataFrame, rows, path: Path) -> pd.DataFrame:
    incoming = pd.DataFrame(rows)
    output = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    output = output.drop_duplicates(
        ["model_id", "task_id", "item_id", "split", "view_id"], keep="last"
    )
    temporary = path.with_suffix(".partial.parquet")
    write_table(output, temporary)
    temporary.replace(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache the frozen option-permutation robustness views"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=CANONICAL_DIR / "robustness_cache"
    )
    args = parser.parse_args()
    validate_model_revision(
        args.model_id, args.revision, read_json(CANONICAL_DIR / "protocol_lock.json")
    )
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    unlock_path = CANONICAL_DIR / "test_unlock.json"
    if not freeze_path.is_file() or not unlock_path.is_file():
        raise RuntimeError(
            "robustness views stay sealed until the one-time test unlock"
        )
    freeze = read_json(freeze_path)
    unlock = read_json(unlock_path)
    freeze_digest = sha256_file(freeze_path)
    if (
        unlock.get("unlock_count") != 1
        or unlock.get("state_freeze_sha256") != freeze_digest
        or unlock.get("source_tree_hash") != freeze.get("source_tree_hash")
        or source_tree_hash() != freeze.get("source_tree_hash")
    ):
        raise RuntimeError("robustness unlock does not match the frozen state release")
    verify_frozen_inputs(freeze)
    roles = {model.model_id: model.role for model in PEERS + TARGETS}
    if args.model_id not in roles:
        raise ValueError("model is not part of the frozen pool")
    inventory_sha256 = _validated_preflight(args.model_id, args.revision)

    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    split = split[split["split"].isin(("fit", "validation", "iid_test", "ood_test"))]
    selected = []
    view_counts = {"mmlu_pro": 3, "hellaswag": 3, "rewardbench": 1}
    for task in TASKS:
        part = split[split["task_id"] == task.task_id].copy()
        part["order"] = part["item_id"].map(
            lambda value: stable_hash([task.task_id, value, "robustness"])
        )
        selected.append(part.sort_values("order").head(512))
    frame = pd.concat(selected, ignore_index=True)
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
            existing["state_freeze_sha256"].astype(str) == freeze_digest
            if "state_freeze_sha256" in existing
            else pd.Series(False, index=existing.index)
        )
        existing = existing[
            (existing["model_id"].astype(str) == args.model_id)
            & (existing["revision"].astype(str) == args.revision)
            & same_preflight
            & same_freeze
        ].copy()
    completed = (
        set(map(tuple, existing[["task_id", "item_id", "split", "view_id"]].to_numpy()))
        if not existing.empty
        else set()
    )
    scorer = SemanticOptionScorer(args.model_id, args.revision, args.device)
    rows = []
    processed = 0
    for item in frame.itertuples(index=False):
        options = tuple(str(value) for value in json.loads(item.options))
        semantic = SemanticItem(
            item.task_id,
            item.item_id,
            item.context,
            options,
            int(item.gold),
            item.domain,
        )
        for view_index in range(view_counts[item.task_id]):
            view_id = "permutation_%d" % (view_index + 1)
            if (item.task_id, item.item_id, item.split, view_id) in completed:
                continue
            permutation = canonical_permutation(
                item.item_id, len(options), salt="robustness_%d" % view_index
            )
            system_prompt, user_prompt = render_prompt(semantic, permutation)
            result = scorer.score(
                system_prompt,
                user_prompt,
                [str(value) for value in range(len(options))],
            )
            if any(length != 1 for length in result.code_token_lengths):
                raise RuntimeError(
                    "formal robustness cache requires a single-token answer codebook"
                )
            semantic_scores = np.empty(len(options), dtype=np.float64)
            semantic_scores[np.asarray(permutation, dtype=int)] = result.log_scores
            coordinates = scores_to_coordinates(semantic_scores)[0]
            row = {
                "model_id": args.model_id,
                "role": roles[args.model_id],
                "revision": args.revision,
                "task_id": item.task_id,
                "item_id": item.item_id,
                "split": item.split,
                "view_id": view_id,
                "permutation": "|".join(str(value) for value in permutation),
                "prompt_hash": stable_hash([system_prompt, user_prompt]),
                "preflight_inventory_sha256": inventory_sha256,
                "state_freeze_sha256": freeze_digest,
            }
            for coordinate_index in range(9):
                row["coord_%d" % coordinate_index] = (
                    float(coordinates[coordinate_index])
                    if coordinate_index < len(coordinates)
                    else np.nan
                )
            rows.append(row)
            processed += 1
            if len(rows) >= 100:
                existing = _write_checkpoint(existing, rows, path)
                rows = []
            if args.limit > 0 and processed >= args.limit:
                break
        if args.limit > 0 and processed >= args.limit:
            break

    output = _write_checkpoint(existing, rows, path)
    print(output.groupby(["task_id", "view_id"]).size().rename("views").to_string())
    print("saved:", path)


if __name__ == "__main__":
    main()
