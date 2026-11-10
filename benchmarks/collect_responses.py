"""Collect responses for the corrected protocol without modifying the paper release.

Use the original revision-pinned scorer. By default reuse original responses only
after a numerical bridge on predeclared original-fit items. --recompute-all creates
an entirely new response bank and does not mix old and new numerical runtimes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reference_implementation"
CANONICAL = BUNDLE / "data/canonical"
sys.path.insert(0, str(BUNDLE))
from kernel_isqed.response import coordinates_to_probabilities, scores_to_coordinates

KEY = ["task_id", "item_id"]
DIMENSIONS = {"mmlu_pro": 9, "hellaswag": 3, "rewardbench": 1}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def checkpoint(path: Path, old: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    incoming = pd.DataFrame(rows)
    result = pd.concat([old, incoming], ignore_index=True) if not old.empty else incoming
    if result.duplicated(KEY).any():
        raise RuntimeError("Duplicate response keys in collection checkpoint")
    tmp = path.with_suffix(".partial.parquet")
    result.to_parquet(tmp, index=False)
    tmp.replace(path)
    return result


def verify_frozen_inputs(splits_dir: Path) -> str:
    freeze_path = splits_dir / "freeze_record.json"
    freeze = json.loads(freeze_path.read_text())
    protocol_path = splits_dir / "protocol.json"
    if digest(protocol_path) != freeze["protocol_sha256"]:
        raise RuntimeError("Split protocol differs from its freeze record")
    artifact_hashes = json.loads((splits_dir / "artifact_hashes.json").read_text())
    for name, expected in artifact_hashes.items():
        if digest(splits_dir / name) != expected:
            raise RuntimeError(f"Frozen split artifact changed: {name}")
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["source_files_sha256"].items():
        if digest(BUNDLE / name) != expected:
            raise RuntimeError(f"Original source changed: {name}")
    return digest(freeze_path)


def score_row(scorer, row) -> np.ndarray:
    count = DIMENSIONS[row.task_id] + 1
    result = scorer.score(row.system_prompt, row.user_prompt, [str(i) for i in range(count)])
    if set(result.code_token_lengths) != {1}:
        raise RuntimeError("Scoring no longer satisfies single-token answer contract")
    permutation = np.asarray(json.loads(row.permutation), dtype=int)
    semantic = np.empty(count, dtype=np.float64)
    semantic[permutation] = result.log_scores
    return scores_to_coordinates(semantic)[0]


def bridge(scorer, original: pd.DataFrame, prompts: pd.DataFrame, count: int) -> dict:
    """Compare original-fit records selected independently of model response values."""
    records = []
    candidates = original[original["split"].eq("fit")].copy()
    candidates["bridge_order"] = candidates["item_id"].map(
        lambda x: hashlib.sha256(("benchmarks-bridge-v1|" + x).encode()).hexdigest()
    )
    candidates = candidates.sort_values("bridge_order").groupby("task_id", sort=True).head(count)
    prompt_columns = KEY + ["system_prompt", "user_prompt", "permutation"]
    candidates = candidates.merge(prompts[prompt_columns], on=KEY, validate="one_to_one")
    if len(candidates) != count * len(DIMENSIONS):
        raise RuntimeError("Numerical bridge is missing original-fit items")
    for row in candidates.itertuples(index=False):
        coordinates = score_row(scorer, row)
        expected = np.asarray([getattr(row, f"coord_{i}") for i in range(len(coordinates))])
        p = coordinates_to_probabilities(coordinates)[0]
        q = coordinates_to_probabilities(expected)[0]
        # The original protocol uses centered-logit and probability tolerances.
        from kernel_isqed.response import helmert_matrix
        centered_difference = (coordinates - expected) @ helmert_matrix(len(coordinates) + 1)
        records.append({
            "task_id": row.task_id, "item_id": row.item_id,
            "centered_logit_max_difference": float(np.abs(centered_difference).max()),
            "probability_max_difference": float(np.abs(p - q).max()),
            "tv": float(np.abs(p - q).sum() / 2),
        })
    passed = all(r["centered_logit_max_difference"] <= 0.02 and
                 r["probability_max_difference"] <= 0.002 for r in records)
    return {"passed": passed, "items_per_task": count, "records": records,
            "centered_logit_tolerance": 0.02, "probability_tolerance": 0.002}


def check_official_forward(scorer, original: pd.DataFrame, prompts: pd.DataFrame) -> dict:
    """Check the selective answer scorer against the official model forward."""
    candidates = original[original["split"].eq("fit")][KEY].copy()
    candidates["check_order"] = candidates["item_id"].map(
        lambda x: hashlib.sha256(("benchmarks-official-check-v1|" + x).encode()).hexdigest()
    )
    candidates = candidates.sort_values("check_order").groupby("task_id", sort=True).head(4)
    candidates = candidates.merge(prompts[KEY + ["system_prompt", "user_prompt"]], on=KEY, validate="one_to_one")
    if len(candidates) != 12:
        raise RuntimeError("Official-forward check is missing original-fit items")
    records = []
    for row in candidates.itertuples(index=False):
        result = scorer.compare_official_logits(row.system_prompt, row.user_prompt,
                                                [str(i) for i in range(DIMENSIONS[row.task_id] + 1)])
        records.append({"task_id": row.task_id, "item_id": row.item_id, **result})
    return {"passed": all(r["centered_logit_max_difference"] <= 0.02 and
                          r["probability_max_difference"] <= 0.002 for r in records),
            "items_per_task": 4, "records": records,
            "centered_logit_tolerance": 0.02, "probability_tolerance": 0.002}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "benchmarks/splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/response_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bridge-items-per-task", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=0, help="Smoke run; incomplete cache stays explicitly incomplete")
    parser.add_argument("--recompute-all", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    revisions = json.loads((BUNDLE / "configs/llm_revisions.json").read_text())["models"]
    if args.model_id not in revisions:
        raise ValueError("Model is outside the frozen model pool")
    if args.bridge_items_per_task < 1 and not args.recompute_all:
        raise ValueError("Reusing original responses requires a numerical bridge")
    freeze_hash = verify_frozen_inputs(args.splits_dir)
    name = args.model_id.replace("/", "--")
    required = pd.read_parquet(args.splits_dir / "required_items.parquet")
    prompts = pd.read_parquet(CANONICAL / "prompt_manifest.parquet")
    if not {"system_prompt", "user_prompt", "permutation"}.issubset(required):
        required = required.merge(prompts[KEY + ["system_prompt", "user_prompt", "permutation"]], on=KEY, validate="one_to_one")
    if required.duplicated(KEY).any():
        raise ValueError("Required item union has duplicate keys")
    fields = ["prompt_hash", "system_prompt", "user_prompt", "permutation"]
    joined = required[KEY + fields].merge(prompts[KEY + fields], on=KEY,
                                          suffixes=("_required", "_original"),
                                          how="left", validate="one_to_one")
    if any(not joined[f"{field}_required"].eq(joined[f"{field}_original"]).all() for field in fields):
        raise RuntimeError("Required prompts differ from the frozen original prompt text/permutation/hash")
    original = pd.read_parquet(CANONICAL / "response_cache" / (name + ".parquet"))
    if set(original["revision"]) != {revisions[args.model_id]}:
        raise RuntimeError("Original response revision mismatch")
    mode = "recompute_all" if args.recompute_all else "bridge_and_reuse"
    output_dir = args.output_dir / mode
    path = output_dir / (name + ".parquet")
    provenance_path = output_dir / (name + ".provenance.json")
    code_hashes = {"collector_sha256": digest(Path(__file__)),
                   "original_scorer_sha256": digest(BUNDLE / "kernel_isqed/llm.py"),
                   "response_math_sha256": digest(BUNDLE / "kernel_isqed/response.py"),
                   "contracts_sha256": digest(BUNDLE / "kernel_isqed/contracts.py")}
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        if not provenance_path.is_file():
            raise RuntimeError("Cache exists without provenance")
        provenance = json.loads(provenance_path.read_text())
        if (provenance["revision"] != revisions[args.model_id] or
            provenance["split_freeze_sha256"] != freeze_hash or provenance["mode"] != mode):
            raise RuntimeError("Checkpoint belongs to a different protocol")
        if any(provenance.get(key) != value for key, value in code_hashes.items()):
            raise RuntimeError("Scoring code changed since this response checkpoint was created")
        if set(existing["model_id"]) != {args.model_id} or set(existing["revision"]) != {revisions[args.model_id]}:
            raise RuntimeError("Checkpoint contains a different model or revision")
    bank = existing if args.recompute_all else pd.concat([original, existing], ignore_index=True)
    if not bank.empty:
        if bank.duplicated(KEY).any():
            raise RuntimeError("Response bank contains duplicate model/item keys")
        matched = required[KEY + ["prompt_hash"]].merge(bank[KEY + ["prompt_hash"]], on=KEY, suffixes=("_required", "_cache"))
        if not matched["prompt_hash_required"].eq(matched["prompt_hash_cache"]).all():
            raise RuntimeError("Cached response has a different prompt hash")
        completed = pd.MultiIndex.from_frame(bank[KEY])
        pending = required[~pd.MultiIndex.from_frame(required[KEY]).isin(completed)].copy()
    else:
        pending = required.copy()
    pending = pending.sort_values(KEY)
    plan = {"model_id": args.model_id, "revision": revisions[args.model_id], "mode": mode,
            "required_items": len(required), "pending_items": len(pending),
            "pending_by_task": pending.groupby("task_id").size().to_dict(),
            "output": str(path), "split_freeze_sha256": freeze_hash}
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent two workers from writing the same model's checkpoint.
    lock = output_dir / (name + ".lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        import torch
        from kernel_isqed.llm import SemanticOptionScorer
        torch.manual_seed(20260722)
        runtime = {x: importlib.metadata.version(x) for x in
                   ["torch", "transformers", "numpy", "pandas", "huggingface-hub", "tokenizers"]}
        runtime.update({"python": sys.version, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(torch.device(args.device))})
        if not existing.empty and provenance.get("runtime") != runtime:
            raise RuntimeError("Refusing to mix numerical runtimes in a response checkpoint")
        scorer = SemanticOptionScorer(args.model_id, revisions[args.model_id], args.device)
        official_check = check_official_forward(scorer, original, prompts)
        bridge_result = None if args.recompute_all or not official_check["passed"] else bridge(scorer, original, prompts, args.bridge_items_per_task)
        provenance = {**plan, "runtime": runtime, "bridge": bridge_result, **code_hashes,
                      "official_forward_check": official_check,
                      "completed": False}
        atomic_json(provenance_path, provenance)
        if not official_check["passed"]:
            raise RuntimeError("Selective scorer does not match official forward within the original protocol tolerances")
        if bridge_result is not None and not bridge_result["passed"]:
            raise RuntimeError("Bridge failed: use --recompute-all for a separate, uniform-runtime bank; no new responses were mixed with the original cache")
        if args.max_items:
            pending = pending.head(args.max_items)
        started = time.monotonic()
        rows = []
        for index, row in enumerate(pending.itertuples(index=False), 1):
            coordinates = score_row(scorer, row)
            result = {"model_id": args.model_id, "revision": revisions[args.model_id],
                      "task_id": row.task_id, "item_id": row.item_id, "prompt_hash": row.prompt_hash,
                      "split": "corrected_protocol_union", "response_source": "benchmarks_collection",
                      "item_cluster": getattr(row, "item_cluster", ""), "domain": getattr(row, "domain", ""),
                      "gold": int(getattr(row, "gold", -1))}
            result.update({f"coord_{i}": float(coordinates[i]) if i < len(coordinates) else np.nan for i in range(9)})
            rows.append(result)
            if len(rows) >= 50:
                existing = checkpoint(path, existing, rows)
                rows = []
                print(f"cached {index}/{len(pending)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
        if rows:
            existing = checkpoint(path, existing, rows)
        provenance["completed"] = len(pending) == plan["pending_items"]
        provenance["new_cache_rows"] = len(existing)
        provenance["collection_seconds"] = time.monotonic() - started
        atomic_json(provenance_path, provenance)
        print(json.dumps({"completed": provenance["completed"], "new_cache_rows": len(existing)}), flush=True)
    finally:
        os.close(lock_fd)
        lock.unlink()


if __name__ == "__main__":
    main()
