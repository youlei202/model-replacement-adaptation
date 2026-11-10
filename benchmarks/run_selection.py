#!/usr/bin/env python3
"""Fit, freeze, and evaluate corrected budget-aware peer selection.

Example (all paths may be absolute):
  python benchmarks/run_selection.py --response-dir benchmarks/response_cache/bridge_and_reuse --response-mode supplemental --stage plan
  python benchmarks/run_selection.py --response-dir benchmarks/response_cache/bridge_and_reuse --response-mode supplemental --stage fit
  python benchmarks/run_selection.py --response-dir benchmarks/response_cache/bridge_and_reuse --response-mode supplemental --stage evaluate

``run`` fits/freezes every requested cell before evaluating any test response.
Resuming validates input/code hashes and all frozen binary states. The original
experiment directory is read-only throughout. Outputs are per seed/target/task.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from selection_core import (
    CODECS, LAMBDAS, RANKS, Context, decode, encode, fit_weights, make_context,
    paired_cluster_interval, predict, probabilities, state_bits, tv,
)

ROOT = Path(__file__).resolve().parents[1]
PEERS = ("Qwen/Qwen3-8B", "allenai/Olmo-3-7B-Instruct", "mistralai/Ministral-3-8B-Instruct-2512-BF16")
TARGETS = ("Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "ibm-granite/granite-3.3-8b-instruct")
DIMENSIONS = {"mmlu_pro": 9, "hellaswag": 3, "rewardbench": 1}
EXPECTED_COUNTS = {"mmlu_pro": (2048, 512, 1024), "hellaswag": (2048, 512, 1024), "rewardbench": (1024, 256, 512)}
SPLITS = ("fit", "validation", "iid_test")
PAIRS = tuple(itertools.combinations(range(3), 2)) + ((0, 1, 2),)
BUDGETS = (1024, 2048, 8192, 16384)
KEYS = ["model_id", "task_id", "item_id", "prompt_hash"]
PROTOCOL = "replacement_corrected_peer_selection_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def verify_split_protocol(directory: Path) -> dict:
    freeze = json.loads((directory / "freeze_record.json").read_text())
    protocol_path = directory / "protocol.json"
    if sha(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("Split protocol differs from its frozen digest")
    hashes = json.loads((directory / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or sha(path) != expected:
            raise ValueError(f"Frozen split artifact changed: {name}")
    protocol = json.loads(protocol_path.read_text())
    bundle = ROOT / "reference_implementation"
    for name, expected in protocol["source_files_sha256"].items():
        if sha(bundle / name) != expected:
            raise ValueError(f"Original source artifact changed: {name}")
    return {"protocol_sha256": sha(protocol_path), "freeze_sha256": sha(directory / "freeze_record.json"), "artifact_manifest_sha256": sha(directory / "artifact_hashes.json")}


def read_manifests(directory: Path, seeds: list[int] | None, tasks: list[str]) -> tuple[pd.DataFrame, dict]:
    files = sorted(directory.glob("split_seed_*.parquet"))
    if seeds:
        files = [p for p in files if int(p.stem.rsplit("_", 1)[1]) in seeds]
    if not files or (seeds and len(files) != len(set(seeds))):
        raise ValueError("Requested split manifest(s) missing")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    required = {"task_id", "item_id", "item_cluster", "domain", "gold", "prompt_hash", "split", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest lacks {sorted(required - set(frame.columns))}")
    frame = frame[frame.task_id.isin(tasks) & frame.split.isin(SPLITS)].copy()
    if frame.duplicated(["seed", "task_id", "item_id"]).any():
        raise ValueError("Item appears in multiple partitions of a split seed")
    overlap = frame.groupby(["seed", "task_id", "item_cluster"]).split.nunique()
    if (overlap > 1).any():
        raise ValueError("Cluster leakage between fit/validation/test")
    for (seed, task), values in frame.groupby(["seed", "task_id"]):
        counts = tuple(int((values.split == s).sum()) for s in SPLITS)
        if counts != EXPECTED_COUNTS[task]:
            raise ValueError(f"Unexpected counts for {seed}/{task}: {counts}")
        if "population_weight" in values and values.population_weight.nunique() > 1:
            raise ValueError("This runner expects uniform cluster-representative sampling")
    expected = {(int(p.stem.rsplit("_", 1)[1]), t) for p in files for t in tasks}
    actual = set(zip(frame.seed.astype(int), frame.task_id))
    if actual != expected:
        raise ValueError("Requested seed/task cell missing from manifests")
    return frame, {str(p.resolve()): sha(p) for p in files}


def read_responses(paths: list[Path], models: tuple[str, ...], expected_revisions: dict | None = None) -> tuple[pd.DataFrame, dict]:
    files = sorted({p.resolve() for source in paths for p in (source.rglob("*.parquet") if source.is_dir() else [source])})
    if not files:
        raise ValueError("No response parquet files found")
    frames, hashes = [], {}
    for path in files:
        d = pd.read_parquet(path)
        required = set(KEYS) | {"revision", "coord_0"}
        if not required.issubset(d.columns):
            raise ValueError(f"Not a response cache: {path}")
        columns = KEYS + ["revision"] + [f"coord_{i}" for i in range(9) if f"coord_{i}" in d]
        frames.append(d.loc[d.model_id.isin(models), columns])
        hashes[str(path)] = sha(path)
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    if frame.duplicated(KEYS).any():
        bad = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(5).to_dict("records")
        raise ValueError(f"Conflicting cached responses for identical prompt: {bad}")
    if frame.duplicated(["model_id", "task_id", "item_id"]).any():
        raise ValueError("Multiple prompt hashes for the same model/item in response bank")
    if (frame.groupby("model_id").revision.nunique() > 1).any():
        raise ValueError("Multiple checkpoint revisions within a model")
    if expected_revisions is not None:
        for model, values in frame.groupby("model_id"):
            if set(values.revision) != {expected_revisions[model]}:
                raise ValueError(f"Response revision differs from pinned checkpoint: {model}")
    return frame, hashes


def response_sources(directory: Path, mode: str, models: tuple[str, ...], manifest_dir: Path, revisions: dict) -> tuple[list[Path], dict, list[str]]:
    """Explicitly choose one cache mode; never search an alternate directory."""
    original = ROOT / "reference_implementation/data/canonical/response_cache"
    expected_mode = {"supplemental": "bridge_and_reuse", "complete": "recompute_all"}[mode]
    freeze_hash = sha(manifest_dir / "freeze_record.json")
    sources, provenance_hashes, issues = [], {}, []
    for model in models:
        name = model.replace("/", "--")
        path = directory / f"{name}.parquet"
        provenance_path = directory / f"{name}.provenance.json"
        if mode == "supplemental":
            sources.append(original / f"{name}.parquet")
        if not path.exists():
            issues.append(f"{model}: response shard not yet present")
            continue
        if not provenance_path.exists():
            raise ValueError(f"Cache lacks provenance: {path}")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("mode") != expected_mode or provenance.get("split_freeze_sha256") != freeze_hash or provenance.get("model_id") != model or provenance.get("revision") != revisions[model]:
            raise ValueError(f"Cache provenance does not match requested mode/splits/model: {path}")
        expected_sources = {"collector_sha256": ROOT / "benchmarks/collect_responses.py", "original_scorer_sha256": ROOT / "reference_implementation/kernel_isqed/llm.py"}
        for field, source_path in expected_sources.items():
            if provenance.get(field) != sha(source_path):
                raise ValueError(f"Cache source fingerprint mismatch ({field}): {path}")
        if mode == "supplemental" and not (provenance.get("bridge") or {}).get("passed"):
            raise ValueError(f"Numerical bridge not passed: {path}")
        if not (provenance.get("official_forward_check") or {}).get("passed"):
            raise ValueError(f"Selective scorer official-forward check not passed: {path}")
        if not provenance.get("completed"):
            issues.append(f"{model}: response collection not complete")
        sources.append(path)
        provenance_hashes[str(provenance_path.resolve())] = sha(provenance_path)
    return sources, provenance_hashes, issues


def coverage(manifest: pd.DataFrame, responses: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    required = manifest[["task_id", "item_id", "prompt_hash"]].drop_duplicates()
    required = required.merge(pd.DataFrame({"model_id": models}), how="cross")
    covered = required.merge(responses[KEYS], on=KEYS, how="left", indicator=True, validate="one_to_one")
    return covered.loc[covered._merge == "left_only", KEYS]


def arrays(manifest: pd.DataFrame, responses: pd.DataFrame, seed: int, task: str, target: str, splits: tuple[str, ...]) -> dict:
    result = {}
    for split in splits:
        rows = manifest[(manifest.seed == seed) & (manifest.task_id == task) & (manifest.split == split)].sort_values("item_id").reset_index(drop=True)
        model_arrays = []
        for model in PEERS + (target,):
            source = responses[responses.model_id == model]
            joined = rows[["task_id", "item_id", "prompt_hash"]].merge(source, on=["task_id", "item_id", "prompt_hash"], how="left", validate="one_to_one", indicator=True, sort=False)
            if not (joined._merge == "both").all():
                raise ValueError(f"Missing responses: {seed}/{task}/{model}/{split}")
            x = joined[[f"coord_{i}" for i in range(DIMENSIONS[task])]].to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite coordinates: {task}/{model}/{split}")
            model_arrays.append(x)
        result[split] = {"rows": rows, "peers": np.stack(model_arrays[:3], axis=1), "target": model_arrays[3]}
    return result


def pair_name(pair) -> str:
    return "-".join(str(x) for x in pair)


def cell_path(output: Path, seed: int, task: str, target: str) -> Path:
    return output / f"seed_{seed}" / task / target.replace("/", "--")


def get_context(output: Path, seed: int, task: str, pair: tuple[int, ...], fit: np.ndarray, validation: np.ndarray, contract: str) -> tuple[Context, Path]:
    directory = output / "public_contexts" / contract / f"{seed}_{task}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"peers_{pair_name(pair)}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            context = Context(pair, data["mean"], data["scale"], float(data["sigma"]), data["features"], data["eigenvalues"], data["eigenvectors"], data["validation_projection"])
        if len(context.features) != len(fit) or len(context.validation_projection) != len(validation):
            raise ValueError("Cached context has wrong split sizes")
        return context, path
    log(f"Building public context {seed}/{task}/peers={pair_name(pair)}")
    context = make_context(fit, validation, pair)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as f:
        np.savez(f, **{k: v for k, v in asdict(context).items() if k != "pair"})
    temp.replace(path)
    return context, path


def choice_key(record: dict) -> tuple:
    return (record["validation_tv"], record["actual_bits"], record["rank"], -LAMBDAS[record["lambda_index"]], record["codec"], record["state_digest"])


def enumerate_pair(data: dict, pair: tuple[int, ...], context: Context, budgets: tuple[int, ...], state_dir: Path, quick: bool = False) -> tuple[list[dict], list[dict]]:
    """Enumerate on fit/validation only. Return best per family/budget and ledger."""
    x, y = data["fit"]["peers"], data["fit"]["target"]
    vx, vy = data["validation"]["peers"], data["validation"]["target"]
    vprob = probabilities(vy)
    best, ledger = {}, []
    largest = max(budgets)
    ranks = sorted(set([r for r in RANKS if r < len(x)] + [len(x)]))
    codecs = CODECS
    lambda_indices = range(len(LAMBDAS))
    if quick:
        # Only used by synthetic unit tests, never exposed by the CLI.
        ranks, codecs, lambda_indices = [min(8, len(x)), len(x)], ("q3", "q8", "native_fp64"), (0, 8, 16)
    projected_target = context.eigenvectors.T @ y

    def consider(matrix, family, codec, li, native_full=False):
        rows, columns = np.asarray(matrix).reshape(len(matrix), -1).shape
        if not native_full and state_bits(family, codec, rows, columns) > largest:
            return
        try:
            encoded = encode(matrix, family, pair, codec, li)
        except ValueError as error:
            if str(error) not in ("Native precision overflow", "Scale overflow"):
                raise
            ledger.append({"family": family, "pair": list(pair), "codec": codec, "rank": rows, "lambda_index": li, "status": "unrepresentable", "reason": str(error)})
            return
        decoded, actual = decode(encoded.blob)
        prediction = predict(decoded, actual, vx, context.validation_projection)
        if not np.isfinite(prediction).all():
            raise ValueError("Decoded candidate gives nonfinite validation prediction")
        error = float(np.abs(vprob - probabilities(prediction)).sum(1).mean() / 2)
        record = {"family": family, "pair": list(pair), "q": len(pair), "codec": codec, "rank": encoded.rank, "lambda_index": li, "actual_bits": encoded.bits, "meaningful_bits": encoded.meaningful_bits, "padding_bits": encoded.bits - encoded.meaningful_bits, "validation_tv": error, "state_digest": encoded.digest, "native_full_precision": bool(native_full), "status": "eligible"}
        ledger.append(record)
        eligible = [b for b in budgets if encoded.bits <= b]
        if native_full:
            eligible += [-1]
        for budget in eligible:
            key = (family, budget)
            if key not in best or choice_key(record) < choice_key(best[key]):
                best[key] = record
                path = state_dir / f"{encoded.digest}.bin"
                if not path.exists():
                    path.write_bytes(encoded.blob)

    for family in ("convex", "nnls"):
        weights = fit_weights(x, y, pair, family).reshape(-1, 1)
        for codec in codecs:
            consider(weights, family, codec, 0, native_full=codec == "native_fp64")
    for li in lambda_indices:
        spectral = projected_target / (context.eigenvalues[:, None] + len(x) * LAMBDAS[li])
        for rank in ranks:
            for codec in codecs:
                consider(spectral[:rank], "krr", codec, li, native_full=(rank == len(x) and codec == "native_fp64"))
    selected = []
    for (family, budget), record in sorted(best.items()):
        selected.append(record | {"budget_bits": budget, "selection_kind": "pair_family"})
    return selected, ledger


def selection_policies(pair_rows: list[dict], budgets: tuple[int, ...]) -> list[dict]:
    """Validation-only decisions; all family sets and budget weights fixed here."""
    result = []
    for scope, families in (("krr", ("krr",)), ("augmented", ("krr", "convex", "nnls"))):
        by_pair_budget = {}
        for row in pair_rows:
            if row["family"] not in families:
                continue
            key = (tuple(row["pair"]), row["budget_bits"])
            if key not in by_pair_budget or choice_key(row) < choice_key(by_pair_budget[key]):
                by_pair_budget[key] = row
        q2 = [pair for pair in PAIRS if len(pair) == 2]
        fixed_full = min(q2, key=lambda p: (choice_key(by_pair_budget[(p, -1)]), p))
        fixed_mean = min(q2, key=lambda p: (sum(by_pair_budget[(p, b)]["validation_tv"] for b in BUDGETS) / len(BUDGETS), p))
        for budget in budgets:
            if budget not in BUDGETS:
                continue  # 848-bit replication is q=3 only.
            dynamic = min((by_pair_budget[(p, budget)] for p in q2), key=lambda r: (choice_key(r), r["pair"]))
            for name, row in (("budget_aware", dynamic), ("fixed_full_precision", by_pair_budget[(fixed_full, budget)]), ("fixed_mean_budget", by_pair_budget[(fixed_mean, budget)])):
                result.append(row | {"selection_kind": "policy", "scope": scope, "policy": name, "budget_bits": budget, "fixed_full_pair": list(fixed_full), "fixed_mean_pair": list(fixed_mean)})
    for budget in BUDGETS:
        nnls_choice = min((row for row in pair_rows if row["family"] == "nnls" and row["q"] == 2 and row["budget_bits"] == budget), key=lambda row: (choice_key(row), row["pair"]))
        result.append(nnls_choice | {"selection_kind": "calibration_policy", "scope": "nnls", "policy": "budget_aware", "budget_bits": budget})
    return result


def validate_freeze(path: Path, expected_contract: str) -> dict:
    freeze = json.loads(path.read_text())
    if freeze["contract"] != expected_contract:
        raise ValueError(f"Inputs or code changed for checkpoint {path}; choose a new --output")
    for relative, expected in freeze["artifacts"].items():
        artifact = (path.parent / relative).resolve()
        if not artifact.is_file() or sha(artifact) != expected:
            raise ValueError(f"Frozen artifact missing or modified: {artifact}")
    return freeze


def fit_cell(output: Path, manifest: pd.DataFrame, responses: pd.DataFrame, seed: int, task: str, target: str, contract: str) -> Path:
    directory = cell_path(output, seed, task, target)
    freeze_path = directory / "selection_freeze.json"
    if freeze_path.exists():
        validate_freeze(freeze_path, contract)
        log(f"Resume: verified selection freeze {seed}/{task}/{target}")
        return freeze_path
    directory.mkdir(parents=True, exist_ok=True)
    state_dir = directory / "states"
    state_dir.mkdir(exist_ok=True)
    data = arrays(manifest, responses, seed, task, target, ("fit", "validation"))
    budgets = BUDGETS + ((848,) if task == "hellaswag" and target == "google/gemma-4-12B-it" else ())
    pair_rows, ledger, contexts = [], [], {}
    for pair in PAIRS:
        context, context_path = get_context(output, seed, task, pair, data["fit"]["peers"], data["validation"]["peers"], contract)
        rows, candidates = enumerate_pair(data, pair, context, budgets if len(pair) == 3 else BUDGETS, state_dir)
        pair_rows.extend(rows)
        ledger.extend(candidates)
        contexts[pair_name(pair)] = os.path.relpath(context_path, directory)
        log(f"Fitted {seed}/{task}/{target}/peers={pair_name(pair)}; {len(candidates)} candidates")
    policies = selection_policies(pair_rows, budgets)
    for pair in PAIRS:
        validation_prediction = data["validation"]["peers"][:, pair, :].mean(axis=1)
        pair_rows.append({"selection_kind": "predetermined_zero_state", "family": "fixed_mean", "pair": list(pair), "q": len(pair), "budget_bits": 0, "actual_bits": 0, "meaningful_bits": 0, "padding_bits": 0, "validation_tv": float(tv(data["validation"]["target"], validation_prediction).mean()), "state_digest": "public_mean_" + pair_name(pair)})
    atomic_parquet(directory / "validation_candidates.parquet", pd.DataFrame(ledger))
    rows = pair_rows + policies
    selected_digests = {r["state_digest"] for r in rows if not r["state_digest"].startswith("public_")}
    # Delete only intermediate best-so-far states produced inside this new cell.
    for path in state_dir.glob("*.bin"):
        if path.stem not in selected_digests:
            path.unlink()
    artifacts = {f"states/{d}.bin": sha(state_dir / f"{d}.bin") for d in sorted(selected_digests)}
    artifacts["validation_candidates.parquet"] = sha(directory / "validation_candidates.parquet")
    for relative in contexts.values():
        artifacts[relative] = sha((directory / relative).resolve())
    freeze = {"protocol": PROTOCOL, "contract": contract, "seed": seed, "task": task, "target": target, "peers": list(PEERS), "budgets": list(budgets), "fixed_mean_budget_weights": {str(b): 1 / len(BUDGETS) for b in BUDGETS}, "contexts": contexts, "selections": rows, "artifacts": artifacts, "test_evaluated": False, "created_unix": time.time()}
    atomic_json(freeze_path, freeze)
    log(f"Froze all decisions before test evaluation: {freeze_path}")
    return freeze_path


def evaluate_cell(path: Path, manifest: pd.DataFrame, responses: pd.DataFrame, contract: str, bootstrap_draws: int) -> None:
    freeze = validate_freeze(path, contract)
    directory = path.parent
    done_path = directory / "evaluation_complete.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done["freeze_sha256"] != sha(path) or done["bootstrap_draws"] != bootstrap_draws:
            raise ValueError("Evaluation checkpoint settings differ")
        for name, expected in done["outputs"].items():
            if sha(directory / name) != expected:
                raise ValueError("Evaluation output changed")
        log(f"Resume: verified evaluation {directory}")
        return
    data = arrays(manifest, responses, freeze["seed"], freeze["task"], freeze["target"], ("iid_test",))["iid_test"]
    metadata, peers, target = data["rows"], data["peers"], data["target"]
    loss_by_digest, projections = {}, {}
    selections = freeze["selections"]
    for row in selections:
        digest = row["state_digest"]
        if digest in loss_by_digest:
            continue
        pair = tuple(row["pair"])
        if digest.startswith("public_mean_"):
            prediction = peers[:, pair, :].mean(axis=1)
        else:
            encoded, matrix = decode((directory / "states" / f"{digest}.bin").read_bytes())
            if encoded.digest != digest or encoded.bits != row["actual_bits"] or encoded.pair != pair:
                raise ValueError("Frozen selection and binary state disagree")
            if encoded.family == "krr" and pair not in projections:
                context_path = (directory / freeze["contexts"][pair_name(pair)]).resolve()
                with np.load(context_path, allow_pickle=False) as c:
                    context = Context(pair, c["mean"], c["scale"], float(c["sigma"]), c["features"], c["eigenvalues"], c["eigenvectors"], np.empty((0, 0)))
                projections[pair] = context.project(peers)
            prediction = predict(encoded, matrix, peers, projections.get(pair))
        loss_by_digest[digest] = tv(target, prediction)
    scores = []
    for row in selections:
        scores.append(row | {"test_tv": float(loss_by_digest[row["state_digest"]].mean()), "seed": freeze["seed"], "task_id": freeze["task"], "target_id": freeze["target"]})
    losses = []
    for digest, values in loss_by_digest.items():
        frame = metadata[["item_id", "item_cluster", "domain", "gold", "prompt_hash"]].copy()
        frame["state_digest"], frame["tv_loss"] = digest, values
        losses.append(frame)
    atomic_parquet(directory / "item_losses.parquet", pd.concat(losses, ignore_index=True))
    atomic_parquet(directory / "test_scores.parquet", pd.DataFrame(scores))
    pd.DataFrame(scores).to_csv(directory / "test_scores.csv", index=False)
    contrasts = []
    policy_rows = [r for r in scores if r["selection_kind"] == "policy"]
    for scope in ("krr", "augmented"):
        for budget in BUDGETS:
            selected = {r["policy"]: r for r in policy_rows if r["scope"] == scope and r["budget_bits"] == budget}
            aware = selected["budget_aware"]
            for policy in ("fixed_full_precision", "fixed_mean_budget"):
                fixed = selected[policy]
                delta = loss_by_digest[fixed["state_digest"]] - loss_by_digest[aware["state_digest"]]
                mean, low, high = paired_cluster_interval(delta, metadata.item_cluster.to_numpy(), draws=bootstrap_draws, seed=int(freeze["seed"]))
                contrasts.append({"seed": freeze["seed"], "task_id": freeze["task"], "target_id": freeze["target"], "scope": scope, "budget_bits": budget, "comparator": policy, "fixed_pair": fixed["pair"], "budget_aware_pair": aware["pair"], "fixed_tv": fixed["test_tv"], "budget_aware_tv": aware["test_tv"], "gain_tv": mean, "ci_low": low, "ci_high": high, "fixed_state_digest": fixed["state_digest"], "aware_state_digest": aware["state_digest"], "uncertainty": "paired cluster bootstrap conditional on this split and frozen selections"})
    pd.DataFrame(contrasts).to_csv(directory / "paired_gains.csv", index=False)
    direct = []

    def add_direct(category, left, right):
        difference = loss_by_digest[left["state_digest"]] - loss_by_digest[right["state_digest"]]
        mean, low, high = paired_cluster_interval(difference, metadata.item_cluster.to_numpy(), draws=bootstrap_draws, seed=int(freeze["seed"]))
        direct.append({"seed": freeze["seed"], "task_id": freeze["task"], "target_id": freeze["target"], "comparison": category, "budget_bits": left["budget_bits"], "q": left["q"], "left_family": left["family"], "right_family": right["family"], "left_pair": left["pair"], "right_pair": right["pair"], "left_tv": left["test_tv"], "right_tv": right["test_tv"], "left_minus_right_tv": mean, "ci_low": low, "ci_high": high, "left_state_digest": left["state_digest"], "right_state_digest": right["state_digest"], "uncertainty": "paired cluster bootstrap conditional on this split and frozen selections; unadjusted descriptive intervals"})

    family_rows = {(tuple(r["pair"]), r["family"], r["budget_bits"]): r for r in scores if r["selection_kind"] == "pair_family"}
    for budget in (-1,) + BUDGETS:
        for family in ("krr", "convex", "nnls"):
            for left_pair, right_pair in itertools.combinations([p for p in PAIRS if len(p) == 2], 2):
                add_direct("all_q2_peer_pairs", family_rows[(left_pair, family, budget)], family_rows[(right_pair, family, budget)])
    for pair in PAIRS:
        for budget in (-1,) + tuple(freeze["budgets"]):
            if (pair, "nnls", budget) in family_rows:
                add_direct("nnls_minus_krr_same_peers", family_rows[(pair, "nnls", budget)], family_rows[(pair, "krr", budget)])
    for budget in BUDGETS:
        nnls_winner = next(r for r in scores if r["selection_kind"] == "calibration_policy" and r["budget_bits"] == budget)
        krr_winner = next(r for r in policy_rows if r["scope"] == "krr" and r["policy"] == "budget_aware" and r["budget_bits"] == budget)
        add_direct("validation_selected_nnls_minus_krr_q2", nnls_winner, krr_winner)
    pd.DataFrame(direct).to_csv(directory / "paired_direct_contrasts.csv", index=False)
    outputs = ("item_losses.parquet", "test_scores.parquet", "test_scores.csv", "paired_gains.csv", "paired_direct_contrasts.csv")
    atomic_json(done_path, {"freeze_sha256": sha(path), "bootstrap_draws": bootstrap_draws, "outputs": {name: sha(directory / name) for name in outputs}, "created_unix": time.time()})
    log(f"Evaluated frozen cell: {directory}")


def aggregate(output: Path, paths: list[Path]) -> None:
    available = [p.parent / "paired_gains.csv" for p in paths if (p.parent / "evaluation_complete.json").exists()]
    if not available:
        return
    gains = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
    gains.to_csv(output / "paired_gains_all.csv", index=False)
    direct = pd.concat([pd.read_csv(p.parent / "paired_direct_contrasts.csv") for p in available], ignore_index=True)
    direct.to_csv(output / "paired_direct_contrasts_all.csv", index=False)
    group = ["task_id", "target_id", "scope", "budget_bits", "comparator"]
    summary = gains.groupby(group, as_index=False).agg(split_count=("gain_tv", "size"), mean_gain_tv=("gain_tv", "mean"), min_gain_tv=("gain_tv", "min"), max_gain_tv=("gain_tv", "max"), positive_splits=("gain_tv", lambda x: int((x > 0).sum())), positive_interval_splits=("ci_low", lambda x: int((x > 0).sum())))
    summary["uncertainty_note"] = "Descriptive across overlapping split seeds; no independent-seed CI is claimed."
    summary.to_csv(output / "gain_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "benchmarks/splits")
    parser.add_argument("--response-dir", type=Path, required=True, help="Exact directory containing model parquet files and collector provenance")
    parser.add_argument("--response-mode", choices=("supplemental", "complete"), required=True, help="Supplemental uses bridge-validated original + new rows; complete uses only the supplied directory")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results")
    parser.add_argument("--stage", choices=("plan", "fit", "evaluate", "run"), default="plan")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--tasks", choices=tuple(DIMENSIONS), nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--targets", choices=TARGETS, nargs="+", default=list(TARGETS))
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.blas_threads < 1 or args.bootstrap_draws < 100:
        parser.error("--blas-threads must be positive and --bootstrap-draws at least 100")
    args.output.mkdir(parents=True, exist_ok=True)
    split_provenance = verify_split_protocol(args.manifest_dir)
    manifest, manifest_hashes = read_manifests(args.manifest_dir, args.seeds, args.tasks)
    revisions_path = ROOT / "reference_implementation/configs/llm_revisions.json"
    revisions = json.loads(revisions_path.read_text())["models"]
    source_paths, provenance_hashes, provenance_issues = response_sources(args.response_dir, args.response_mode, PEERS + tuple(args.targets), args.manifest_dir, revisions)
    if source_paths:
        responses, response_hashes = read_responses(source_paths, PEERS + tuple(args.targets), expected_revisions=revisions)
    else:
        responses, response_hashes = pd.DataFrame(columns=KEYS + ["revision", "coord_0"]), {}
    protocol = {"protocol": PROTOCOL, "source_hashes": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("selection_core.py"))}, "split_provenance": split_provenance, "pinned_revisions_sha256": sha(revisions_path), "manifest_hashes": manifest_hashes, "response_hashes": response_hashes, "response_provenance_hashes": provenance_hashes, "response_mode": args.response_mode, "response_dir": str(args.response_dir.resolve()), "peers": PEERS, "targets": args.targets, "tasks": args.tasks, "budgets": BUDGETS, "lambda_grid": LAMBDAS, "rank_grid": RANKS, "codecs": CODECS, "count_actual_padded_bytes": True, "blas_threads": args.blas_threads, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    contract = digest_json(protocol)
    missing = coverage(manifest, responses, PEERS + tuple(args.targets))
    atomic_parquet(args.output / "missing_responses.parquet", missing)
    seeds = sorted(int(x) for x in manifest.seed.unique())
    jobs = [(seed, task, target) for seed in seeds for task in args.tasks for target in args.targets]
    plan = {"contract": contract, "cells": len(jobs), "seeds": seeds, "tasks": args.tasks, "targets": args.targets, "required_response_rows": len(manifest[["task_id", "item_id", "prompt_hash"]].drop_duplicates()) * (len(PEERS) + len(args.targets)), "missing_response_rows": len(missing), "missing_by_model_task": missing.groupby(["model_id", "task_id"]).size().rename("count").reset_index().to_dict("records"), "provenance_issues": provenance_issues, "protocol": protocol}
    atomic_json(args.output / "run_plan.json", plan)
    log(json.dumps({k: v for k, v in plan.items() if k != "protocol"}, ensure_ascii=False))
    if len(missing):
        raise SystemExit("Missing matched responses: see missing_responses.parquet. No fit/selection/evaluation ran; no rows were dropped.")
    if provenance_issues:
        raise SystemExit("Response collection provenance incomplete; no fit/selection/evaluation ran.")
    if args.stage == "plan":
        return
    paths = []
    with threadpool_limits(limits=args.blas_threads):
        if args.stage in ("fit", "run"):
            for seed, task, target in jobs:
                paths.append(fit_cell(args.output, manifest, responses, seed, task, target, contract))
        else:
            paths = [cell_path(args.output, seed, task, target) / "selection_freeze.json" for seed, task, target in jobs]
        if args.stage in ("evaluate", "run"):
            # Validate all freezes before constructing test evaluation arrays.
            for path in paths:
                validate_freeze(path, contract)
            for path in paths:
                evaluate_cell(path, manifest, responses, contract, args.bootstrap_draws)
            aggregate(args.output, paths)


if __name__ == "__main__":
    main()
