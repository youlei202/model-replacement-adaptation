#!/usr/bin/env python3
"""Summarize the complete 45-cell campaign without selecting favorable cells.

This script refuses to read test results until the coordinator reports success
and all 45 evaluation markers exist. Across-seed ranges are descriptive: the
five seeds share items, so this script never pools their observations or CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(20261109, 20261114))
TASKS = ("mmlu_pro", "hellaswag", "rewardbench")
TARGETS = (
    "Qwen/Qwen3.5-9B", "google/gemma-4-12B-it",
    "ibm-granite/granite-3.3-8b-instruct",
)
BUDGETS = (1024, 2048, 8192, 16384)
PEER_NAMES = ("Qwen3", "Olmo3", "Ministral3")
PAIRS = tuple(itertools.combinations(range(3), 2)) + ((0, 1, 2),)
FAMILIES = ("krr", "convex", "nnls")
CELL = ["seed", "task_id", "target_id"]
GROUP = ["task_id", "target_id", "scope", "budget_bits", "comparator"]
TOLERANCE = 1e-12
UNCERTAINTY = (
    "Five overlapping split seeds; mean/min/max and sign counts are descriptive. "
    "Per-seed confidence intervals are paired cluster-bootstrap intervals conditional "
    "on that split and its frozen selections, with no multiple-comparison adjustment. "
    "No pooled item-seed or independent-seed confidence interval is computed."
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(int(x) for x in value)


def pair_label(value) -> str:
    return "+".join(PEER_NAMES[x] for x in pair_tuple(value))


def normalize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("pair", "fixed_pair", "budget_aware_pair", "left_pair", "right_pair"):
        if column in result:
            result[column] = result[column].map(pair_tuple)
            result[column + "_names"] = result[column].map(pair_label)
    return result


def completed_inputs(results: Path, campaign_path: Path) -> tuple[list[Path], dict]:
    """Check the full gate before opening any performance table."""
    expected = [results / f"seed_{seed}" / task / target.replace("/", "--")
                for seed, task, target in itertools.product(SEEDS, TASKS, TARGETS)]
    complete = sum((p / "evaluation_complete.json").is_file() for p in expected)
    if not campaign_path.is_file():
        raise RuntimeError(f"Campaign is not complete: no coordinator status; {complete}/45 cell markers.")
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("stage") != "complete" or campaign.get("analysis_exit_code") != 0 or complete != 45:
        raise RuntimeError(
            f"Campaign is not complete: coordinator={campaign.get('stage')}, "
            f"analysis_exit_code={campaign.get('analysis_exit_code')}, {complete}/45 cell markers. "
            "No test-result table was opened."
        )
    fingerprints, contracts, checked_artifacts = {}, set(), {}
    for directory in expected:
        marker_path = directory / "evaluation_complete.json"
        freeze_path = directory / "selection_freeze.json"
        marker = json.loads(marker_path.read_text())
        if marker["freeze_sha256"] != sha(freeze_path):
            raise ValueError(f"Selection freeze changed after evaluation: {directory}")
        freeze = json.loads(freeze_path.read_text())
        contracts.add(freeze["contract"])
        expected_path = results / f"seed_{freeze['seed']}" / freeze["task"] / freeze["target"].replace("/", "--")
        if expected_path != directory:
            raise ValueError(f"Cell identity does not match path: {directory}")
        fingerprints[str(marker_path)] = sha(marker_path)
        fingerprints[str(freeze_path)] = sha(freeze_path)
        for name, expected_hash in freeze["artifacts"].items():
            path = (directory / name).resolve()
            if results.resolve() not in path.parents:
                raise ValueError(f"Frozen artifact leaves the result directory: {path}")
            if path not in checked_artifacts:
                if not path.is_file():
                    raise ValueError(f"Frozen artifact is missing: {path}")
                checked_artifacts[path] = sha(path)
            if checked_artifacts[path] != expected_hash:
                raise ValueError(f"Frozen state or public context changed: {path}")
            fingerprints[str(path)] = expected_hash
        for name, expected_hash in marker["outputs"].items():
            path = directory / name
            if not path.is_file() or sha(path) != expected_hash:
                raise ValueError(f"Completed output changed: {path}")
            fingerprints[str(path)] = expected_hash
    if len(contracts) != 1:
        raise ValueError("The 45 cells do not share one frozen analysis contract")
    return expected, {"analysis_contract": next(iter(contracts)),
                      "campaign_status_sha256": sha(campaign_path),
                      "source_sha256": fingerprints}


def load_results(directories: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores, gains, direct = [], [], []
    for directory in directories:
        scores.append(pd.read_parquet(directory / "test_scores.parquet"))
        gains.append(pd.read_csv(directory / "paired_gains.csv"))
        direct.append(pd.read_csv(directory / "paired_direct_contrasts.csv"))
    frames = tuple(normalize_pairs(pd.concat(parts, ignore_index=True)) for parts in (scores, gains, direct))
    validate_tables(*frames)
    return frames


def validate_tables(scores: pd.DataFrame, gains: pd.DataFrame, direct: pd.DataFrame) -> None:
    """Validate exact frozen comparison coverage, not only aggregate row counts."""
    expected_cells = set(itertools.product(SEEDS, TASKS, TARGETS))
    for label, frame in (("scores", scores), ("policy gains", gains), ("direct contrasts", direct)):
        if set(frame[CELL].itertuples(index=False, name=None)) != expected_cells:
            raise ValueError(f"{label} do not contain all 45 expected cells")
    expected_gains = {cell + (scope, budget, comparator) for cell in expected_cells
                      for scope, budget, comparator in itertools.product(
                          ("krr", "augmented"), BUDGETS,
                          ("fixed_full_precision", "fixed_mean_budget"))}
    keys = CELL + ["scope", "budget_bits", "comparator"]
    if gains.duplicated(keys).any() or set(gains[keys].itertuples(index=False, name=None)) != expected_gains:
        raise ValueError("Unexpected, missing, or duplicate policy contrasts")
    expected_policies = {cell + (scope, budget, policy) for cell in expected_cells
                         for scope, budget, policy in itertools.product(
                             ("krr", "augmented"), BUDGETS,
                             ("budget_aware", "fixed_full_precision", "fixed_mean_budget"))}
    policies = scores[scores.selection_kind == "policy"]
    keys = CELL + ["scope", "budget_bits", "policy"]
    if policies.duplicated(keys).any() or set(policies[keys].itertuples(index=False, name=None)) != expected_policies:
        raise ValueError("Unexpected, missing, or duplicate frozen policies")
    valid_kinds = {"pair_family", "policy", "calibration_policy", "predetermined_zero_state"}
    if set(scores.selection_kind) != valid_kinds:
        raise ValueError("Unknown or missing score selection kind")
    if not scores.q.eq(scores.pair.map(len)).all():
        raise ValueError("Reported call count differs from the selected peer set")
    if not (direct.q.eq(direct.left_pair.map(len)) & direct.q.eq(direct.right_pair.map(len))).all():
        raise ValueError("A direct comparison does not match peer-call counts")
    finite_budgets = scores.budget_bits >= 0
    if (scores.loc[finite_budgets, "actual_bits"] > scores.loc[finite_budgets, "budget_bits"]).any():
        raise ValueError("A selected state exceeds its reported budget")
    for cell, group in scores.groupby(CELL, sort=False):
        compact = cell[1] == "hellaswag" and cell[2] == TARGETS[1]
        expected_pair_rows = {(family, pair, budget) for family, pair, budget in itertools.product(FAMILIES, PAIRS, (-1,) + BUDGETS)}
        if compact:
            expected_pair_rows |= {(family, (0, 1, 2), 848) for family in FAMILIES}
        pair_rows = group[group.selection_kind == "pair_family"]
        keys = ["family", "pair", "budget_bits"]
        if pair_rows.duplicated(keys).any() or set(pair_rows[keys].itertuples(index=False, name=None)) != expected_pair_rows:
            raise ValueError(f"Incomplete or duplicate pair/family/budget matrix: {cell}")
        calibration = group[group.selection_kind == "calibration_policy"]
        if len(calibration) != 4 or set(calibration.budget_bits) != set(BUDGETS) or set(calibration.family) != {"nnls"} or set(calibration.q) != {2}:
            raise ValueError(f"Invalid frozen calibration policy rows: {cell}")
        zero = group[group.selection_kind == "predetermined_zero_state"]
        if len(zero) != 4 or set(zero.pair) != set(PAIRS) or not (zero.actual_bits == 0).all():
            raise ValueError(f"Invalid predetermined zero-state references: {cell}")
    allowed_categories = {"all_q2_peer_pairs", "nnls_minus_krr_same_peers", "validation_selected_nnls_minus_krr_q2"}
    if set(direct.comparison) != allowed_categories:
        raise ValueError("Unknown or missing direct-contrast category")
    for cell, group in direct.groupby(CELL, sort=False):
        compact = cell[1] == "hellaswag" and cell[2] == TARGETS[1]
        pairwise = group[group.comparison == "all_q2_peer_pairs"]
        expected = {(family, family, left, right, budget) for family, (left, right), budget in itertools.product(
            FAMILIES, tuple(itertools.combinations(PAIRS[:3], 2)), (-1,) + BUDGETS)}
        keys = ["left_family", "right_family", "left_pair", "right_pair", "budget_bits"]
        if pairwise.duplicated(keys).any() or set(pairwise[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Incomplete all-pair paired contrasts: {cell}")
        calibration = group[group.comparison == "nnls_minus_krr_same_peers"]
        expected = {("nnls", "krr", pair, pair, budget) for pair, budget in itertools.product(PAIRS, (-1,) + BUDGETS)}
        if compact:
            expected.add(("nnls", "krr", (0, 1, 2), (0, 1, 2), 848))
        if calibration.duplicated(keys).any() or set(calibration[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Incomplete matched-peer calibration contrasts: {cell}")
        selected = group[group.comparison == "validation_selected_nnls_minus_krr_q2"]
        if len(selected) != 4 or set(selected.budget_bits) != set(BUDGETS) or set(selected.q) != {2} or set(selected.left_family) != {"nnls"} or set(selected.right_family) != {"krr"}:
            raise ValueError(f"Incomplete selected calibration contrasts: {cell}")
    for frame, left, right, difference in ((gains, "fixed_tv", "budget_aware_tv", "gain_tv"), (direct, "left_tv", "right_tv", "left_minus_right_tv")):
        numeric = frame[[left, right, difference, "ci_low", "ci_high"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.allclose(frame[left] - frame[right], frame[difference], atol=TOLERANCE, rtol=0):
            raise ValueError("Paired contrast values do not match their displayed losses")


def policy_lookup(scores: pd.DataFrame) -> dict:
    rows = scores[scores.selection_kind == "policy"]
    keys = CELL + ["scope", "budget_bits", "policy"]
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate frozen policy rows")
    return {tuple(row[key] for key in keys): row for row in rows.to_dict("records")}


def enrich_gains(scores: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    lookup = policy_lookup(scores)
    rows = []
    for row in gains.to_dict("records"):
        key = tuple(row[name] for name in CELL + ["scope", "budget_bits"])
        aware = lookup[key + ("budget_aware",)]
        fixed = lookup[key + (row["comparator"],)]
        if aware["state_digest"] != row["aware_state_digest"] or fixed["state_digest"] != row["fixed_state_digest"]:
            raise ValueError("Policy score and paired contrast refer to different states")
        if not np.allclose([aware["test_tv"], fixed["test_tv"]], [row["budget_aware_tv"], row["fixed_tv"]], atol=TOLERANCE, rtol=0):
            raise ValueError("Policy score and paired contrast report different losses")
        row.update({
            "budget_aware_family": aware["family"], "fixed_family": fixed["family"],
            "budget_aware_actual_bits": int(aware["actual_bits"]),
            "fixed_actual_bits": int(fixed["actual_bits"]),
            "budget_aware_codec": aware["codec"], "fixed_codec": fixed["codec"],
            "different_peer_pair": pair_tuple(aware["pair"]) != pair_tuple(fixed["pair"]),
            "gain_tv_percentage_points": 100 * row["gain_tv"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in gains.groupby(GROUP, sort=True):
        if set(group.seed) != set(SEEDS):
            raise ValueError(f"A contrast does not include all five seeds: {key}")
        delta = group.gain_tv.to_numpy()
        mean = float(delta.mean())
        positives = int((delta > TOLERANCE).sum())
        rows.append(dict(zip(GROUP, key)) | {
            "split_count": len(group), "budget_aware_tv_mean": float(group.budget_aware_tv.mean()),
            "fixed_tv_mean": float(group.fixed_tv.mean()),
            "gain_tv_mean": mean, "gain_tv_min": float(delta.min()), "gain_tv_max": float(delta.max()),
            "gain_tv_percentage_points_mean": 100 * mean,
            "positive_gain_splits": positives,
            "zero_gain_splits": int((np.abs(delta) <= TOLERANCE).sum()),
            "negative_gain_splits": int((delta < -TOLERANCE).sum()),
            "positive_paired_interval_splits": int((group.ci_low > 0).sum()),
            "different_peer_pair_splits": int(group.different_peer_pair.sum()),
            "nonpositive_mean_gain": mean <= TOLERANCE,
            "positive_gain_in_every_split": positives == len(SEEDS),
            "uncertainty_note": UNCERTAINTY,
        })
    return pd.DataFrame(rows)


def switch_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic = scores[(scores.selection_kind == "policy") & (scores.policy == "budget_aware")]
    rows = []
    for key, group in dynamic.groupby(CELL + ["scope"], sort=True):
        group = group.sort_values("budget_bits")
        if tuple(group.budget_bits) != BUDGETS:
            raise ValueError("A dynamic policy lacks one of the four fixed budgets")
        pairs = [pair_tuple(x) for x in group.pair]
        families = group.family.tolist()
        switches = sum(left != right for left, right in zip(pairs, pairs[1:]))
        rows.append(dict(zip(CELL + ["scope"], key)) | {
            "budget_bits_sequence": json.dumps(BUDGETS),
            "peer_pair_sequence": json.dumps([pair_label(p) for p in pairs]),
            "adapter_family_sequence": json.dumps(families),
            "distinct_peer_pairs": len(set(pairs)), "adjacent_peer_switches": switches,
            "any_peer_switch": switches > 0,
            "adjacent_family_switches": sum(a != b for a, b in zip(families, families[1:])),
        })
    per_seed = pd.DataFrame(rows)
    grouped = per_seed.groupby(["task_id", "target_id", "scope"], as_index=False).agg(
        split_count=("seed", "size"), seeds_with_any_peer_switch=("any_peer_switch", "sum"),
        total_adjacent_peer_switches=("adjacent_peer_switches", "sum"),
        min_adjacent_peer_switches=("adjacent_peer_switches", "min"),
        max_adjacent_peer_switches=("adjacent_peer_switches", "max"),
    )
    grouped["note"] = "A change in validation-selected peer pair alone is not evidence of a held-out gain."
    return per_seed, grouped


def direct_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["task_id", "target_id", "comparison", "q", "budget_bits"]
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        if set(group.seed) != set(SEEDS) or len(group) != len(SEEDS):
            raise ValueError("A calibration comparison is missing or duplicating seeds")
        delta = group.left_minus_right_tv.to_numpy()
        rows.append(dict(zip(keys, key)) | {
            "split_count": len(group), "nnls_tv_mean": float(group.left_tv.mean()),
            "krr_tv_mean": float(group.right_tv.mean()),
            "nnls_minus_krr_tv_mean": float(delta.mean()),
            "nnls_minus_krr_tv_min": float(delta.min()), "nnls_minus_krr_tv_max": float(delta.max()),
            "nnls_better_splits": int((delta < -TOLERANCE).sum()),
            "tied_splits": int((np.abs(delta) <= TOLERANCE).sum()),
            "krr_better_splits": int((delta > TOLERANCE).sum()),
            "nnls_no_worse_mean": float(delta.mean()) <= TOLERANCE,
            "krr_positive_paired_interval_splits": int((group.ci_low > 0).sum()),
            "nnls_positive_paired_interval_splits": int((group.ci_high < 0).sum()),
            "uncertainty_note": UNCERTAINTY,
        })
    return pd.DataFrame(rows)


def compact_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = scores[(scores.task_id == "hellaswag") & (scores.target_id == TARGETS[1]) & (scores.q == 3)]
    selected = cell[((cell.selection_kind == "pair_family") & cell.budget_bits.isin((-1, 848, 1024))) |
                    (cell.selection_kind == "predetermined_zero_state")].copy()
    full = selected[(selected.family == "krr") & (selected.budget_bits == -1)]
    if len(full) != len(SEEDS) or full.seed.duplicated().any():
        raise ValueError("Missing Gemma/HellaSwag full-precision KRR reference")
    selected["full_precision_krr_test_tv"] = selected.seed.map(full.set_index("seed").test_tv)
    selected["tv_minus_full_precision_krr"] = selected.test_tv - selected.full_precision_krr_test_tv
    selected["budget_label"] = selected.budget_bits.map(lambda b: "full_precision" if b == -1 else str(int(b)))
    fields = CELL + ["family", "q", "budget_bits", "budget_label", "actual_bits", "meaningful_bits",
                     "padding_bits", "codec", "rank", "validation_tv", "test_tv", "full_precision_krr_test_tv",
                     "tv_minus_full_precision_krr", "state_digest"]
    per_seed = selected.reindex(columns=fields).sort_values(["seed", "family", "budget_bits"])
    grouped = per_seed.groupby(["family", "q", "budget_bits", "budget_label"], as_index=False).agg(
        split_count=("seed", "size"), actual_bits_min=("actual_bits", "min"), actual_bits_max=("actual_bits", "max"),
        test_tv_mean=("test_tv", "mean"), test_tv_min=("test_tv", "min"), test_tv_max=("test_tv", "max"),
        tv_minus_full_precision_krr_mean=("tv_minus_full_precision_krr", "mean"),
        tv_minus_full_precision_krr_min=("tv_minus_full_precision_krr", "min"),
        tv_minus_full_precision_krr_max=("tv_minus_full_precision_krr", "max"),
    )
    grouped["reference_note"] = "Full-precision KRR is an empirical reference, not an irreducible error floor."
    return per_seed, grouped


def calibration_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    key = ["task_id", "target_id", "budget_bits", "comparator"]
    values = ["gain_tv_mean", "gain_tv_min", "gain_tv_max", "positive_gain_splits", "positive_paired_interval_splits"]
    krr = summary[summary.scope == "krr"][key + values]
    augmented = summary[summary.scope == "augmented"][key + values]
    result = krr.merge(augmented, on=key, suffixes=("_krr_only", "_with_calibration"), validate="one_to_one")
    def label(row):
        if row.gain_tv_mean_with_calibration <= TOLERANCE:
            return "no_positive_mean_gain_with_calibration_available"
        if row.positive_gain_splits_with_calibration < len(SEEDS):
            return "positive_augmented_mean_but_not_positive_in_every_split"
        return "positive_augmented_gain_in_all_five_overlapping_splits"
    result["descriptive_status"] = result.apply(label, axis=1)
    result["krr_positive_mean_gain_not_retained_with_calibration"] = (
        (result.gain_tv_mean_krr_only > TOLERANCE) & (result.gain_tv_mean_with_calibration <= TOLERANCE)
    )
    result["note"] = "The augmented menu gives BOTH dynamic and fixed-peer policies KRR, convex and NNLS at the same two-call budget."
    return result


def build_tables(scores: pd.DataFrame, gains: pd.DataFrame, direct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    enriched = enrich_gains(scores, gains)
    summary = gain_summary(enriched)
    primary = (enriched.task_id == "mmlu_pro") & (enriched.target_id == TARGETS[0])
    primary_summary = (summary.task_id == "mmlu_pro") & (summary.target_id == TARGETS[0])
    switches, switch_counts = switch_tables(scores)
    q3 = direct[(direct.comparison == "nnls_minus_krr_same_peers") & (direct.q == 3)].copy()
    q2 = direct[direct.comparison == "validation_selected_nnls_minus_krr_q2"].copy()
    compact, compact_summary = compact_tables(scores)
    pair_scores = scores[(scores.selection_kind == "pair_family") & (scores.q == 2)].copy()
    pair_summary = pair_scores.groupby(["task_id", "target_id", "family", "pair_names", "budget_bits"], as_index=False).agg(
        split_count=("seed", "size"), test_tv_mean=("test_tv", "mean"), test_tv_min=("test_tv", "min"),
        test_tv_max=("test_tv", "max"), actual_bits_min=("actual_bits", "min"), actual_bits_max=("actual_bits", "max"),
    )
    return {
        "primary_qwen_mmlu_per_seed.csv": enriched[primary].sort_values(["scope", "budget_bits", "comparator", "seed"]),
        "primary_qwen_mmlu_summary.csv": summary[primary_summary],
        "all_cells_policy_gains_per_seed.csv": enriched,
        "all_cells_budget_gains.csv": summary,
        "calibration_robustness_summary.csv": calibration_robustness(summary),
        "peer_switches_per_seed.csv": switches,
        "peer_switch_counts.csv": switch_counts,
        "all_q2_peer_pair_scores_per_seed.csv": pair_scores,
        "all_q2_peer_pair_scores_summary.csv": pair_summary,
        "q3_nnls_vs_krr_per_seed.csv": q3,
        "q3_nnls_vs_krr_summary.csv": direct_summary(q3),
        "q2_selected_nnls_vs_krr_per_seed.csv": q2,
        "q2_selected_nnls_vs_krr_summary.csv": direct_summary(q2),
        "gemma_hellaswag_compact_per_seed.csv": compact,
        "gemma_hellaswag_compact_summary.csv": compact_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results_complete")
    parser.add_argument("--campaign-status", type=Path, default=HERE / "campaign_status.json")
    parser.add_argument("--output", type=Path, default=HERE / "results_complete/summary")
    args = parser.parse_args()
    try:
        directories, provenance = completed_inputs(args.results.resolve(), args.campaign_status.resolve())
    except RuntimeError as error:
        raise SystemExit(str(error))
    scores, gains, direct = load_results(directories)
    tables = build_tables(scores, gains, direct)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False, float_format="%.12g")
    manifest = {
        **provenance, "summary_script_sha256": sha(Path(__file__)), "expected_cells": 45,
        "seeds": SEEDS, "tasks": TASKS, "targets": TARGETS, "budgets": BUDGETS,
        "gain_sign": "Positive gain_tv means fixed-peer TV minus budget-aware TV is positive.",
        "calibration_sign": "Positive nnls_minus_krr_tv means KRR has lower held-out TV than NNLS.",
        "uncertainty": UNCERTAINTY,
        "no_result_selection": "All nine cells, four fixed budgets, two scopes, two fixed-peer comparators and five seeds are included.",
        "positive_sign_tolerance": TOLERANCE,
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "output_sha256": {name: sha(args.output / name) for name in tables},
    }
    (args.output / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete_cells": 45, "output": str(args.output.resolve()),
                      "tables": manifest["table_rows"], "uncertainty": UNCERTAINTY}, indent=2))


if __name__ == "__main__":
    main()
