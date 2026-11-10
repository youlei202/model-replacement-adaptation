# experiments/exp4_planted_bit_llm_geometry.py
#
# Exp 4: Planted bits in frozen LLM response geometry.
#
# Every planted target, selected configuration, and serialized state is created
# before the one-time test unlock. This file only audits the frozen controlled
# worlds and publishes their recovery ledger. The controlled law calibrates a
# shared semantic-TV strength per cell and uses a capacity-first 256-Kibit
# primary state with 512 fresh worlds.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.io import (
    CANONICAL_DIR,
    RESULTS_DIR,
    read_json,
    save_experiment,
    sha256_file,
    write_table,
)
from kernel_isqed.planted import (
    CONFIG_SELECTION_RULE,
    DESCRIPTIVE_WORLD_COUNT,
    INFERENCE_RULE,
    PLANTED_SCOPE_ID,
    PLANTED_ADAPTERS,
    PLANTED_BUDGETS,
    PLANTED_LAW,
    PLANTED_TV_STRENGTHS,
    PRIMARY_BUDGET,
    PRIMARY_TV_STRENGTH,
    PRIMARY_WORLD_COUNT,
    RECOVERY_RULE,
    SIMULTANEOUS_ALPHA,
    commitment_scope,
    world_count,
)
from kernel_isqed.frozen import require_frozen_evaluation, source_tree_hash


FROZEN_RESULTS = CANONICAL_DIR / "frozen_planted_results.parquet"
FROZEN_ORACLE = CANONICAL_DIR.parent / "oracle_truth/frozen_planted_openings.parquet"
FROZEN_MANIFEST = CANONICAL_DIR / "frozen_planted_manifest.json"
RESULT_COLUMNS = (
    "row_kind",
    "world_law_scope",
    "audit_cell",
    "target_id",
    "task_id",
    "adapter",
    "tv_strength",
    "delta",
    "budget_cap",
    "draw_id",
    "commitment_id",
    "bit_error",
    "raw_y0_bit_error",
    "bit_error_upper_bound",
    "plugin_information_per_bit",
    "information_lower_bound_per_bit",
    "natural_perturbation_tv",
    "calibration_tv",
    "complete_coded_bits",
    "planted_bit_count",
    "bits_per_planted_bit",
    "world_count",
    "registered_points",
    "simultaneous_alpha",
    "inference_rule",
    "configuration_rule",
    "recovery_rule",
    "planted_law",
    "geometry_digest",
    "inference_tier",
    "codec",
    "rank",
    "lambda_index",
    "bitstream_digest",
    "state_path",
    "accounting_status",
    "minimum_success",
    "strong_point",
    "primary_family_pass_count",
    "primary_family_success",
    "status",
)


def expected_run_keys(audit_grid):
    rows = []
    for cell_id, target_id, task_id in audit_grid:
        for tv_strength in PLANTED_TV_STRENGTHS:
            for adapter in PLANTED_ADAPTERS:
                for budget in PLANTED_BUDGETS:
                    for draw_id in range(world_count(tv_strength, budget)):
                        rows.append(
                            {
                                "key_family": "EXP4_PLANTED",
                                "world_law_scope": PLANTED_SCOPE_ID,
                                "audit_cell": cell_id,
                                "target_id": target_id,
                                "task_id": task_id,
                                "tv_strength": tv_strength,
                                "draw_id": draw_id,
                                "adapter": adapter,
                                "budget_cap": budget,
                                "commitment_scope": commitment_scope(
                                    cell_id,
                                    adapter,
                                    tv_strength,
                                    budget,
                                    draw_id,
                                ),
                            }
                        )
    return rows


def load_frozen_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    if (
        not freeze_path.is_file()
        or not FROZEN_RESULTS.is_file()
        or not FROZEN_ORACLE.is_file()
        or not FROZEN_MANIFEST.is_file()
    ):
        raise RuntimeError("planted worlds were not frozen before the test unlock")
    freeze = read_json(freeze_path)
    if source_tree_hash() != freeze.get("source_tree_hash"):
        raise RuntimeError("source code changed after the planted worlds were frozen")
    require_frozen_evaluation()
    if sha256_file(FROZEN_RESULTS) != freeze.get("frozen_planted_results_sha256"):
        raise RuntimeError("frozen planted result ledger changed after test unlock")
    if sha256_file(FROZEN_ORACLE) != freeze.get("frozen_planted_oracle_sha256"):
        raise RuntimeError("frozen planted oracle changed after test unlock")
    if sha256_file(FROZEN_MANIFEST) != freeze.get(
        "frozen_planted_manifest_sha256"
    ):
        raise RuntimeError("frozen planted manifest changed after test unlock")
    return pd.read_parquet(FROZEN_RESULTS), pd.read_parquet(FROZEN_ORACLE)


def validate_results(
    results: pd.DataFrame, oracle: pd.DataFrame, cells: Sequence[AuditCell]
) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 4 output is missing columns: %s" % sorted(missing))
    if len(cells) != 9:
        raise RuntimeError("Exp 4 requires the nine frozen target/task cells")
    summaries = results[results["row_kind"] == "summary"]
    expected_summaries = (
        len(cells)
        * len(PLANTED_ADAPTERS)
        * len(PLANTED_TV_STRENGTHS)
        * len(PLANTED_BUDGETS)
    )
    if len(summaries) != expected_summaries:
        raise RuntimeError("Exp 4 summary grid is incomplete")
    summary_keys = ["audit_cell", "adapter", "tv_strength", "budget_cap"]
    if summaries.duplicated(summary_keys).any():
        raise RuntimeError("Exp 4 contains duplicate summary keys")

    worlds = results[results["row_kind"] == "world"]
    expected_worlds = sum(
        world_count(float(row.tv_strength), int(row.budget_cap))
        for row in summaries[summaries["status"] == "completed"].itertuples(index=False)
    )
    if len(worlds) != expected_worlds or len(oracle) != expected_worlds:
        raise RuntimeError(
            "Exp 4 controlled-world openings do not match the frozen design"
        )
    world_keys = summary_keys + ["draw_id"]
    if worlds.duplicated(world_keys).any() or oracle.duplicated(world_keys).any():
        raise RuntimeError("Exp 4 contains duplicate planted-bit worlds")
    if not worlds["bit_error"].between(0.0, 1.0).all():
        raise RuntimeError("Exp 4 produced an invalid bit error")
    if not worlds["natural_perturbation_tv"].between(0.0, 1.0).all():
        raise RuntimeError("Exp 4 produced an invalid natural-geometry TV shift")
    if (worlds["complete_coded_bits"] > worlds["budget_cap"]).any():
        raise RuntimeError("Exp 4 contains a state above its registered cap")
    if set(results["world_law_scope"].astype(str)) != {PLANTED_SCOPE_ID}:
        raise RuntimeError("Exp 4 mixes controlled world-law scopes")
    if set(results["configuration_rule"].astype(str)) != {
        CONFIG_SELECTION_RULE
    }:
        raise RuntimeError("Exp 4 uses an unregistered configuration rule")
    if set(results["recovery_rule"].astype(str)) != {RECOVERY_RULE}:
        raise RuntimeError("Exp 4 uses an unregistered bit decoder")
    if set(results["inference_rule"].astype(str)) != {INFERENCE_RULE}:
        raise RuntimeError("Exp 4 uses an unregistered inference rule")
    if set(results["planted_law"].astype(str)) != {PLANTED_LAW}:
        raise RuntimeError("Exp 4 uses an unregistered planted-world law")
    if worlds["geometry_digest"].astype(str).str.len().ne(64).any():
        raise RuntimeError("Exp 4 is missing its response-geometry digest")

    primary = summaries[
        (summaries["tv_strength"] == PRIMARY_TV_STRENGTH)
        & (summaries["budget_cap"] == PRIMARY_BUDGET)
    ]
    descriptive = summaries.drop(primary.index)
    if len(primary) != len(cells) * len(PLANTED_ADAPTERS):
        raise RuntimeError("Exp 4 primary simultaneous family is incomplete")
    if primary["information_lower_bound_per_bit"].isna().any():
        raise RuntimeError("Exp 4 primary points are missing simultaneous lower bounds")
    if set(primary["world_count"].astype(int)) != {PRIMARY_WORLD_COUNT}:
        raise RuntimeError("Exp 4 primary points use the wrong world count")
    if set(primary["registered_points"].astype(int)) != {27}:
        raise RuntimeError("Exp 4 primary family size is incorrect")
    if set(primary["simultaneous_alpha"].astype(float)) != {
        SIMULTANEOUS_ALPHA
    }:
        raise RuntimeError("Exp 4 primary family uses the wrong alpha")
    if not primary["bit_error_upper_bound"].between(0.0, 1.0).all():
        raise RuntimeError("Exp 4 primary bit-error bounds are invalid")
    observed_passes = int(
        (
            (primary["bit_error_upper_bound"] < 0.5)
            & (primary["information_lower_bound_per_bit"] > 0.0)
        ).sum()
    )
    if set(primary["primary_family_pass_count"].astype(int)) != {
        observed_passes
    }:
        raise RuntimeError("Exp 4 family-level success count is inconsistent")
    if set(primary["primary_family_success"].astype(bool)) != {
        observed_passes >= 14
    }:
        raise RuntimeError("Exp 4 family-level success verdict is inconsistent")
    exceeds_code = primary["information_lower_bound_per_bit"] > (
        primary["bits_per_planted_bit"].clip(upper=1.0) + 1e-12
    )
    if exceeds_code.any():
        bad = primary[
            exceeds_code
            & (
                primary["accounting_status"]
                != "accounting_interval_diagnostic"
            )
        ]
        if not bad.empty:
            raise RuntimeError("Exp 4 information bound exceeds its code bound")
    if descriptive["information_lower_bound_per_bit"].notna().any():
        raise RuntimeError(
            "Exp 4 descriptive points must not be presented as simultaneous lower bounds"
        )
    if set(primary["inference_tier"].astype(str)) != {"simultaneous_primary"}:
        raise RuntimeError("Exp 4 primary inference tier is mislabeled")
    if set(descriptive["inference_tier"].astype(str)) != {"descriptive"}:
        raise RuntimeError("Exp 4 descriptive inference tier is mislabeled")


def main() -> None:
    args = common_parser("Exp 4: frozen planted-bit LLM geometry").parse_args()
    cells = get_cells(args.responses)
    results, oracle = load_frozen_geometry()
    validate_results(results, oracle, cells)
    output = save_experiment(tag_run_mode(results), "exp4_planted_bit_llm_geometry.csv")
    oracle_path = write_table(
        oracle, RESULTS_DIR / "oracle_truth/exp4_planted_bit_openings.parquet"
    )
    summary = results[results["row_kind"] == "summary"]
    print(
        summary.groupby(["adapter", "inference_tier"], as_index=False)["bit_error"]
        .median()
        .to_string(index=False)
    )
    print(
        "Primary worlds:",
        PRIMARY_WORLD_COUNT,
        "descriptive worlds:",
        DESCRIPTIVE_WORLD_COUNT,
    )
    print("Saved:", output)
    print("Private oracle:", oracle_path)


if __name__ == "__main__":
    main()
