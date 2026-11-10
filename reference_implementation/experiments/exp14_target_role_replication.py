# experiments/exp14_target_role_replication.py
#
# Exp 14: Target-role replication across two peer rosters.
#
# Three models that were not members of the primary target population are
# evaluated as secondary targets. Two disjoint peer rosters reuse already
# registered response models. The experiment repeats native adapter
# sensitivity, ordered target-state swaps, and matched-budget deployment
# without fitting or selecting on test responses.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp13_peer_roster_replication import (
    RESULT_COLUMNS,
    evaluate_roster,
)
from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.audit import load_audit_cells
from kernel_isqed.contracts import (
    CROSS_ROLE_PEERS,
    CROSS_ROLE_ROSTER_ID,
    MATCHED_BUDGETS,
    PEERS,
    REFERENCE_ROSTER_ID,
    ROLE_ROTATION_TARGETS,
    ROSTER_REPLICATION_ADAPTERS,
    TASKS,
)
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table


ROSTERS = (
    (REFERENCE_ROSTER_ID, PEERS),
    (CROSS_ROLE_ROSTER_ID, CROSS_ROLE_PEERS),
)
NATIVE_FAMILY = "role_rotation_native"
FRONTIER_FAMILY = "role_rotation_frontier"
KEY_FAMILY = "EXP14_ROLE_ROTATION"


def expected_run_keys(audit_grid) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    target_ids = [target.model_id for target in ROLE_ROTATION_TARGETS]
    for roster_id, _ in ROSTERS:
        for cell_id, target_id, task_id in audit_grid:
            for adapter in ROSTER_REPLICATION_ADAPTERS:
                rows.append(
                    {
                        "key_family": KEY_FAMILY,
                        "analysis": "native",
                        "roster_id": roster_id,
                        "audit_cell": cell_id,
                        "target_id": target_id,
                        "task_id": task_id,
                        "adapter": adapter,
                        "q": 3,
                    }
                )
                for budget in MATCHED_BUDGETS:
                    rows.append(
                        {
                            "key_family": KEY_FAMILY,
                            "analysis": "frontier",
                            "roster_id": roster_id,
                            "audit_cell": cell_id,
                            "target_id": target_id,
                            "task_id": task_id,
                            "adapter": adapter,
                            "budget_cap": budget,
                            "q": 3,
                        }
                    )
        for task in TASKS:
            for adapter in ROSTER_REPLICATION_ADAPTERS:
                for recipient in target_ids:
                    for donor in target_ids:
                        if donor == recipient:
                            continue
                        rows.append(
                            {
                                "key_family": KEY_FAMILY,
                                "analysis": "state_swap",
                                "roster_id": roster_id,
                                "task_id": task.task_id,
                                "adapter": adapter,
                                "donor": donor,
                                "recipient": recipient,
                                "q": 3,
                            }
                        )
    return rows


def run_target_role_replication(
    responses: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frozen = FrozenStateStore()
    rows: List[Dict[str, object]] = []
    ledger: List[Dict[str, object]] = []
    for roster_id, peer_contracts in ROSTERS:
        cells = load_audit_cells(
            responses,
            peer_contracts=peer_contracts,
            target_contracts=ROLE_ROTATION_TARGETS,
        )
        roster_rows, roster_ledger = evaluate_roster(
            roster_id,
            cells,
            NATIVE_FAMILY,
            FRONTIER_FAMILY,
            frozen,
        )
        rows.extend(roster_rows)
        ledger.extend(roster_ledger)
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def validate_results(results: pd.DataFrame) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 14 output is missing columns: %s" % sorted(missing))
    counts = results.groupby(["roster_id", "analysis"]).size().to_dict()
    expected = {
        (roster_id, analysis): count
        for roster_id, _ in ROSTERS
        for analysis, count in (("native", 27), ("frontier", 81), ("state_swap", 54))
    }
    if counts != expected:
        raise RuntimeError("Exp 14 did not cover the registered role-rotation grid")
    if set(results["target_id"].dropna().astype(str)) != {
        target.model_id for target in ROLE_ROTATION_TARGETS
    }:
        raise RuntimeError("Exp 14 changed the registered secondary targets")
    if set(results["q"].astype(int)) != {3}:
        raise RuntimeError("Exp 14 changed the fixed peer count")
    completed = results[results["status"] == "completed"]
    losses = completed[
        completed["analysis"].isin(("native", "frontier"))
    ]["iid_test_tv"]
    if not np.isfinite(losses.to_numpy(dtype=float)).all():
        raise RuntimeError("Exp 14 contains non-finite completed losses")
    swaps = completed[completed["analysis"] == "state_swap"]
    if not np.isfinite(
        swaps[["delta_swap", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Exp 14 contains non-finite swap intervals")


def main() -> None:
    args = common_parser(
        "Exp 14: target-role replication across two peer rosters"
    ).parse_args()
    results, ledger = run_target_role_replication(args.responses)
    results = tag_run_mode(results)
    ledger = tag_run_mode(ledger)
    validate_results(results)
    output = save_experiment(results, "exp14_target_role_replication.csv")
    write_table(
        ledger,
        RESULTS_DIR / "tables/item_losses/exp14_target_role_item_losses.parquet",
    )
    print(results.groupby(["roster_id", "analysis"]).size().to_string())
    print("saved:", output)


if __name__ == "__main__":
    main()
