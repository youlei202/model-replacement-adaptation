# experiments/exp9_error_state_query_frontier.py
#
# Exp 9: Operational error-state-query frontier.
#
# This is the main cost-matched audit. For every fixed target/task cell we sweep
# five trained adapters, q in {1,2,3}, and fourteen complete retained-state bit
# caps. Peer subsets, lambda, rank, codec, and quantization are selected using
# validation only before the test unlock. Exp 9 loads one registered state per
# frontier point and evaluates IID test without refitting or reselection.
#
# Zero-state reference rows include all three fixed peers, all three fixed pair
# means, the validation-selected pair mean, and the three-peer mean. Curves are
# compared only at equal q; no cross-cell Pareto envelope is constructed.

from __future__ import annotations

import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import (
    AdapterState,
    mean_tv,
    peer_subsets,
    predict_coordinates,
)
from kernel_isqed.audit import AuditCell, get_cells, split_tv
from kernel_isqed.contracts import ADAPTERS, BIT_CAPS, MATCHED_BUDGETS
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table


def baseline_row(
    cell: AuditCell,
    adapter: str,
    q: int,
    subset: Tuple[int, ...],
    validation_tv: float,
    iid_test_tv: float,
    complete_coded_bits: int = 0,
) -> Dict[str, object]:
    return {
        "audit_cell": cell.cell_id,
        "target_id": cell.target_id,
        "task_id": cell.task_id,
        "adapter": adapter,
        "q": q,
        "peer_indices": "|".join(str(value) for value in subset),
        "budget_cap": complete_coded_bits,
        "complete_coded_bits": complete_coded_bits,
        "validation_tv": validation_tv,
        "iid_test_tv": iid_test_tv,
        "codec": "selector_id_2bit" if complete_coded_bits else "public_zero_state",
        "rank": 0,
        "status": "completed",
    }


def expected_run_keys(audit_grid):
    rows = []
    for cell_id, target_id, task_id in audit_grid:
        for adapter in ADAPTERS:
            for q in (2, 3) if adapter == "convex" else (1, 2, 3):
                for budget in BIT_CAPS:
                    rows.append(
                        {
                            "key_family": "EXP9_FRONTIER",
                            "audit_cell": cell_id,
                            "target_id": target_id,
                            "task_id": task_id,
                            "adapter": adapter,
                            "q": q,
                            "budget_cap": budget,
                            "condition": "trained",
                        }
                    )
        for peer_index in range(3):
            rows.append(
                {
                    "key_family": "EXP9_FRONTIER",
                    "audit_cell": cell_id,
                    "target_id": target_id,
                    "task_id": task_id,
                    "adapter": "fixed_peer",
                    "q": 1,
                    "peer_indices": str(peer_index),
                    "budget_cap": 0,
                    "condition": "baseline",
                }
            )
        for subset in ((0, 1), (0, 2), (1, 2)):
            rows.append(
                {
                    "key_family": "EXP9_FRONTIER",
                    "audit_cell": cell_id,
                    "target_id": target_id,
                    "task_id": task_id,
                    "adapter": "fixed_pair_mean",
                    "q": 2,
                    "peer_indices": "%d|%d" % subset,
                    "budget_cap": 0,
                    "condition": "baseline",
                }
            )
        for adapter, q, bits in (
            ("val_select_peer", 1, 2),
            ("val_select_pair_mean", 2, 2),
            ("mean_all", 3, 0),
        ):
            rows.append(
                {
                    "key_family": "EXP9_FRONTIER",
                    "audit_cell": cell_id,
                    "target_id": target_id,
                    "task_id": task_id,
                    "adapter": adapter,
                    "q": q,
                    "budget_cap": bits,
                    "condition": "baseline",
                }
            )
    return rows


# ---------------------------------------------------------------------
# 1. Registered zero-state references
# ---------------------------------------------------------------------
def zero_state_references(cell: AuditCell) -> Sequence[Dict[str, object]]:
    rows = []
    pair_candidates = []
    peer_candidates = []
    for q in (1, 2, 3):
        for subset in peer_subsets(q):
            context = cell.context(subset)
            state = AdapterState("peer" if q == 1 else "mean_all", subset)
            validation_prediction = predict_coordinates(
                state, context, cell.peer_coordinates["validation"],
            )
            test_prediction = predict_coordinates(
                state, context, cell.peer_coordinates["iid_test"],
            )
            validation_error = mean_tv(
                cell.target_coordinates["validation"], validation_prediction
            )
            test_error = mean_tv(cell.target_coordinates["iid_test"], test_prediction)
            name = (
                "fixed_peer" if q == 1 else "fixed_pair_mean" if q == 2 else "mean_all"
            )
            rows.append(
                baseline_row(cell, name, q, subset, validation_error, test_error)
            )
            if q == 2:
                pair_candidates.append((validation_error, test_error, subset))
            if q == 1:
                peer_candidates.append((validation_error, test_error, subset))

    validation_error, test_error, subset = min(
        peer_candidates, key=lambda value: (value[0], value[2])
    )
    rows.append(
        baseline_row(
            cell,
            "val_select_peer",
            1,
            subset,
            validation_error,
            test_error,
            complete_coded_bits=2,
        )
    )

    validation_error, test_error, subset = min(
        pair_candidates, key=lambda value: (value[0], value[2])
    )
    rows.append(
        baseline_row(
            cell,
            "val_select_pair_mean",
            2,
            subset,
            validation_error,
            test_error,
            complete_coded_bits=2,
        )
    )
    return rows


# ---------------------------------------------------------------------
# 2. Cost-matched trained frontier
# ---------------------------------------------------------------------
def run_error_state_query_frontier(
    cells: Sequence[AuditCell], fast: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if fast or len(cells) <= 1:
        groups = [_frontier_for_cell(cell, fast) for cell in cells]
    else:
        workers = min(
            len(cells),
            int(os.environ.get("KERNEL_ISQED_EXPERIMENT_WORKERS", os.cpu_count() or 1)),
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            groups = list(
                executor.map(
                    _frontier_for_cell_worker,
                    [(cell, fast) for cell in cells],
                )
            )
    rows = []
    ledger = []
    for group_rows, group_ledger in groups:
        rows.extend(group_rows)
        ledger.extend(group_ledger)
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def _frontier_for_cell(cell: AuditCell, fast: bool):
    rows = []
    ledger = []
    caps = MATCHED_BUDGETS if fast else BIT_CAPS
    frozen = FrozenStateStore()

    rows.extend(zero_state_references(cell))
    for q in (1, 2, 3):
        for adapter in ADAPTERS:
            if adapter == "convex" and q == 1:
                continue
            for cap in caps:
                selected = frozen.manifest[
                    (frozen.manifest["audit_cell"].astype(str) == cell.cell_id)
                    & (
                        frozen.manifest["state_family"].astype(str)
                        == "frontier_selected"
                    )
                    & (frozen.manifest["adapter"].astype(str) == adapter)
                    & (frozen.manifest["q"].astype(str) == str(q))
                    & (frozen.manifest["budget_cap"].astype(str) == str(cap))
                ]
                if len(selected) != 1:
                    raise RuntimeError("registered frontier point is not unique")
                peer_indices = tuple(
                    int(value)
                    for value in str(selected.iloc[0]["peer_indices"]).split("|")
                    if value
                )
                context = cell.context(
                    peer_indices if peer_indices else tuple(range(q))
                )
                state, contract = frozen.load(
                    cell,
                    context,
                    "frontier_selected",
                    adapter,
                    q=q,
                    budget_cap=cap,
                )
                if state is None:
                    rows.append(
                        {
                            "audit_cell": cell.cell_id,
                            "target_id": cell.target_id,
                            "task_id": cell.task_id,
                            "adapter": adapter,
                            "q": q,
                            "peer_indices": "",
                            "budget_cap": cap,
                            "complete_coded_bits": np.nan,
                            "validation_tv": np.nan,
                            "iid_test_tv": np.nan,
                            "codec": "",
                            "rank": np.nan,
                            "status": "no_feasible_configuration",
                        }
                    )
                    continue

                prediction = predict_coordinates(
                    state, context, cell.peer_coordinates["iid_test"],
                )
                losses = split_tv(cell, "iid_test", prediction)
                rows.append(
                    {
                        "audit_cell": cell.cell_id,
                        "target_id": cell.target_id,
                        "task_id": cell.task_id,
                        "adapter": adapter,
                        "q": q,
                        "peer_indices": "|".join(
                            str(value) for value in state.peer_indices
                        ),
                        "budget_cap": cap,
                        "complete_coded_bits": int(contract["complete_coded_bits"]),
                        "validation_tv": float(contract["validation_tv"]),
                        "iid_test_tv": mean_tv(
                            cell.target_coordinates["iid_test"], prediction
                        ),
                        "codec": str(contract["codec"]),
                        "rank": int(contract["rank"]),
                        "status": "completed",
                    }
                )
                ledger.extend(
                    [
                        {
                            "audit_cell": cell.cell_id,
                            "target_id": cell.target_id,
                            "task_id": cell.task_id,
                            "item_id": str(cell.item_ids["iid_test"][index]),
                            "item_cluster": str(
                                cell.item_clusters["iid_test"][index]
                            ),
                            "domain": str(cell.domains["iid_test"][index]),
                            "adapter": adapter,
                            "q": q,
                            "budget_cap": cap,
                            "complete_coded_bits": int(
                                contract["complete_coded_bits"]
                            ),
                            "tv_loss": float(loss),
                        }
                        for index, loss in enumerate(losses)
                    ]
                )
    return rows, ledger


def _frontier_for_cell_worker(payload):
    cell, fast = payload
    return _frontier_for_cell(cell, fast)


# ---------------------------------------------------------------------
# 3. Frontier checks
# ---------------------------------------------------------------------
def validate_results(
    results: pd.DataFrame, cells: Sequence[AuditCell], fast: bool
) -> None:
    required = {
        "audit_cell",
        "target_id",
        "task_id",
        "adapter",
        "q",
        "peer_indices",
        "budget_cap",
        "complete_coded_bits",
        "validation_tv",
        "iid_test_tv",
        "codec",
        "rank",
        "status",
    }
    missing = required - set(results.columns)
    if missing:
        raise RuntimeError("Exp 9 output is missing columns: %s" % sorted(missing))
    if len(cells) != 9:
        raise RuntimeError("Exp 9 requires the nine frozen target/task cells")

    trained = results[results["adapter"].isin(ADAPTERS)]
    caps = set(MATCHED_BUDGETS if fast else BIT_CAPS)
    if set(trained["budget_cap"].astype(int)) != caps:
        raise RuntimeError("Exp 9 retained-state cap universe is incomplete")
    if set(trained["q"].astype(int)) != {1, 2, 3}:
        raise RuntimeError("Exp 9 peer-query universe is incomplete")
    keys = ["audit_cell", "adapter", "q", "budget_cap"]
    if trained.duplicated(keys).any():
        raise RuntimeError("Exp 9 contains duplicate frontier keys")

    completed = trained[trained["status"] == "completed"]
    if (completed["complete_coded_bits"] > completed["budget_cap"]).any():
        raise RuntimeError("Exp 9 selected a state above its registered cap")
    numeric = completed[
        ["complete_coded_bits", "validation_tv", "iid_test_tv", "rank"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Exp 9 contains non-finite completed frontier rows")
    allowed = {"completed", "no_feasible_configuration"}
    if not set(trained["status"].astype(str)).issubset(allowed):
        raise RuntimeError("Exp 9 contains an unknown frontier status")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = common_parser(
        "Exp 9: operational error-state-query frontier", include_fast=True
    )
    args = parser.parse_args()
    cells = get_cells(args.responses)
    results, ledger = run_error_state_query_frontier(cells, args.fast)
    validate_results(results, cells, args.fast)
    results = tag_run_mode(results, args.fast)
    output = save_experiment(results, "exp9_error_state_query_frontier.csv")
    write_table(
        ledger, RESULTS_DIR / "tables/item_losses/exp9_frontier_item_losses.parquet"
    )

    completed = results[results["status"] == "completed"]
    print(
        completed.groupby(["adapter", "q"], as_index=False)["iid_test_tv"]
        .median()
        .to_string(index=False)
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
