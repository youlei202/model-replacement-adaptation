# experiments/exp13_peer_roster_replication.py
#
# Exp 13: Independent peer-roster replication.
#
# The reference and independently specified instruction-model pools share the
# same targets, tasks, prompts, splits, and evaluation items. Target-trained
# states are frozen before test access. Evaluation repeats native adapter
# sensitivity, ordered target-state swaps, and the matched-budget frontier
# without fitting or selecting on test responses.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import AdapterContext, AdapterState, mean_tv, predict_coordinates
from kernel_isqed.audit import AuditCell, load_audit_cells, paired_bootstrap_interval, split_tv
from kernel_isqed.contracts import (
    MATCHED_BUDGETS,
    PEERS,
    REFERENCE_ROSTER_ID,
    REPLICATION_PEERS,
    REPLICATION_ROSTER_ID,
    ROSTER_REPLICATION_ADAPTERS,
    TARGETS,
    TASKS,
)
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table


ROSTERS = (
    (REFERENCE_ROSTER_ID, PEERS, "native", "frontier_selected"),
    (REPLICATION_ROSTER_ID, REPLICATION_PEERS, "roster_native", "roster_frontier"),
)
RESULT_COLUMNS = (
    "analysis",
    "roster_id",
    "audit_cell",
    "target_id",
    "task_id",
    "adapter",
    "donor",
    "recipient",
    "budget_cap",
    "validation_tv",
    "iid_test_tv",
    "correct_tv",
    "controlled_tv",
    "delta_swap",
    "ci_lower",
    "ci_upper",
    "complete_coded_bits",
    "codec",
    "q",
    "status",
)


def expected_run_keys(audit_grid):
    rows = []
    target_ids = [target.model_id for target in TARGETS]
    for roster_id, _, _, _ in ROSTERS:
        for cell_id, target_id, task_id in audit_grid:
            for adapter in ROSTER_REPLICATION_ADAPTERS:
                rows.append(
                    {
                        "key_family": "EXP13_ROSTER_REPLICATION",
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
                            "key_family": "EXP13_ROSTER_REPLICATION",
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
                        if donor != recipient:
                            rows.append(
                                {
                                    "key_family": "EXP13_ROSTER_REPLICATION",
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


def _state(
    frozen: FrozenStateStore,
    cell: AuditCell,
    context: AdapterContext,
    family: str,
    adapter: str,
    roster_id: str,
    **selectors: object,
) -> Tuple[AdapterState | None, pd.Series]:
    if family not in {"native", "frontier_selected"}:
        selectors["roster_id"] = roster_id
    return frozen.load(cell, context, family, adapter, **selectors)


def _blank_row(**values: object) -> Dict[str, object]:
    row: Dict[str, object] = {column: np.nan for column in RESULT_COLUMNS}
    row.update({"donor": "", "recipient": "", "codec": "", "q": 3})
    row.update(values)
    return row


def evaluate_roster(
    roster_id: str,
    cells: Sequence[AuditCell],
    native_family: str,
    frontier_family: str,
    frozen: FrozenStateStore,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    ledger: List[Dict[str, object]] = []
    states: Dict[Tuple[str, str], Tuple[AdapterState, AdapterContext, pd.Series]] = {}
    for cell in cells:
        context = cell.context((0, 1, 2))
        for adapter in ROSTER_REPLICATION_ADAPTERS:
            state, contract = _state(
                frozen, cell, context, native_family, adapter, roster_id
            )
            if state is None:
                raise RuntimeError("native roster state cannot be infeasible")
            states[(cell.cell_id, adapter)] = (state, context, contract)
            prediction = predict_coordinates(
                state, context, cell.peer_coordinates["iid_test"]
            )
            losses = split_tv(cell, "iid_test", prediction)
            rows.append(
                _blank_row(
                    analysis="native",
                    roster_id=roster_id,
                    audit_cell=cell.cell_id,
                    target_id=cell.target_id,
                    task_id=cell.task_id,
                    adapter=adapter,
                    validation_tv=float(contract["validation_tv"]),
                    iid_test_tv=float(losses.mean()),
                    complete_coded_bits=int(contract["complete_coded_bits"]),
                    codec=str(contract["codec"]),
                    status="completed",
                )
            )
            for index, loss in enumerate(losses):
                ledger.append(
                    {
                        "analysis": "native",
                        "roster_id": roster_id,
                        "audit_cell": cell.cell_id,
                        "target_id": cell.target_id,
                        "task_id": cell.task_id,
                        "adapter": adapter,
                        "item_id": str(cell.item_ids["iid_test"][index]),
                        "item_cluster": str(cell.item_clusters["iid_test"][index]),
                        "tv_loss": float(loss),
                    }
                )

            for budget in MATCHED_BUDGETS:
                frontier_state, frontier_contract = _state(
                    frozen,
                    cell,
                    context,
                    frontier_family,
                    adapter,
                    roster_id,
                    q=3,
                    budget_cap=budget,
                )
                if frontier_state is None:
                    rows.append(
                        _blank_row(
                            analysis="frontier",
                            roster_id=roster_id,
                            audit_cell=cell.cell_id,
                            target_id=cell.target_id,
                            task_id=cell.task_id,
                            adapter=adapter,
                            budget_cap=budget,
                            status="no_feasible_configuration",
                        )
                    )
                    continue
                frontier_prediction = predict_coordinates(
                    frontier_state,
                    context,
                    cell.peer_coordinates["iid_test"],
                )
                rows.append(
                    _blank_row(
                        analysis="frontier",
                        roster_id=roster_id,
                        audit_cell=cell.cell_id,
                        target_id=cell.target_id,
                        task_id=cell.task_id,
                        adapter=adapter,
                        budget_cap=budget,
                        validation_tv=float(frontier_contract["validation_tv"]),
                        iid_test_tv=mean_tv(
                            cell.target_coordinates["iid_test"],
                            frontier_prediction,
                        ),
                        complete_coded_bits=int(
                            frontier_contract["complete_coded_bits"]
                        ),
                        codec=str(frontier_contract["codec"]),
                        status="completed",
                    )
                )

    for task in TASKS:
        task_cells = [cell for cell in cells if cell.task_id == task.task_id]
        for recipient in task_cells:
            for adapter in ROSTER_REPLICATION_ADAPTERS:
                correct, correct_context, _ = states[(recipient.cell_id, adapter)]
                correct_prediction = predict_coordinates(
                    correct,
                    correct_context,
                    recipient.peer_coordinates["iid_test"],
                )
                correct_loss = split_tv(recipient, "iid_test", correct_prediction)
                for donor in task_cells:
                    if donor.target_id == recipient.target_id:
                        continue
                    donor_state, donor_context, _ = states[(donor.cell_id, adapter)]
                    controlled_prediction = predict_coordinates(
                        donor_state,
                        donor_context,
                        recipient.peer_coordinates["iid_test"],
                    )
                    controlled_loss = split_tv(
                        recipient, "iid_test", controlled_prediction
                    )
                    estimate, lower, upper = paired_bootstrap_interval(
                        controlled_loss,
                        correct_loss,
                        clusters=recipient.item_clusters["iid_test"],
                    )
                    rows.append(
                        _blank_row(
                            analysis="state_swap",
                            roster_id=roster_id,
                            audit_cell=recipient.cell_id,
                            target_id=recipient.target_id,
                            task_id=task.task_id,
                            adapter=adapter,
                            donor=donor.target_id,
                            recipient=recipient.target_id,
                            correct_tv=float(correct_loss.mean()),
                            controlled_tv=float(controlled_loss.mean()),
                            delta_swap=estimate,
                            ci_lower=lower,
                            ci_upper=upper,
                            status="completed",
                        )
                    )
    return rows, ledger


def run_peer_roster_replication(
    responses: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frozen = FrozenStateStore()
    rows: List[Dict[str, object]] = []
    ledger: List[Dict[str, object]] = []
    for roster_id, peer_contracts, native_family, frontier_family in ROSTERS:
        cells = load_audit_cells(responses, peer_contracts=peer_contracts)
        roster_rows, roster_ledger = evaluate_roster(
            roster_id, cells, native_family, frontier_family, frozen
        )
        rows.extend(roster_rows)
        ledger.extend(roster_ledger)
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def validate_results(results: pd.DataFrame) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 13 output is missing columns: %s" % sorted(missing))
    counts = results.groupby(["roster_id", "analysis"]).size().to_dict()
    expected = {
        (roster_id, analysis): count
        for roster_id, _, _, _ in ROSTERS
        for analysis, count in (("native", 27), ("frontier", 81), ("state_swap", 54))
    }
    if counts != expected:
        raise RuntimeError("Exp 13 did not cover the registered roster grid")
    if set(results["q"].astype(int)) != {3}:
        raise RuntimeError("Exp 13 changed the fixed peer count")
    completed = results[results["status"] == "completed"]
    numeric = completed[
        completed["analysis"].isin(("native", "frontier"))
    ]["iid_test_tv"]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("Exp 13 contains non-finite completed losses")
    swaps = completed[completed["analysis"] == "state_swap"]
    if not np.isfinite(
        swaps[["delta_swap", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Exp 13 contains non-finite swap intervals")


def main() -> None:
    args = common_parser("Exp 13: independent peer-roster replication").parse_args()
    results, ledger = run_peer_roster_replication(args.responses)
    results = tag_run_mode(results)
    ledger = tag_run_mode(ledger)
    validate_results(results)
    output = save_experiment(results, "exp13_peer_roster_replication.csv")
    write_table(
        ledger,
        RESULTS_DIR / "tables/item_losses/exp13_peer_roster_item_losses.parquet",
    )
    print(results.groupby(["roster_id", "analysis"]).size().to_string())
    print("saved:", output)


if __name__ == "__main__":
    main()
