# experiments/exp10_peer_source_controls.py
#
# Exp 10: Peer-source and alternative-feature controls.
#
# The main q=3 conditions compare actual peer responses with row-shuffled and
# channel-permuted peer tables at a fixed 16-Kibit state cap. Two flexible
# adapters, RFF and KRR, are evaluated. A second q=0 family replaces peer
# behavior by public random addresses, input-only TF-IDF features, or a constant
# feature.
#
# Vectorizers and adapter settings are determined on fit/validation before test
# unlock. This file reconstructs the public features, verifies the frozen state,
# and evaluates without fitting a target-derived control after the unlock.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import AdapterState, predict_coordinates
from kernel_isqed.audit import AuditCell, get_cells, split_tv
from kernel_isqed.contracts import GLOBAL_SEED
from kernel_isqed.controls import transform_tfidf_state
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table


# ---------------------------------------------------------------------
# 1. Registered controls
# ---------------------------------------------------------------------
STATE_CAP = 2 ** 14
CONTROL_ADAPTERS = ("rff", "krr")
PEER_CONDITIONS = ("actual", "row_shuffled", "channel_permuted")
ALTERNATIVE_CONDITIONS = ("random_address", "input_tfidf", "constant")
SPLITS = ("fit", "validation", "iid_test", "ood_test")
ALL_CONDITIONS = PEER_CONDITIONS + ALTERNATIVE_CONDITIONS
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "adapter",
    "condition",
    "q",
    "budget_cap",
    "complete_coded_bits",
    "validation_tv",
    "iid_test_tv",
    "codec",
    "status",
)


def expected_run_keys(audit_grid):
    rows = []
    for cell_id, target_id, task_id in audit_grid:
        for adapter in CONTROL_ADAPTERS:
            for condition in PEER_CONDITIONS:
                rows.append(
                    {
                        "key_family": "EXP10_CONTROLS",
                        "audit_cell": cell_id,
                        "target_id": target_id,
                        "task_id": task_id,
                        "adapter": adapter,
                        "condition": condition,
                        "q": 3,
                        "budget_cap": STATE_CAP,
                    }
                )
            for condition in ALTERNATIVE_CONDITIONS:
                rows.append(
                    {
                        "key_family": "EXP10_CONTROLS",
                        "audit_cell": cell_id,
                        "target_id": target_id,
                        "task_id": task_id,
                        "adapter": adapter,
                        "condition": condition,
                        "q": 0,
                        "budget_cap": STATE_CAP,
                    }
                )
    return rows


def shuffled_peer_cell(cell: AuditCell, condition: str) -> AuditCell:
    peers = {}
    for split_index, split in enumerate(cell.item_ids):
        values = cell.peer_coordinates[split].copy()
        rng = np.random.RandomState(GLOBAL_SEED + split_index)
        if condition == "row_shuffled":
            values = values[rng.permutation(len(values))]
        elif condition == "channel_permuted":
            for peer_index in range(values.shape[1]):
                values[:, peer_index, :] = values[
                    :, peer_index, rng.permutation(values.shape[2])
                ]
        else:
            raise ValueError(condition)
        peers[split] = values
    return AuditCell(
        cell.target_id,
        cell.task_id,
        cell.item_ids,
        peers,
        cell.target_coordinates,
        cell.input_text,
        cell.item_clusters,
        cell.domains,
        cell.gold,
    )


def alternative_feature_cell(
    cell: AuditCell, condition: str, feature_state: Optional[Dict[str, object]] = None,
) -> AuditCell:
    peers = {}
    if condition == "random_address":
        rng = np.random.RandomState(GLOBAL_SEED)
        total = sum(len(cell.item_ids[split]) for split in cell.item_ids)
        public_features = rng.normal(0.0, 1.0, size=(total, 1, 256))
        offset = 0
        for split in cell.item_ids:
            size = len(cell.item_ids[split])
            peers[split] = public_features[offset : offset + size]
            offset += size
    elif condition == "input_tfidf":
        if feature_state is None:
            raise RuntimeError("input_tfidf requires its pre-test frozen feature state")
        text = cell.input_text if cell.input_text is not None else cell.item_ids
        for split in cell.item_ids:
            peers[split] = transform_tfidf_state(feature_state, text[split])[:, None, :]
    elif condition == "constant":
        for split in cell.item_ids:
            peers[split] = np.ones((len(cell.item_ids[split]), 1, 1), dtype=np.float64)
    else:
        raise ValueError(condition)
    return AuditCell(
        cell.target_id,
        cell.task_id,
        cell.item_ids,
        peers,
        cell.target_coordinates,
        cell.input_text,
        cell.item_clusters,
        cell.domains,
        cell.gold,
    )


def control_row(
    natural: AuditCell,
    controlled: AuditCell,
    state: Optional[AdapterState],
    contract: pd.Series,
    adapter: str,
    condition: str,
    q: int,
) -> tuple[Dict[str, object], np.ndarray]:
    row = {
        "audit_cell": natural.cell_id,
        "target_id": natural.target_id,
        "task_id": natural.task_id,
        "adapter": adapter,
        "condition": condition,
        "q": q,
        "budget_cap": STATE_CAP,
    }
    if state is None:
        return (
            {
                **row,
                "complete_coded_bits": np.nan,
                "validation_tv": np.nan,
                "iid_test_tv": np.nan,
                "codec": "",
                "status": "no_feasible_configuration",
            },
            np.asarray([], dtype=np.float64),
        )
    peer_indices = (0, 1, 2) if q == 3 else (0,)
    context = controlled.context(peer_indices)
    prediction = predict_coordinates(
        state, context, controlled.peer_coordinates["iid_test"]
    )
    losses = split_tv(natural, "iid_test", prediction)
    return (
        {
            **row,
            "complete_coded_bits": int(contract["complete_coded_bits"]),
            "validation_tv": float(contract["validation_tv"]),
            "iid_test_tv": float(losses.mean()),
            "codec": str(contract["codec"]),
            "status": "completed",
        },
        losses,
    )


# ---------------------------------------------------------------------
# 2. Control experiment
# ---------------------------------------------------------------------
def run_peer_source_controls(
    cells: Sequence[AuditCell], fast: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ledger = []
    frozen = FrozenStateStore()
    for cell in cells:
        for adapter in CONTROL_ADAPTERS:
            for condition in PEER_CONDITIONS:
                controlled = (
                    cell
                    if condition == "actual"
                    else shuffled_peer_cell(cell, condition)
                )
                context = controlled.context((0, 1, 2))
                state, contract = frozen.load(
                    cell, context, "peer_source_control", adapter, condition=condition,
                )
                row, losses = control_row(
                    cell, controlled, state, contract, adapter, condition, q=3
                )
                rows.append(row)
                ledger.extend(_control_loss_rows(cell, adapter, condition, 3, losses))

            for condition in ALTERNATIVE_CONDITIONS:
                feature_state = (
                    frozen.load_feature_state(cell, condition)
                    if condition == "input_tfidf"
                    else None
                )
                controlled = alternative_feature_cell(cell, condition, feature_state)
                context = controlled.context((0,))
                state, contract = frozen.load(
                    cell, context, "peer_source_control", adapter, condition=condition,
                )
                row, losses = control_row(
                    cell, controlled, state, contract, adapter, condition, q=0
                )
                rows.append(row)
                ledger.extend(_control_loss_rows(cell, adapter, condition, 0, losses))
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def _control_loss_rows(
    cell: AuditCell, adapter: str, condition: str, q: int, losses: np.ndarray,
) -> Sequence[Dict[str, object]]:
    return [
        {
            "audit_cell": cell.cell_id,
            "target_id": cell.target_id,
            "task_id": cell.task_id,
            "item_id": str(cell.item_ids["iid_test"][index]),
            "item_cluster": str(cell.item_clusters["iid_test"][index]),
            "domain": str(cell.domains["iid_test"][index]),
            "adapter": adapter,
            "condition": condition,
            "q": q,
            "tv_loss": float(loss),
        }
        for index, loss in enumerate(losses)
    ]


# ---------------------------------------------------------------------
# 3. Control-universe checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell]) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 10 output is missing columns: %s" % sorted(missing))
    expected_rows = 9 * len(CONTROL_ADAPTERS) * len(ALL_CONDITIONS)
    if len(cells) != 9 or len(results) != expected_rows:
        raise RuntimeError("Exp 10 did not cover the registered 9 x 2 x 6 grid")
    if set(results["adapter"].astype(str)) != set(CONTROL_ADAPTERS):
        raise RuntimeError("Exp 10 adapter universe is incomplete")
    if set(results["condition"].astype(str)) != set(ALL_CONDITIONS):
        raise RuntimeError("Exp 10 control universe is incomplete")
    if results.duplicated(["audit_cell", "adapter", "condition"]).any():
        raise RuntimeError("Exp 10 contains duplicate control keys")

    actual_q = results[results["condition"].isin(PEER_CONDITIONS)]["q"]
    alternative_q = results[results["condition"].isin(ALTERNATIVE_CONDITIONS)]["q"]
    if set(actual_q.astype(int)) != {3} or set(alternative_q.astype(int)) != {0}:
        raise RuntimeError("Exp 10 control rows use an incorrect query count")
    if set(results["budget_cap"].astype(int)) != {STATE_CAP}:
        raise RuntimeError("Exp 10 changed the fixed retained-state cap")

    completed = results[results["status"] == "completed"]
    if (completed["complete_coded_bits"] > STATE_CAP).any():
        raise RuntimeError("Exp 10 selected a control state above the registered cap")
    numeric = completed[
        ["complete_coded_bits", "validation_tv", "iid_test_tv"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Exp 10 contains non-finite completed control rows")
    allowed = {"completed", "no_feasible_configuration"}
    if not set(results["status"].astype(str)).issubset(allowed):
        raise RuntimeError("Exp 10 contains an unknown completion status")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = common_parser(
        "Exp 10: peer-source and alternative-feature controls", include_fast=True
    )
    args = parser.parse_args()
    cells = get_cells(args.responses)
    results, ledger = run_peer_source_controls(cells, args.fast)
    validate_results(results, cells)
    results = tag_run_mode(results, args.fast)
    output = save_experiment(results, "exp10_peer_source_controls.csv")
    write_table(
        ledger, RESULTS_DIR / "tables/item_losses/exp10_control_item_losses.parquet"
    )

    completed = results[results["status"] == "completed"]
    print(
        completed.groupby(["adapter", "condition", "q"], as_index=False)["iid_test_tv"]
        .median()
        .to_string(index=False)
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
