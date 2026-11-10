# experiments/exp5_llm_adapter_sensitivity.py
#
# Exp 5: Native-state adapter sensitivity on the fixed LLM audit cells.
#
# The comparison holds the three peers, semantic response contract, four data
# splits, and q=3 fixed. Mean-all, Convex, Linear, RFF, KRR, and a compact
# two-layer MLP are evaluated with native full-precision states. Every trained
# configuration was selected and serialized before the test unlock.
#
# This is intentionally an unmatched-state diagnostic: state sizes differ and
# no claim about cost-matched superiority is made from this table alone.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import AdapterState, predict_coordinates
from kernel_isqed.audit import AuditCell, get_cells, paired_bootstrap_interval, split_tv
from kernel_isqed.contracts import ADAPTERS
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table


# ---------------------------------------------------------------------
# 1. Result construction
# ---------------------------------------------------------------------
PEER_INDICES = (0, 1, 2)
EXPECTED_ADAPTERS = ("mean_all",) + ADAPTERS
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "split",
    "adapter",
    "q",
    "peer_indices",
    "lambda_index",
    "rank",
    "tv",
    "ci_lower",
    "ci_upper",
    "n_items",
    "validation_tv",
    "comparison",
    "codec",
    "complete_coded_bits",
    "status",
)


def expected_run_keys(audit_grid):
    return [
        {
            "key_family": "EXP5_NATIVE",
            "audit_cell": cell_id,
            "target_id": target_id,
            "task_id": task_id,
            "adapter": adapter,
            "q": 3,
        }
        for cell_id, target_id, task_id in audit_grid
        for adapter in EXPECTED_ADAPTERS
    ]


def result_row(cell: AuditCell, state: AdapterState, prediction: np.ndarray,) -> Tuple[Dict[str, object], np.ndarray]:
    losses = split_tv(cell, "iid_test", prediction)
    estimate, lower, upper = paired_bootstrap_interval(losses, clusters=cell.item_clusters["iid_test"],)
    return (
        {
            "audit_cell": cell.cell_id,
            "target_id": cell.target_id,
            "task_id": cell.task_id,
            "split": "iid_test",
            "adapter": state.adapter,
            "q": len(state.peer_indices),
            "peer_indices": "|".join(str(value) for value in state.peer_indices),
            "lambda_index": state.lambda_index,
            "rank": state.rank,
            "tv": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "n_items": len(losses),
        },
        losses,
    )


# ---------------------------------------------------------------------
# 2. Native-state comparison
# ---------------------------------------------------------------------
def run_adapter_sensitivity(cells: Sequence[AuditCell]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ledger = []
    frozen = FrozenStateStore()

    for cell in cells:
        context = cell.context(PEER_INDICES)
        mean_state = AdapterState("mean_all", PEER_INDICES)
        mean_prediction = predict_coordinates(mean_state, context, cell.peer_coordinates["iid_test"],)
        mean_row, mean_losses = result_row(cell, mean_state, mean_prediction)
        mean_row["validation_tv"] = float("nan")
        mean_row["comparison"] = "unmatched_native_state"
        mean_row["codec"] = "public_zero_state"
        mean_row["complete_coded_bits"] = 0
        mean_row["status"] = "completed"
        rows.append(mean_row)
        ledger.extend(_loss_rows(cell, mean_state.adapter, mean_losses, 0, "public_zero_state"))

        for adapter in ADAPTERS:
            state, frozen_row = frozen.load(cell, context, "native", adapter)
            if state is None:
                raise RuntimeError("native state cannot be infeasible")
            prediction = predict_coordinates(state, context, cell.peer_coordinates["iid_test"],)
            row, losses = result_row(cell, state, prediction)
            row["validation_tv"] = float(frozen_row["validation_tv"])
            row["comparison"] = "unmatched_native_state"
            row["codec"] = str(frozen_row["codec"])
            row["complete_coded_bits"] = int(frozen_row["complete_coded_bits"])
            row["status"] = "completed"
            rows.append(row)
            ledger.extend(
                _loss_rows(cell, adapter, losses, int(frozen_row["complete_coded_bits"]), str(frozen_row["codec"]),)
            )
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def _loss_rows(
    cell: AuditCell, adapter: str, losses: np.ndarray, complete_coded_bits: int, codec: str,
) -> Sequence[Dict[str, object]]:
    return [
        {
            "audit_cell": cell.cell_id,
            "target_id": cell.target_id,
            "task_id": cell.task_id,
            "split": "iid_test",
            "item_id": str(cell.item_ids["iid_test"][index]),
            "item_cluster": str(cell.item_clusters["iid_test"][index]),
            "domain": str(cell.domains["iid_test"][index]),
            "adapter": adapter,
            "q": 3,
            "complete_coded_bits": complete_coded_bits,
            "codec": codec,
            "tv_loss": float(loss),
        }
        for index, loss in enumerate(losses)
    ]


# ---------------------------------------------------------------------
# 3. Result checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell]) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 5 output is missing columns: %s" % sorted(missing))
    expected_rows = 9 * len(EXPECTED_ADAPTERS)
    if len(cells) != 9 or len(results) != expected_rows:
        raise RuntimeError("Exp 5 did not cover the frozen native comparison grid")
    if set(results["adapter"].astype(str)) != set(EXPECTED_ADAPTERS):
        raise RuntimeError("Exp 5 adapter universe is incomplete")
    if results.duplicated(["audit_cell", "adapter"]).any():
        raise RuntimeError("Exp 5 contains duplicate audit-cell/adapter rows")
    if set(results["q"].astype(int)) != {3} or set(results["split"].astype(str)) != {"iid_test"}:
        raise RuntimeError("Exp 5 changed the fixed query or evaluation split")
    if set(results["status"].astype(str)) != {"completed"}:
        raise RuntimeError("Exp 5 contains an incomplete adapter row")

    intervals = results[["ci_lower", "tv", "ci_upper"]].to_numpy(dtype=float)
    if not np.isfinite(intervals).all():
        raise RuntimeError("Exp 5 contains non-finite test estimates")
    if np.any(intervals[:, 0] > intervals[:, 1]) or np.any(intervals[:, 1] > intervals[:, 2]):
        raise RuntimeError("Exp 5 confidence intervals do not contain their estimates")
    trained = results[results["adapter"] != "mean_all"]
    if not np.isfinite(trained[["validation_tv", "complete_coded_bits"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Exp 5 contains invalid trained-state accounting")
    if (trained["complete_coded_bits"] <= 0).any():
        raise RuntimeError("Exp 5 trained adapters must retain a non-empty state")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 5: native-state LLM adapter sensitivity").parse_args()
    cells = get_cells(args.responses)
    results, ledger = run_adapter_sensitivity(cells)
    validate_results(results, cells)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp5_llm_adapter_sensitivity.csv")
    write_table(ledger, RESULTS_DIR / "tables/item_losses/exp5_native_item_losses.parquet")

    print(results.groupby("adapter", as_index=False)["tv"].median().to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
