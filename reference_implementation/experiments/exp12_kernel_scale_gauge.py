# experiments/exp12_kernel_scale_gauge.py
#
# Exp 12: Kernel-scale gauge check.
#
# The RBF kernel is rescaled by c in {1e-6, 1e-3, 1, 1e3, 1e6}. KRR alpha and
# lambda are transformed by the corresponding inverse/direct factors so that
# the deployed predictor should remain unchanged. This is a numerical and code-
# length accounting check, not a new model comparison.
#
# A scale-dependent prediction, test TV, or retained-state bit count indicates
# that a nominal kernel gauge has leaked into the implementation contract.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import mean_tv, rbf_kernel
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.contracts import LAMBDA_GRID
from kernel_isqed.io import save_experiment
from kernel_isqed.frozen import FrozenStateStore


KERNEL_SCALES = (1e-6, 1e-3, 1.0, 1e3, 1e6)
EXPECTED_AUDIT_CELLS = 9
PREDICTION_TOLERANCE = 1e-8
TV_TOLERANCE = 1e-8
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "kernel_scale",
    "lambda_prime",
    "inverse_lambda_prime",
    "max_prediction_difference",
    "test_tv_difference",
    "complete_coded_bits",
    "q",
    "status",
)


def expected_run_keys(audit_grid):
    return [
        {
            "key_family": "EXP12_GAUGE",
            "audit_cell": cell_id,
            "target_id": target_id,
            "task_id": task_id,
            "kernel_scale": scale,
            "q": 3,
        }
        for cell_id, target_id, task_id in audit_grid
        for scale in KERNEL_SCALES
    ]


# ---------------------------------------------------------------------
# 1. Gauge-invariance experiment
# ---------------------------------------------------------------------
def run_kernel_scale_gauge(cells: Sequence[AuditCell]) -> pd.DataFrame:
    rows = []
    frozen = FrozenStateStore()
    for cell in cells:
        context = cell.context((0, 1, 2))
        state, row_contract = frozen.load(cell, context, "native", "krr")
        if state is None:
            raise RuntimeError("native KRR state cannot be infeasible")
        values = cell.peer_coordinates["iid_test"].reshape(len(cell.peer_coordinates["iid_test"]), -1)
        features = (values - context.mean) / context.scale
        test_kernel = rbf_kernel(features, context.fit_features, context.sigma)
        original = test_kernel @ state.alpha
        regularization = float(LAMBDA_GRID[state.lambda_index])

        for scale in KERNEL_SCALES:
            transformed = (scale * test_kernel) @ (state.alpha / scale)
            prediction_difference = float(np.max(np.abs(transformed - original)))
            tv_difference = mean_tv(cell.target_coordinates["iid_test"], transformed) - mean_tv(
                cell.target_coordinates["iid_test"], original
            )
            passed = prediction_difference <= PREDICTION_TOLERANCE and abs(tv_difference) <= TV_TOLERANCE
            rows.append(
                {
                    "audit_cell": cell.cell_id,
                    "target_id": cell.target_id,
                    "task_id": cell.task_id,
                    "kernel_scale": scale,
                    "lambda_prime": scale * regularization,
                    "inverse_lambda_prime": 1.0 / (scale * regularization),
                    "max_prediction_difference": prediction_difference,
                    "test_tv_difference": tv_difference,
                    "complete_coded_bits": int(row_contract["complete_coded_bits"]),
                    "q": 3,
                    "status": "completed" if passed else "numerical_failure",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Cartesian-universe and numerical checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell]) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 12 output is missing columns: %s" % sorted(missing))
    if len(cells) != EXPECTED_AUDIT_CELLS:
        raise RuntimeError("Exp 12 requires exactly nine frozen audit cells")
    expected_rows = EXPECTED_AUDIT_CELLS * len(KERNEL_SCALES)
    if len(results) != expected_rows:
        raise RuntimeError("Exp 12 expected %d rows, observed %d" % (expected_rows, len(results)))
    if set(results["kernel_scale"].astype(float)) != set(KERNEL_SCALES):
        raise RuntimeError("Exp 12 kernel-scale universe is incomplete")
    if results.duplicated(["audit_cell", "kernel_scale"]).any():
        raise RuntimeError("Exp 12 contains duplicate Cartesian keys")
    if set(results["q"].astype(int)) != {3}:
        raise RuntimeError("Exp 12 must use q=3 throughout")
    if not set(results["status"].astype(str)).issubset({"completed", "numerical_failure"}):
        raise RuntimeError("Exp 12 contains an unknown numerical status")
    numeric = results[
        [
            "lambda_prime",
            "inverse_lambda_prime",
            "max_prediction_difference",
            "test_tv_difference",
            "complete_coded_bits",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Exp 12 contains NaN or infinite accounting values")
    if not np.allclose(results["lambda_prime"] * results["inverse_lambda_prime"], 1.0, atol=1e-12, rtol=1e-12,):
        raise RuntimeError("Exp 12 reciprocal regularization accounting failed")
    coded_sizes = results.groupby("audit_cell")["complete_coded_bits"].nunique()
    if not (coded_sizes == 1).all():
        raise RuntimeError("Exp 12 kernel scale changed the retained-state length")


# ---------------------------------------------------------------------
# 3. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 12: KRR kernel-scale gauge check").parse_args()
    cells = get_cells(args.responses)
    results = run_kernel_scale_gauge(cells)
    validate_results(results, cells)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp12_kernel_scale_gauge.csv")

    print(results.groupby("kernel_scale", as_index=False)["max_prediction_difference"].max().to_string(index=False))
    print("Saved:", output)
    if (results["status"] == "numerical_failure").any():
        raise RuntimeError("Exp 12 gauge tolerance failed; inspect the saved rows")


if __name__ == "__main__":
    main()
