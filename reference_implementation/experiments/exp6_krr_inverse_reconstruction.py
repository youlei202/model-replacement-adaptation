# experiments/exp6_krr_inverse_reconstruction.py
#
# Exp 6: Algebraic reconstruction from the native KRR state.
#
# For each target/task cell, KRR is selected using fit + validation. The decoder
# receives the canonical state and public kernel context, then reconstructs the
# target fitting-response table through
#
#   Y_fit_hat = (K + m * lambda * I) alpha.
#
# FP16, FP32, and FP64 states are tested separately. Tolerances scale with the
# measured condition number; quantized codecs are deliberately excluded from
# this machine-precision check.

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
from kernel_isqed.adapters import rbf_kernel
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.contracts import LAMBDA_GRID
from kernel_isqed.io import save_experiment
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.response import coordinates_to_probabilities, total_variation


# ---------------------------------------------------------------------
# 1. Registered numerical checks
# ---------------------------------------------------------------------
NATIVE_CODECS = (
    ("native_fp16", np.finfo(np.float16).eps),
    ("native_fp32", np.finfo(np.float32).eps),
    ("native_fp64", np.finfo(np.float64).eps),
)
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "codec",
    "complete_coded_bits",
    "coordinate_mae",
    "max_absolute_error",
    "relative_l2_error",
    "probability_tv",
    "top_choice_recovery",
    "condition_number",
    "numerical_threshold",
    "status",
)


def expected_run_keys(audit_grid):
    return [
        {
            "key_family": "EXP6_INVERSE",
            "audit_cell": cell_id,
            "target_id": target_id,
            "task_id": task_id,
            "adapter": "krr",
            "codec": codec,
            "q": 3,
        }
        for cell_id, target_id, task_id in audit_grid
        for codec, _ in NATIVE_CODECS
    ]


def run_krr_inverse_reconstruction(cells: Sequence[AuditCell]) -> pd.DataFrame:
    rows = []
    frozen = FrozenStateStore()
    for cell in cells:
        context = cell.context((0, 1, 2))
        native_state, _ = frozen.load(cell, context, "native", "krr")
        if native_state is None:
            raise RuntimeError("native KRR state cannot be infeasible")
        kernel = rbf_kernel(context.fit_features, context.fit_features, context.sigma)
        regularization = float(LAMBDA_GRID[native_state.lambda_index])
        system = kernel + len(kernel) * regularization * np.eye(len(kernel))
        condition_number = float(np.linalg.cond(system))
        target = cell.target_coordinates["fit"]
        target_probability = coordinates_to_probabilities(target)

        for codec_name, machine_epsilon in NATIVE_CODECS:
            decoded, row_contract = frozen.load(
                cell, context, "disclosure", "krr", codec=codec_name, retained_rank=len(kernel),
            )
            if decoded is None:
                raise RuntimeError("inverse state cannot be infeasible")
            recovered = system @ decoded.alpha
            residual = recovered - target
            relative_error = float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-15))
            recovered_probability = coordinates_to_probabilities(recovered)
            tolerance = max(1e-6, 100.0 * float(machine_epsilon) * condition_number)

            rows.append(
                {
                    "audit_cell": cell.cell_id,
                    "target_id": cell.target_id,
                    "task_id": cell.task_id,
                    "codec": codec_name,
                    "complete_coded_bits": int(row_contract["complete_coded_bits"]),
                    "coordinate_mae": float(np.mean(np.abs(residual))),
                    "max_absolute_error": float(np.max(np.abs(residual))),
                    "relative_l2_error": relative_error,
                    "probability_tv": float(np.mean(total_variation(target_probability, recovered_probability))),
                    "top_choice_recovery": float(
                        np.mean(np.argmax(target_probability, axis=1) == np.argmax(recovered_probability, axis=1))
                    ),
                    "condition_number": condition_number,
                    "numerical_threshold": tolerance,
                    "status": "completed" if relative_error <= tolerance else "numerical_failure",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Numerical and accounting checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell]) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 6 output is missing columns: %s" % sorted(missing))
    if len(cells) != 9 or len(results) != 9 * len(NATIVE_CODECS):
        raise RuntimeError("Exp 6 did not cover all audit cells and native codecs")
    if set(results["codec"].astype(str)) != {name for name, _ in NATIVE_CODECS}:
        raise RuntimeError("Exp 6 native codec universe is incomplete")
    if results.duplicated(["audit_cell", "codec"]).any():
        raise RuntimeError("Exp 6 contains duplicate reconstruction keys")
    numeric = results[
        [
            "complete_coded_bits",
            "coordinate_mae",
            "max_absolute_error",
            "relative_l2_error",
            "probability_tv",
            "top_choice_recovery",
            "condition_number",
            "numerical_threshold",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Exp 6 contains non-finite numerical diagnostics")
    for _, cell_rows in results.groupby("audit_cell"):
        bits = cell_rows.set_index("codec")["complete_coded_bits"]
        if not bits["native_fp16"] < bits["native_fp32"] < bits["native_fp64"]:
            raise RuntimeError("Exp 6 native state sizes do not increase with precision")


# ---------------------------------------------------------------------
# 3. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 6: native KRR inverse reconstruction").parse_args()
    cells = get_cells(args.responses)
    results = run_krr_inverse_reconstruction(cells)
    validate_results(results, cells)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp6_krr_inverse_reconstruction.csv")

    print(results.groupby("codec", as_index=False)["relative_l2_error"].max().to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
