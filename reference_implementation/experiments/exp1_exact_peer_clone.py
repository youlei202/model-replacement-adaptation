# experiments/exp1_exact_peer_clone.py
#
# Exp 1: Exact online-peer clone control.
#
# This experiment isolates the zero-state information path. We declare the
# target response to be the response of frozen peer P1 on 512 RewardBench
# addresses. At deployment the decoder may query P1 once, but it receives no
# target-trained state. The prediction must therefore match the target exactly:
#
#   S = empty, q = 1, P_hat(z) = P1(z), TV(P_target, P_hat) = 0.
#
# The result is a contract check, not evidence about the nine real target/task
# audit cells. Any non-zero value is an implementation or accounting failure.

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
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.frozen import require_frozen_evaluation
from kernel_isqed.io import save_experiment
from kernel_isqed.response import coordinates_to_probabilities, total_variation


# ---------------------------------------------------------------------
# 1. Frozen contract
# ---------------------------------------------------------------------
ADDRESS_COUNT = 512
PEER_INDEX = 0
TOLERANCE = 1e-12
RESULT_COLUMNS = (
    "condition",
    "task_id",
    "peer_index",
    "split",
    "distortion",
    "max_distortion",
    "complete_coded_bits",
    "q",
    "n_items",
    "status",
)


def expected_run_keys():
    return [
        {
            "key_family": "EXP1_CLONE",
            "contract": "rewardbench_512",
            "q": 1,
            "state": "empty",
        }
    ]


def reference_cell(cells: Sequence[AuditCell]) -> AuditCell:
    rewardbench = [cell for cell in cells if cell.task_id == "rewardbench"]
    if len(cells) != 9 or len(rewardbench) != 3:
        raise RuntimeError("Exp 1 requires the frozen 3-target x 3-task audit grid")
    rewardbench.sort(key=lambda cell: cell.target_id)
    reference = rewardbench[0]
    if len(reference.item_ids["iid_test"]) < ADDRESS_COUNT:
        raise RuntimeError(
            "Exp 1 requires at least %d RewardBench IID addresses" % ADDRESS_COUNT
        )

    reference_ids = reference.item_ids["iid_test"][:ADDRESS_COUNT]
    reference_peer = reference.peer_coordinates["iid_test"][:ADDRESS_COUNT]
    for cell in rewardbench[1:]:
        if not np.array_equal(cell.item_ids["iid_test"][:ADDRESS_COUNT], reference_ids):
            raise RuntimeError("Exp 1 found target-dependent RewardBench addresses")
        if not np.array_equal(
            cell.peer_coordinates["iid_test"][:ADDRESS_COUNT], reference_peer
        ):
            raise RuntimeError("Exp 1 found target-dependent peer responses")
    return reference


# ---------------------------------------------------------------------
# 2. Exact-clone evaluation
# ---------------------------------------------------------------------
def run_exact_peer_clone(cells: Sequence[AuditCell]) -> pd.DataFrame:
    cell = reference_cell(cells)
    peer_response = cell.peer_coordinates["iid_test"][:ADDRESS_COUNT, PEER_INDEX, :]
    target_response = peer_response.copy()

    target_probability = coordinates_to_probabilities(target_response)
    deployment_probability = coordinates_to_probabilities(peer_response)
    loss = total_variation(target_probability, deployment_probability)
    max_distortion = float(loss.max())

    return pd.DataFrame(
        [
            {
                "condition": "exact_online_peer_clone",
                "task_id": "rewardbench",
                "peer_index": PEER_INDEX,
                "split": "iid_test",
                "distortion": float(loss.mean()),
                "max_distortion": max_distortion,
                "complete_coded_bits": 0,
                "q": 1,
                "n_items": len(loss),
                "status": "completed"
                if max_distortion <= TOLERANCE
                else "accounting_failure",
            }
        ]
    )


# ---------------------------------------------------------------------
# 3. Contract checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 1 output is missing columns: %s" % sorted(missing))
    if len(results) != 1:
        raise RuntimeError("Exp 1 must emit one pre-registered contract row")
    row = results.iloc[0]
    if int(row["complete_coded_bits"]) != 0 or int(row["q"]) != 1:
        raise RuntimeError("Exp 1 violated the zero-state, one-query contract")
    if int(row["n_items"]) != ADDRESS_COUNT:
        raise RuntimeError("Exp 1 evaluated the wrong address count")
    if not np.isfinite(float(row["distortion"])) or not np.isfinite(
        float(row["max_distortion"])
    ):
        raise RuntimeError("Exp 1 produced a non-finite distortion")
    if float(row["max_distortion"]) > TOLERANCE or row["status"] != "completed":
        raise RuntimeError("Exp 1 exact-clone accounting check failed")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 1: exact online-peer clone control").parse_args()
    require_frozen_evaluation()
    cells = get_cells(args.responses)
    results = run_exact_peer_clone(cells)
    validate_results(results)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp1_exact_peer_clone.csv")

    print(results.to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
