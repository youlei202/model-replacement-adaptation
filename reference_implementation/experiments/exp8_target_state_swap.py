# experiments/exp8_target_state_swap.py
#
# Exp 8: Target-state swap intervention.
#
# Within each task, three target models share exactly the same peer-response
# interface. Convex, RFF, KRR, and compact-MLP states are frozen for each target.
# At evaluation we replace the correct state with a state trained for another
# target while keeping the recipient inputs and peers fixed.
#
# Ordered donor -> recipient swaps test target-specific dependence. Neutral and
# row-shuffled-refit states are registered negative controls. Paired bootstrap
# intervals resample near-duplicate item clusters within the fixed audit cell;
# targets are not treated as independent draws from a model population.

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
from kernel_isqed.adapters import AdapterContext, AdapterState, predict_coordinates
from kernel_isqed.audit import AuditCell, get_cells, paired_bootstrap_interval, split_tv
from kernel_isqed.contracts import TARGETS, TASKS
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table
from kernel_isqed.response import coordinates_to_probabilities, total_variation


# ---------------------------------------------------------------------
# 1. Registered swap adapters
# ---------------------------------------------------------------------
SWAP_ADAPTERS = ("convex", "rff", "krr", "mlp")
CONTROL_CONDITIONS = ("other_target", "neutral", "row_shuffled_refit")
RESULT_COLUMNS = (
    "task_id",
    "adapter",
    "donor",
    "recipient",
    "condition",
    "correct_tv",
    "controlled_tv",
    "delta_swap",
    "ci_lower",
    "ci_upper",
    "donor_target_correct_tv",
    "donor_target_controlled_tv",
    "donor_target_tv_change",
    "donor_ci_lower",
    "donor_ci_upper",
    "q",
    "status",
)


def expected_run_keys():
    rows = []
    target_ids = [target.model_id for target in TARGETS]
    for task in TASKS:
        for recipient in target_ids:
            for adapter in SWAP_ADAPTERS:
                for donor in target_ids:
                    if donor != recipient:
                        rows.append(
                            {
                                "key_family": "EXP8_SWAP",
                                "task_id": task.task_id,
                                "donor": donor,
                                "recipient": recipient,
                                "adapter": adapter,
                                "condition": "other_target",
                                "q": 3,
                            }
                        )
                for condition in ("neutral", "row_shuffled_refit"):
                    rows.append(
                        {
                            "key_family": "EXP8_SWAP",
                            "task_id": task.task_id,
                            "donor": condition,
                            "recipient": recipient,
                            "adapter": adapter,
                            "condition": condition,
                            "q": 3,
                        }
                    )
    return rows


def neutral_state(state: AdapterState) -> AdapterState:
    neutral = AdapterState(
        state.adapter, state.peer_indices, state.lambda_index, state.rank
    )
    if state.adapter == "convex":
        neutral.weights = np.ones(len(state.peer_indices), dtype=np.float64) / len(
            state.peer_indices
        )
    elif state.adapter in ("linear", "rff"):
        neutral.coefficient = np.zeros_like(state.coefficient)
        neutral.bias = np.zeros_like(state.bias)
    elif state.adapter == "mlp":
        neutral.hidden_weight = np.zeros_like(state.hidden_weight)
        neutral.hidden_bias = np.zeros_like(state.hidden_bias)
        neutral.output_weight = np.zeros_like(state.output_weight)
        neutral.output_bias = np.zeros_like(state.output_bias)
    else:
        neutral.alpha = np.zeros_like(state.alpha)
    return neutral


# ---------------------------------------------------------------------
# 2. State-swap experiment
# ---------------------------------------------------------------------
def run_target_state_swap(
    cells: Sequence[AuditCell],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ledger = []
    frozen = FrozenStateStore()
    for task_id in sorted(set(cell.task_id for cell in cells)):
        task_cells = [cell for cell in cells if cell.task_id == task_id]
        states: Dict[Tuple[str, str], Tuple[AdapterState, AdapterContext]] = {}

        for cell in task_cells:
            context = cell.context((0, 1, 2))
            for adapter in SWAP_ADAPTERS:
                state, _ = frozen.load(cell, context, "native", adapter)
                if state is None:
                    raise RuntimeError("native swap state cannot be infeasible")
                states[(cell.target_id, adapter)] = (state, context)

        for recipient in task_cells:
            for adapter in SWAP_ADAPTERS:
                correct, context = states[(recipient.target_id, adapter)]
                correct_prediction = predict_coordinates(
                    correct, context, recipient.peer_coordinates["iid_test"],
                )
                correct_loss = split_tv(recipient, "iid_test", correct_prediction)

                for donor in task_cells:
                    if donor.target_id == recipient.target_id:
                        continue
                    donor_state, donor_context = states[(donor.target_id, adapter)]
                    donor_prediction = predict_coordinates(
                        donor_state,
                        donor_context,
                        recipient.peer_coordinates["iid_test"],
                    )
                    donor_loss = split_tv(recipient, "iid_test", donor_prediction)
                    estimate, lower, upper = paired_bootstrap_interval(
                        donor_loss,
                        correct_loss,
                        clusters=recipient.item_clusters["iid_test"],
                    )
                    if not np.array_equal(
                        donor.item_ids["iid_test"], recipient.item_ids["iid_test"]
                    ):
                        raise RuntimeError(
                            "donor and recipient IID items are not aligned"
                        )
                    donor_target_correct_loss = _coordinate_tv(
                        donor.target_coordinates["iid_test"], correct_prediction
                    )
                    donor_target_controlled_loss = _coordinate_tv(
                        donor.target_coordinates["iid_test"], donor_prediction
                    )
                    (
                        donor_target_change,
                        donor_lower,
                        donor_upper,
                    ) = paired_bootstrap_interval(
                        donor_target_controlled_loss,
                        donor_target_correct_loss,
                        clusters=recipient.item_clusters["iid_test"],
                    )
                    correct_mean, donor_mean = _cluster_weighted_means(
                        correct_loss, donor_loss, recipient.item_clusters["iid_test"]
                    )
                    donor_correct_mean, donor_controlled_mean = _cluster_weighted_means(
                        donor_target_correct_loss,
                        donor_target_controlled_loss,
                        recipient.item_clusters["iid_test"],
                    )
                    rows.append(
                        {
                            "task_id": task_id,
                            "adapter": adapter,
                            "donor": donor.target_id,
                            "recipient": recipient.target_id,
                            "condition": "other_target",
                            "correct_tv": correct_mean,
                            "controlled_tv": donor_mean,
                            "delta_swap": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "donor_target_correct_tv": donor_correct_mean,
                            "donor_target_controlled_tv": donor_controlled_mean,
                            "donor_target_tv_change": donor_target_change,
                            "donor_ci_lower": donor_lower,
                            "donor_ci_upper": donor_upper,
                            "q": 3,
                            "status": "completed",
                        }
                    )
                    ledger.extend(
                        _swap_loss_rows(
                            recipient,
                            adapter,
                            donor.target_id,
                            "other_target",
                            correct_loss,
                            donor_loss,
                            donor_target_correct_loss,
                            donor_target_controlled_loss,
                        )
                    )

                neutral, _ = frozen.load(
                    recipient, context, "swap_control", adapter, condition="neutral",
                )
                if neutral is None:
                    raise RuntimeError("neutral swap control cannot be infeasible")
                neutral_loss = split_tv(
                    recipient,
                    "iid_test",
                    predict_coordinates(
                        neutral, context, recipient.peer_coordinates["iid_test"]
                    ),
                )
                shuffled, _ = frozen.load(
                    recipient,
                    context,
                    "swap_control",
                    adapter,
                    condition="row_shuffled_refit",
                )
                if shuffled is None:
                    raise RuntimeError("row-shuffled swap control cannot be infeasible")
                shuffled_loss = split_tv(
                    recipient,
                    "iid_test",
                    predict_coordinates(
                        shuffled, context, recipient.peer_coordinates["iid_test"]
                    ),
                )

                for condition, loss in (
                    ("neutral", neutral_loss),
                    ("row_shuffled_refit", shuffled_loss),
                ):
                    estimate, lower, upper = paired_bootstrap_interval(
                        loss,
                        correct_loss,
                        clusters=recipient.item_clusters["iid_test"],
                    )
                    correct_mean, controlled_mean = _cluster_weighted_means(
                        correct_loss, loss, recipient.item_clusters["iid_test"]
                    )
                    rows.append(
                        {
                            "task_id": task_id,
                            "adapter": adapter,
                            "donor": condition,
                            "recipient": recipient.target_id,
                            "condition": condition,
                            "correct_tv": correct_mean,
                            "controlled_tv": controlled_mean,
                            "delta_swap": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "donor_target_correct_tv": np.nan,
                            "donor_target_controlled_tv": np.nan,
                            "donor_target_tv_change": np.nan,
                            "donor_ci_lower": np.nan,
                            "donor_ci_upper": np.nan,
                            "q": 3,
                            "status": "completed",
                        }
                    )
                    ledger.extend(
                        _swap_loss_rows(
                            recipient, adapter, condition, condition, correct_loss, loss
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def _cluster_weighted_means(
    correct_loss: np.ndarray, controlled_loss: np.ndarray, clusters: np.ndarray
) -> Tuple[float, float]:
    frame = pd.DataFrame(
        {
            "cluster": np.asarray(clusters).astype(str),
            "correct": correct_loss,
            "controlled": controlled_loss,
        }
    )
    means = (
        frame.groupby("cluster", sort=False)[["correct", "controlled"]].mean().mean()
    )
    return float(means["correct"]), float(means["controlled"])


def _coordinate_tv(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return total_variation(
        coordinates_to_probabilities(actual), coordinates_to_probabilities(predicted)
    )


def _swap_loss_rows(
    cell: AuditCell,
    adapter: str,
    donor: str,
    condition: str,
    correct_loss: np.ndarray,
    controlled_loss: np.ndarray,
    donor_target_correct_loss: np.ndarray = None,
    donor_target_controlled_loss: np.ndarray = None,
) -> Sequence[Dict[str, object]]:
    return [
        {
            "task_id": cell.task_id,
            "recipient": cell.target_id,
            "adapter": adapter,
            "donor": donor,
            "condition": condition,
            "item_id": str(cell.item_ids["iid_test"][index]),
            "item_cluster": str(cell.item_clusters["iid_test"][index]),
            "domain": str(cell.domains["iid_test"][index]),
            "correct_tv_loss": float(correct_loss[index]),
            "controlled_tv_loss": float(controlled_loss[index]),
            "paired_delta": float(controlled_loss[index] - correct_loss[index]),
            "donor_target_correct_tv_loss": (
                float(donor_target_correct_loss[index])
                if donor_target_correct_loss is not None
                else np.nan
            ),
            "donor_target_controlled_tv_loss": (
                float(donor_target_controlled_loss[index])
                if donor_target_controlled_loss is not None
                else np.nan
            ),
            "paired_donor_target_tv_change": (
                float(
                    donor_target_controlled_loss[index]
                    - donor_target_correct_loss[index]
                )
                if donor_target_correct_loss is not None
                else np.nan
            ),
        }
        for index in range(len(correct_loss))
    ]


# ---------------------------------------------------------------------
# 3. Swap-universe and interval checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell]) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 8 output is missing columns: %s" % sorted(missing))
    if len(cells) != 9:
        raise RuntimeError("Exp 8 requires the nine frozen target/task cells")
    expected_rows = 3 * 3 * len(SWAP_ADAPTERS) * 4
    if len(results) != expected_rows:
        raise RuntimeError(
            "Exp 8 expected %d rows, observed %d" % (expected_rows, len(results))
        )
    if set(results["adapter"].astype(str)) != set(SWAP_ADAPTERS):
        raise RuntimeError("Exp 8 adapter universe is incomplete")
    if set(results["condition"].astype(str)) != set(CONTROL_CONDITIONS):
        raise RuntimeError("Exp 8 control universe is incomplete")
    keys = ["task_id", "adapter", "recipient", "condition", "donor"]
    if results.duplicated(keys).any():
        raise RuntimeError("Exp 8 contains duplicate ordered swap rows")
    if set(results["q"].astype(int)) != {3} or set(results["status"].astype(str)) != {
        "completed"
    }:
        raise RuntimeError("Exp 8 contains an invalid query count or status")
    numeric = results[
        ["correct_tv", "controlled_tv", "delta_swap", "ci_lower", "ci_upper"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Exp 8 contains non-finite swap estimates")
    if np.any(numeric[:, 3] > numeric[:, 2]) or np.any(numeric[:, 2] > numeric[:, 4]):
        raise RuntimeError("Exp 8 confidence intervals do not contain their estimates")
    if not np.allclose(
        results["controlled_tv"] - results["correct_tv"],
        results["delta_swap"],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError("Exp 8 paired deltas disagree with the reported mean losses")
    donor = results[results["condition"] == "other_target"]
    donor_numeric = donor[
        [
            "donor_target_correct_tv",
            "donor_target_controlled_tv",
            "donor_target_tv_change",
            "donor_ci_lower",
            "donor_ci_upper",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(donor_numeric).all():
        raise RuntimeError("Exp 8 donor-direction evidence is incomplete")
    if np.any(donor_numeric[:, 3] > donor_numeric[:, 2]) or np.any(
        donor_numeric[:, 2] > donor_numeric[:, 4]
    ):
        raise RuntimeError(
            "Exp 8 donor-direction intervals do not contain their estimates"
        )
    if not np.allclose(
        donor["donor_target_controlled_tv"] - donor["donor_target_correct_tv"],
        donor["donor_target_tv_change"],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "Exp 8 donor-direction deltas disagree with their mean losses"
        )


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 8: target-state swap intervention").parse_args()
    cells = get_cells(args.responses)
    results, ledger = run_target_state_swap(cells)
    validate_results(results, cells)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp8_target_state_swap.csv")
    write_table(
        ledger, RESULTS_DIR / "tables/item_losses/exp8_swap_item_losses.parquet"
    )

    print(
        results.groupby(["adapter", "condition"], as_index=False)["delta_swap"]
        .median()
        .to_string(index=False)
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
