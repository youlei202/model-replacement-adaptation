# experiments/exp7_quantized_state_disclosure.py
#
# Exp 7: Declared-decoder disclosure across state codecs.
#
# Five adapters are selected at q=3 using fit + validation and serialized before
# the test unlock. A clean decoder loads each frozen native or quantized state
# using only the declared public context C, then reports fitting-table recovery
# and held-out IID error.
#
# Complete coded bits include tensor payloads, selected hyperparameters, shape,
# peer-subset ID, quantizer scale, and state metadata. Raw file size is never
# used as the rate axis.

from __future__ import annotations

import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.adapters import (
    AdapterContext,
    AdapterState,
    mean_tv,
    predict_coordinates,
)
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.contracts import (
    ADAPTERS,
    NATIVE_CODECS,
    QUANTIZED_BITS,
    SPECTRAL_RANKS,
    TASKS,
)
from kernel_isqed.io import save_experiment
from kernel_isqed.frozen import FrozenStateStore
from kernel_isqed.response import coordinates_to_probabilities, total_variation


# ---------------------------------------------------------------------
# 1. Registered state representations
# ---------------------------------------------------------------------
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "adapter",
    "codec",
    "rank",
    "complete_coded_bits",
    "bits_per_fit_row",
    "validation_tv",
    "fit_recovery_mae",
    "fit_probability_tv",
    "iid_test_tv",
    "decoder_type",
    "q",
    "status",
)


def expected_run_keys(audit_grid):
    rows = []
    codecs = tuple("q%d" % bits for bits in QUANTIZED_BITS) + NATIVE_CODECS
    for cell_id, target_id, task_id in audit_grid:
        task = next(contract for contract in TASKS if contract.task_id == task_id)
        for adapter in ADAPTERS:
            ranks = ("validation_selected",)
            if adapter == "krr":
                ranks = tuple(rank for rank in SPECTRAL_RANKS if rank < task.fit) + (
                    task.fit,
                )
            for codec in codecs:
                for rank in ranks:
                    rows.append(
                        {
                            "key_family": "EXP7_DISCLOSURE",
                            "audit_cell": cell_id,
                            "target_id": target_id,
                            "task_id": task_id,
                            "adapter": adapter,
                            "codec": codec,
                            "retained_rank": str(rank),
                            "q": 3,
                        }
                    )
    return rows


def registered_codecs(fast: bool) -> Tuple[str, ...]:
    if fast:
        return "q2", "q8", "native_fp32"
    quantized = tuple("q%d" % bits for bits in QUANTIZED_BITS)
    return quantized + ("native_fp16", "native_fp32", "native_fp64")


def retained_ranks(
    state: AdapterState, context: AdapterContext, fast: bool,
) -> Tuple[int, ...]:
    if state.adapter != "krr":
        return (int(state.rank),) if state.adapter in ("linear", "rff", "mlp") else (0,)

    full_rank = len(context.fit_features)
    ranks = tuple(rank for rank in SPECTRAL_RANKS if rank < full_rank) + (full_rank,)
    return (ranks[0], ranks[-1]) if fast else ranks


# ---------------------------------------------------------------------
# 2. Codec disclosure sweep
# ---------------------------------------------------------------------
def run_quantized_state_disclosure(
    cells: Sequence[AuditCell], fast: bool = False,
) -> pd.DataFrame:
    if fast or len(cells) * len(ADAPTERS) <= 1:
        row_groups = [
            _disclosure_rows_for_cell_adapter(cell, adapter, fast)
            for cell in cells
            for adapter in ADAPTERS
        ]
    else:
        workers = min(
            len(cells) * len(ADAPTERS),
            int(os.environ.get("KERNEL_ISQED_EXPERIMENT_WORKERS", os.cpu_count() or 1)),
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            row_groups = list(
                executor.map(
                    _disclosure_rows_for_cell_adapter_worker,
                    [(cell, adapter, fast) for cell in cells for adapter in ADAPTERS],
                )
            )
    rows = []
    for group in row_groups:
        rows.extend(group)
    return pd.DataFrame(rows)


def _disclosure_rows_for_cell_adapter(
    cell: AuditCell, adapter: str, fast: bool
) -> list:
    rows = []
    frozen = FrozenStateStore()
    context = cell.context((0, 1, 2))
    selected = frozen.manifest[
        (frozen.manifest["audit_cell"].astype(str) == cell.cell_id)
        & (frozen.manifest["state_family"].astype(str) == "disclosure")
        & (frozen.manifest["adapter"].astype(str) == adapter)
        & (frozen.manifest["codec"].astype(str).isin(registered_codecs(fast)))
    ].sort_values(["codec", "retained_rank"])
    for frozen_row in selected.itertuples(index=False):
        decoded, row_contract = frozen.load(
            cell,
            context,
            "disclosure",
            adapter,
            codec=frozen_row.codec,
            retained_rank=frozen_row.retained_rank,
        )
        if decoded is None:
            raise RuntimeError("disclosure state cannot be infeasible")
        fit_prediction = predict_coordinates(
            decoded, context, cell.peer_coordinates["fit"],
        )
        test_prediction = predict_coordinates(
            decoded, context, cell.peer_coordinates["iid_test"],
        )
        fit_probability = coordinates_to_probabilities(fit_prediction)
        target_probability = coordinates_to_probabilities(
            cell.target_coordinates["fit"]
        )
        rows.append(
            {
                "audit_cell": cell.cell_id,
                "target_id": cell.target_id,
                "task_id": cell.task_id,
                "adapter": adapter,
                "codec": str(row_contract["codec"]),
                "rank": int(row_contract["rank"]),
                "complete_coded_bits": int(row_contract["complete_coded_bits"]),
                "bits_per_fit_row": int(row_contract["complete_coded_bits"])
                / len(context.fit_features),
                "validation_tv": float(row_contract["validation_tv"]),
                "fit_recovery_mae": float(
                    np.mean(
                        np.abs(fit_prediction - cell.target_coordinates["fit"])
                    )
                ),
                "fit_probability_tv": float(
                    np.mean(
                        total_variation(target_probability, fit_probability)
                    )
                ),
                "iid_test_tv": mean_tv(
                    cell.target_coordinates["iid_test"], test_prediction
                ),
                "decoder_type": "declared_deployment_decoder",
                "q": 3,
                "status": "completed",
            }
        )
    return rows


def _disclosure_rows_for_cell_adapter_worker(payload):
    cell, adapter, fast = payload
    return _disclosure_rows_for_cell_adapter(cell, adapter, fast)


# ---------------------------------------------------------------------
# 3. Result checks
# ---------------------------------------------------------------------
def validate_results(
    results: pd.DataFrame, cells: Sequence[AuditCell], fast: bool
) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 7 output is missing columns: %s" % sorted(missing))
    if len(cells) != 9:
        raise RuntimeError("Exp 7 requires the nine frozen target/task cells")
    if set(results["adapter"].astype(str)) != set(ADAPTERS):
        raise RuntimeError("Exp 7 did not evaluate every registered adapter")
    if set(results["codec"].astype(str)) != set(registered_codecs(fast)):
        raise RuntimeError("Exp 7 codec universe is incomplete")
    keys = ["audit_cell", "adapter", "codec", "rank"]
    if results.duplicated(keys).any():
        raise RuntimeError("Exp 7 contains duplicate state representations")
    if set(results["q"].astype(int)) != {3}:
        raise RuntimeError("Exp 7 must query all three peers")
    if set(results["status"].astype(str)) != {"completed"}:
        raise RuntimeError("Exp 7 contains an incomplete decoder row")

    numeric = results[
        [
            "complete_coded_bits",
            "bits_per_fit_row",
            "validation_tv",
            "fit_recovery_mae",
            "fit_probability_tv",
            "iid_test_tv",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (results["complete_coded_bits"] <= 0).any():
        raise RuntimeError("Exp 7 contains invalid rate or distortion values")
    for _, curve in results.groupby(["audit_cell", "adapter", "rank"]):
        bits = curve.set_index("codec")["complete_coded_bits"]
        native = [
            name
            for name in ("native_fp16", "native_fp32", "native_fp64")
            if name in bits
        ]
        if native and list(bits[native]) != sorted(bits[native]):
            raise RuntimeError("Exp 7 native precision bit counts are not monotone")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = common_parser("Exp 7: quantized state disclosure", include_fast=True)
    args = parser.parse_args()
    cells = get_cells(args.responses)
    results = run_quantized_state_disclosure(cells, args.fast)
    validate_results(results, cells, args.fast)
    results = tag_run_mode(results, args.fast)
    output = save_experiment(results, "exp7_quantized_state_disclosure.csv")

    print(
        results.groupby(["adapter", "codec"], as_index=False)["fit_recovery_mae"]
        .median()
        .to_string(index=False)
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
