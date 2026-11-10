# experiments/exp11_fit_size_generalization.py
#
# Exp 11: Nested fit-size generalization.
#
# Each audit cell is refit on deterministic nested prefixes with labels
# {128, 256, 512, 1024, m_max}. The explicit m_max label is retained even for
# RewardBench, where m_max equals 1024, because the five-condition Cartesian
# universe is part of the frozen protocol.
#
# Five adapters, including the compact two-layer MLP, are selected independently
# at each prefix before test unlock. Fit, validation, IID, and OOD TV are then
# reported from the frozen states without test-dependent tuning.

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
from kernel_isqed.adapters import predict_coordinates
from kernel_isqed.audit import AuditCell, get_cells, paired_bootstrap_interval, split_tv
from kernel_isqed.contracts import ADAPTERS, FIT_SIZES
from kernel_isqed.io import save_experiment
from kernel_isqed.frozen import FrozenStateStore


SPLITS = ("fit", "validation", "iid_test", "ood_test")
EXPECTED_AUDIT_CELLS = 9
RESULT_COLUMNS = (
    "audit_cell",
    "target_id",
    "task_id",
    "adapter",
    "fit_size",
    "fit_size_label",
    "split",
    "tv",
    "ci_lower",
    "ci_upper",
    "bootstrap_unit",
    "validation_selected_tv",
    "q",
    "status",
)


def expected_run_keys(audit_grid):
    return [
        {
            "key_family": "EXP11_GENERALIZATION",
            "audit_cell": cell_id,
            "target_id": target_id,
            "task_id": task_id,
            "adapter": adapter,
            "fit_size_label": size_label,
            "split": split,
            "q": 3,
        }
        for cell_id, target_id, task_id in audit_grid
        for adapter in ADAPTERS
        for size_label in tuple(str(value) for value in FIT_SIZES) + ("m_max",)
        for split in SPLITS
    ]


def fit_prefix_cell(cell: AuditCell, size: int) -> AuditCell:
    item_ids = dict(cell.item_ids)
    peer_coordinates = dict(cell.peer_coordinates)
    target_coordinates = dict(cell.target_coordinates)
    input_text = dict(cell.input_text) if cell.input_text is not None else None
    item_clusters = dict(cell.item_clusters) if cell.item_clusters is not None else None
    domains = dict(cell.domains) if cell.domains is not None else None
    gold = dict(cell.gold) if cell.gold is not None else None
    item_ids["fit"] = item_ids["fit"][:size]
    peer_coordinates["fit"] = peer_coordinates["fit"][:size]
    target_coordinates["fit"] = target_coordinates["fit"][:size]
    if input_text is not None:
        input_text["fit"] = input_text["fit"][:size]
    if item_clusters is not None:
        item_clusters["fit"] = item_clusters["fit"][:size]
    if domains is not None:
        domains["fit"] = domains["fit"][:size]
    if gold is not None:
        gold["fit"] = gold["fit"][:size]
    return AuditCell(
        cell.target_id,
        cell.task_id,
        item_ids,
        peer_coordinates,
        target_coordinates,
        input_text,
        item_clusters,
        domains,
        gold,
    )


# ---------------------------------------------------------------------
# 1. Nested-prefix experiment
# ---------------------------------------------------------------------
def run_fit_size_generalization(cells: Sequence[AuditCell], fast: bool = False,) -> pd.DataFrame:
    rows = []
    frozen = FrozenStateStore()
    for cell in cells:
        sizes = [(str(size), min(size, len(cell.item_ids["fit"]))) for size in FIT_SIZES]
        sizes.append(("m_max", len(cell.item_ids["fit"])))
        if fast:
            sizes = [sizes[0], sizes[-1]]

        for size_label, fit_size in sizes:
            prefix = fit_prefix_cell(cell, fit_size)
            context = prefix.context((0, 1, 2))
            for adapter in ADAPTERS:
                state, row_contract = frozen.load(cell, context, "fit_size", adapter, fit_size_label=size_label,)
                if state is None:
                    raise RuntimeError("fit-size state cannot be infeasible")
                for split in SPLITS:
                    prediction = predict_coordinates(state, context, prefix.peer_coordinates[split],)
                    losses = split_tv(prefix, split, prediction)
                    blocks = prefix.domains[split] if split == "ood_test" else None
                    estimate, lower, upper = paired_bootstrap_interval(
                        losses, clusters=prefix.item_clusters[split], blocks=blocks,
                    )
                    rows.append(
                        {
                            "audit_cell": cell.cell_id,
                            "target_id": cell.target_id,
                            "task_id": cell.task_id,
                            "adapter": adapter,
                            "fit_size": fit_size,
                            "fit_size_label": size_label,
                            "split": split,
                            "tv": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "bootstrap_unit": "domain_block" if split == "ood_test" else "item_cluster",
                            "validation_selected_tv": float(row_contract["validation_tv"]),
                            "q": 3,
                            "status": "completed",
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Cartesian-universe and schema checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, cells: Sequence[AuditCell], fast: bool,) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 11 output is missing columns: %s" % sorted(missing))
    if len(cells) != EXPECTED_AUDIT_CELLS:
        raise RuntimeError("Exp 11 requires exactly nine frozen audit cells")

    size_labels = (str(FIT_SIZES[0]), "m_max") if fast else tuple(str(size) for size in FIT_SIZES) + ("m_max",)
    expected_rows = EXPECTED_AUDIT_CELLS * len(ADAPTERS) * len(size_labels) * len(SPLITS)
    if len(results) != expected_rows:
        raise RuntimeError("Exp 11 expected %d rows, observed %d" % (expected_rows, len(results)))
    if set(results["fit_size_label"].astype(str)) != set(size_labels):
        raise RuntimeError("Exp 11 fit-size labels do not match the frozen contract")
    if set(results["adapter"].astype(str)) != set(ADAPTERS):
        raise RuntimeError("Exp 11 adapter universe is incomplete")
    if set(results["split"].astype(str)) != set(SPLITS):
        raise RuntimeError("Exp 11 evaluation splits are incomplete")
    if results.duplicated(["audit_cell", "adapter", "fit_size_label", "split"]).any():
        raise RuntimeError("Exp 11 contains duplicate Cartesian keys")
    if set(results["q"].astype(int)) != {3} or set(results["status"].astype(str)) != {"completed"}:
        raise RuntimeError("Exp 11 contains an invalid q or status value")
    if not np.isfinite(results[["tv", "validation_selected_tv"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Exp 11 contains NaN or infinite errors")


# ---------------------------------------------------------------------
# 3. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = common_parser("Exp 11: nested fit-size IID/OOD generalization", include_fast=True)
    args = parser.parse_args()
    cells = get_cells(args.responses)
    results = run_fit_size_generalization(cells, args.fast)
    validate_results(results, cells, args.fast)
    results = tag_run_mode(results, args.fast)
    output = save_experiment(results, "exp11_fit_size_generalization.csv")

    print(results.groupby(["adapter", "fit_size_label", "split"], as_index=False)["tv"].median().to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
