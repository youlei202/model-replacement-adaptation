#!/usr/bin/env python3
"""Reproduce C.6 on the published split; NOT corrected experimental evidence.

Run: .venv-benchmarks/bin/python benchmarks/reproduce_calibration.py
"""
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from run_selection import ROOT, PEERS, TARGETS, DIMENSIONS, atomic_json, sha
from selection_core import fit_weights, encode, decode, predict, tv


def main():
    started = time.monotonic()
    label = "published_split_diagnostic_not_corrected_evidence"
    source = ROOT / "reference_implementation/data/canonical/matched_responses.parquet"
    frame = pd.read_parquet(source)
    rows = []
    with threadpool_limits(limits=2):
        for target in TARGETS:
            for task, dimension in DIMENSIONS.items():
                data = {}
                for split in ("fit", "iid_test"):
                    values = frame[(frame.task_id == task) & (frame.split == split)]
                    aligned = [values[values.model_id == model].sort_values("item_id") for model in PEERS + (target,)]
                    ids = aligned[-1].item_id.tolist()
                    if not all(part.item_id.tolist() == ids for part in aligned):
                        raise ValueError("Unmatched original response arrays")
                    coordinates = [part[[f"coord_{j}" for j in range(dimension)]].to_numpy(dtype=float) for part in aligned]
                    data[split] = (np.stack(coordinates[:3], axis=1), coordinates[3])
                fit, fit_target = data["fit"]
                test, test_target = data["iid_test"]
                row = {"evidence_scope": label, "target_id": target, "task_id": task, "fit_items": len(fit), "test_items": len(test), "peer_calls": 3}
                for family in ("convex", "nnls"):
                    weights = fit_weights(fit, fit_target, (0, 1, 2), family)
                    encoded = encode(weights[:, None], family, (0, 1, 2), "native_fp64")
                    restored, matrix = decode(encoded.blob)
                    loss = tv(test_target, predict(restored, matrix, test))
                    row[f"{family}_test_tv"] = float(loss.mean())
                    row[f"{family}_actual_bits"] = encoded.bits
                    for peer, weight in enumerate(matrix.reshape(-1)):
                        row[f"{family}_weight_{peer}"] = float(weight)
                row["convex_minus_nnls_tv"] = row["convex_test_tv"] - row["nnls_test_tv"]
                rows.append(row)
    output = ROOT / "benchmarks/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "calibration_original_split.csv", index=False)
    summary = {
        "evidence_scope": label,
        "purpose": "Check that the newly implemented encoded C.6 calibration reproduces the old aggregate. This does not correct original sampling or establish a peer-selection gain.",
        "definition": "Fit nonnegative least squares to flattened target Helmert logit contrasts using one scalar per peer; no sum-to-one constraint. Decode a distinct NNLS family with a 176-bit header and three FP64 coefficients before evaluating test TV. Convex comparator additionally imposes sum(weights)=1.",
        "peer_models": list(PEERS),
        "aggregation": "Equal mean over the nine named target-task cells, on the published fit and iid_test partitions.",
        "convex_mean_test_tv": float(table.convex_test_tv.mean()),
        "nnls_mean_test_tv": float(table.nnls_test_tv.mean()),
        "encoded_nnls_bits": 368,
        "published_rounded_values": {"convex": .3028, "nnls": .2892},
        "data_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "selection_core_sha256": sha(Path(__file__).with_name("selection_core.py")),
        "csv_sha256": sha(output / "calibration_original_split.csv"),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "blas_threads": 2, "elapsed_seconds": time.monotonic() - started},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ".venv-benchmarks/bin/python benchmarks/reproduce_calibration.py",
    }
    atomic_json(output / "calibration_original_split.json", summary)
    print(table[["target_id", "task_id", "convex_test_tv", "nnls_test_tv"]].to_string(index=False))
    print(f"Published-split diagnostic only: convex={summary['convex_mean_test_tv']:.10f}, NNLS={summary['nnls_mean_test_tv']:.10f}")


if __name__ == "__main__":
    main()
