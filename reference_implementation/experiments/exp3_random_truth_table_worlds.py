# experiments/exp3_random_truth_table_worlds.py
#
# Exp 3: Random truth-table rate-distortion worlds.
#
# Each draw is an independent Bernoulli(1/2) table on 512 public addresses.
# The deployment state stores a registered prefix of r truth bits and predicts
# zero elsewhere. This gives an exact source law against which the empirical
# distortion and complete code length can be audited.
#
# Public results contain only draw IDs and Pedersen commitment IDs. Truth bits,
# commitment openings, and blindings are written to results/oracle_truth/ and
# must not be exposed to adapter fitting, selection, or decoding.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import tag_run_mode
from kernel_isqed.controlled import load_world_opening
from kernel_isqed.frozen import require_frozen_evaluation
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table
from kernel_isqed.statistics import truth_table_rate_distortion


# ---------------------------------------------------------------------
# 1. Registered source and codecs
# ---------------------------------------------------------------------
TABLE_SIZE = 512
WORLD_COUNT = 512
STORED_BITS = (0, 16, 32, 64, 128, 256, 384, 512)
RESULT_COLUMNS = (
    "row_kind",
    "draw_id",
    "commitment_id",
    "table_size",
    "stored_bits",
    "complete_coded_bits",
    "distortion",
    "analytic_distortion",
    "rate_distortion_lower_bound",
    "q",
    "status",
)


def expected_run_keys():
    rows = [
        {
            "key_family": "EXP3_TRUTH",
            "row_kind": "world",
            "draw_id": draw_id,
            "stored_bits": stored_bits,
            "commitment_scope": "truth:%d" % draw_id,
        }
        for draw_id in range(WORLD_COUNT)
        for stored_bits in STORED_BITS
    ]
    rows.extend(
        {
            "key_family": "EXP3_TRUTH",
            "row_kind": "analytic",
            "draw_id": -1,
            "stored_bits": stored_bits,
        }
        for stored_bits in STORED_BITS
    )
    return rows


def run_truth_table_worlds(
    table_size: int, world_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranks = sorted(set(min(value, table_size) for value in STORED_BITS))
    rows = []
    oracle_rows = []

    for draw_id in range(world_count):
        opening = load_world_opening("truth:%d" % draw_id)
        rng = np.random.RandomState(opening.seed % (2 ** 32))
        truth = rng.randint(0, 2, size=table_size, dtype=np.uint8)
        packed = np.packbits(truth).tobytes()
        oracle_rows.append(
            {
                "draw_id": draw_id,
                "seed_hex": "%064x" % opening.seed,
                "truth_bits": packed.hex(),
                "commitment_id": opening.commitment.commitment_id,
                "commitment_value": str(opening.commitment.value),
                "blinding": str(opening.commitment.blinding),
            }
        )

        for stored_bits in ranks:
            prediction = np.zeros(table_size, dtype=np.uint8)
            prediction[:stored_bits] = truth[:stored_bits]
            analytic_distortion = (table_size - stored_bits) / (2.0 * table_size)
            rows.append(
                {
                    "row_kind": "world",
                    "draw_id": draw_id,
                    "commitment_id": opening.commitment.commitment_id,
                    "table_size": table_size,
                    "stored_bits": stored_bits,
                    "complete_coded_bits": stored_bits,
                    "distortion": float(np.mean(prediction != truth)),
                    "analytic_distortion": analytic_distortion,
                    "rate_distortion_lower_bound": truth_table_rate_distortion(
                        table_size, analytic_distortion
                    ),
                    "q": 0,
                    "status": "completed",
                }
            )

    for stored_bits in ranks:
        distortion = (table_size - stored_bits) / (2.0 * table_size)
        lower_bound = truth_table_rate_distortion(table_size, distortion)
        rows.append(
            {
                "row_kind": "analytic",
                "draw_id": -1,
                "commitment_id": "",
                "table_size": table_size,
                "stored_bits": stored_bits,
                "complete_coded_bits": stored_bits,
                "distortion": distortion,
                "analytic_distortion": distortion,
                "rate_distortion_lower_bound": lower_bound,
                "q": 0,
                "status": "completed"
                if stored_bits + 1e-10 >= lower_bound
                else "accounting_failure",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


# ---------------------------------------------------------------------
# 2. Source-law and accounting checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, oracle: pd.DataFrame) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 3 output is missing columns: %s" % sorted(missing))
    world_rows = results[results["row_kind"] == "world"]
    analytic_rows = results[results["row_kind"] == "analytic"]
    expected_world_rows = WORLD_COUNT * len(STORED_BITS)
    if len(world_rows) != expected_world_rows or len(analytic_rows) != len(STORED_BITS):
        raise RuntimeError("Exp 3 did not emit the registered truth-table universe")
    if len(oracle) != WORLD_COUNT or oracle["draw_id"].duplicated().any():
        raise RuntimeError("Exp 3 private openings are incomplete or duplicated")
    if world_rows.duplicated(["draw_id", "stored_bits"]).any():
        raise RuntimeError("Exp 3 contains duplicate world/rate rows")
    if set(world_rows["draw_id"].astype(int)) != set(range(WORLD_COUNT)):
        raise RuntimeError("Exp 3 truth-table draws are incomplete")
    if set(results["stored_bits"].astype(int)) != set(STORED_BITS):
        raise RuntimeError("Exp 3 retained-prefix grid is incomplete")
    if not np.array_equal(
        results["stored_bits"].to_numpy(dtype=int),
        results["complete_coded_bits"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Exp 3 complete code length differs from the retained prefix"
        )
    if set(results["q"].astype(int)) != {0} or set(results["status"].astype(str)) != {
        "completed"
    }:
        raise RuntimeError("Exp 3 violated the zero-query source contract")
    if not results["distortion"].between(0.0, 1.0).all():
        raise RuntimeError("Exp 3 produced an invalid Hamming distortion")
    if world_rows["commitment_id"].eq("").any():
        raise RuntimeError("Exp 3 contains an uncommitted controlled world")


# ---------------------------------------------------------------------
# 3. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    require_frozen_evaluation()
    results, oracle = run_truth_table_worlds(TABLE_SIZE, WORLD_COUNT)
    validate_results(results, oracle)
    results = tag_run_mode(results)

    output = save_experiment(results, "exp3_random_truth_table_worlds.csv")
    oracle_path = write_table(
        oracle, RESULTS_DIR / "oracle_truth/exp3_truth_table_openings.parquet"
    )
    summary = results.groupby(["row_kind", "stored_bits"], as_index=False)[
        "distortion"
    ].mean()
    print(summary.to_string(index=False))
    print("Saved:", output)
    print("Private oracle:", oracle_path)


if __name__ == "__main__":
    main()
