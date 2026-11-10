# experiments/exp2_convex_selector_worlds.py
#
# Exp 2: Sixty-four convex-selector worlds.
#
# We construct 64 targets from convex mixtures of the same three frozen peer
# response tables. The online interface C is identical across worlds; only the
# six-bit selector state identifies which target table must be reproduced.
# For every retained-state budget b in {0,...,6}, the decoder receives the
# nearest registered medoid inside the corresponding binary bucket.
#
# The experiment reports all 64 x 7 pre-registered world/codec rows. The
# affine-rank and pairwise-separation checks must pass before the six-bit result
# is interpreted as an injective selector code.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import common_parser, tag_run_mode
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.controlled import load_world_opening
from kernel_isqed.frozen import require_frozen_evaluation
from kernel_isqed.io import RESULTS_DIR, save_experiment, write_table
from kernel_isqed.response import coordinates_to_probabilities
from kernel_isqed.statistics import fano_lower_bound


# ---------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------
WORLD_COUNT = 64
ADDRESS_COUNT = 512
PEER_COUNT = 3
MINIMUM_SEPARATION = 1e-4
RESULT_COLUMNS = (
    "condition",
    "draw_id",
    "commitment_id",
    "decoded_world",
    "medoid_world",
    "codec_bits",
    "complete_coded_bits",
    "q",
    "distortion",
    "world_decoding_error",
    "fano_lower_bound",
    "affine_rank",
    "minimum_pairwise_tv",
    "status",
)


def expected_run_keys():
    return [
        {
            "key_family": "EXP2_SELECTOR",
            "draw_id": draw_id,
            "codec_bits": coded_bits,
            "commitment_scope": "selector:%d" % draw_id,
        }
        for draw_id in range(WORLD_COUNT)
        for coded_bits in range(7)
    ]


def selector_weights() -> Tuple[np.ndarray, list]:
    weights = []
    openings = []
    for world_id in range(WORLD_COUNT):
        opening = load_world_opening("selector:%d" % world_id)
        rng = np.random.RandomState(opening.seed % (2 ** 32))
        weights.append(rng.dirichlet(np.ones(PEER_COUNT)))
        openings.append(opening)
    return np.asarray(weights), openings


# ---------------------------------------------------------------------
# 2. Controlled worlds
# ---------------------------------------------------------------------
def run_convex_selector_worlds(
    cells: Sequence[AuditCell],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cell = next(cell for cell in cells if cell.task_id == "rewardbench")
    coordinates = cell.peer_coordinates["fit"][:ADDRESS_COUNT]
    peer_tables = np.stack(
        [
            coordinates_to_probabilities(coordinates[:, index, :])
            for index in range(PEER_COUNT)
        ],
        axis=1,
    )
    weights, openings = selector_weights()
    targets = np.einsum("mpk,wp->wmk", peer_tables, weights)
    flattened = targets.reshape(WORLD_COUNT, -1)
    affine_rank = int(np.linalg.matrix_rank(flattened[1:] - flattened[0]))

    distances = np.zeros((WORLD_COUNT, WORLD_COUNT), dtype=np.float64)
    for left in range(WORLD_COUNT):
        distances[left] = (
            0.5
            * np.abs(targets[left][None, :, :] - targets).sum(axis=(1, 2))
            / ADDRESS_COUNT
        )
    separation = float(np.min(distances[np.triu_indices(WORLD_COUNT, 1)]))
    contract_valid = affine_rank >= 2 and separation > MINIMUM_SEPARATION
    oracle_rows = []
    for world_id, opening in enumerate(openings):
        oracle_rows.append(
            {
                "draw_id": world_id,
                "seed_hex": "%064x" % opening.seed,
                "weights": "|".join("%.17g" % value for value in weights[world_id]),
                "commitment_id": opening.commitment.commitment_id,
                "commitment_value": str(opening.commitment.value),
                "blinding": str(opening.commitment.blinding),
            }
        )

    rows = []
    for coded_bits in range(7):
        bucket_width = 1 << (6 - coded_bits)
        decoded_worlds = []
        distortions = []
        medoids = []
        for world_id in range(WORLD_COUNT):
            start = (world_id // bucket_width) * bucket_width
            bucket = np.arange(start, start + bucket_width)
            medoid = int(
                bucket[np.argmin(distances[np.ix_(bucket, bucket)].mean(axis=1))]
            )
            prediction = targets[medoid]
            decoded = int(
                np.argmin(
                    0.5 * np.abs(targets - prediction).sum(axis=(1, 2)) / ADDRESS_COUNT
                )
            )
            distortion = float(
                0.5 * np.abs(targets[world_id] - prediction).sum(axis=1).mean()
            )
            medoids.append(medoid)
            decoded_worlds.append(decoded)
            distortions.append(distortion)

        decoding_error = float(
            np.mean(np.asarray(decoded_worlds) != np.arange(WORLD_COUNT))
        )
        information_bound = (
            fano_lower_bound(WORLD_COUNT, decoding_error) if contract_valid else np.nan
        )
        for world_id in range(WORLD_COUNT):
            rows.append(
                {
                    "condition": "convex_selector",
                    "draw_id": world_id,
                    "commitment_id": openings[world_id].commitment.commitment_id,
                    "decoded_world": decoded_worlds[world_id],
                    "medoid_world": medoids[world_id],
                    "codec_bits": coded_bits,
                    "complete_coded_bits": coded_bits,
                    "q": 3,
                    "distortion": distortions[world_id],
                    "world_decoding_error": decoding_error,
                    "fano_lower_bound": information_bound,
                    "affine_rank": affine_rank,
                    "minimum_pairwise_tv": separation,
                    "status": "completed"
                    if contract_valid
                    else "invalid_noninjective_interface",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


# ---------------------------------------------------------------------
# 3. Registered-universe checks
# ---------------------------------------------------------------------
def validate_results(results: pd.DataFrame, oracle: pd.DataFrame) -> None:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise RuntimeError("Exp 2 output is missing columns: %s" % sorted(missing))
    if len(results) != WORLD_COUNT * 7 or len(oracle) != WORLD_COUNT:
        raise RuntimeError("Exp 2 did not emit the registered 64 x 7 universe")
    if results.duplicated(["draw_id", "codec_bits"]).any():
        raise RuntimeError("Exp 2 contains duplicate world/codec rows")
    if set(results["draw_id"].astype(int)) != set(range(WORLD_COUNT)):
        raise RuntimeError("Exp 2 selector worlds are incomplete")
    if set(results["codec_bits"].astype(int)) != set(range(7)):
        raise RuntimeError("Exp 2 selector code lengths are incomplete")
    if not np.array_equal(
        results["codec_bits"].to_numpy(dtype=int),
        results["complete_coded_bits"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Exp 2 complete code length differs from the selector prefix"
        )
    if set(results["q"].astype(int)) != {PEER_COUNT}:
        raise RuntimeError("Exp 2 changed the registered peer interface")
    if set(results["status"].astype(str)) != {"completed"}:
        raise RuntimeError("Exp 2 failed its injectivity precondition")
    if not results["distortion"].between(0.0, 1.0).all():
        raise RuntimeError("Exp 2 produced an invalid distortion")
    if (
        results["commitment_id"].eq("").any()
        or oracle["commitment_id"].duplicated().any()
    ):
        raise RuntimeError("Exp 2 controlled worlds are not uniquely committed")


# ---------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------
def main() -> None:
    args = common_parser("Exp 2: 64-way convex-selector worlds").parse_args()
    require_frozen_evaluation()
    cells = get_cells(args.responses)
    results, oracle = run_convex_selector_worlds(cells)
    validate_results(results, oracle)
    results = tag_run_mode(results)
    output = save_experiment(results, "exp2_convex_selector_worlds.csv")
    oracle_path = write_table(
        oracle, RESULTS_DIR / "oracle_truth/exp2_selector_openings.parquet"
    )

    summary = results.groupby("codec_bits", as_index=False).agg(
        distortion=("distortion", "mean"),
        decoding_error=("world_decoding_error", "first"),
        fano_bits=("fano_lower_bound", "first"),
    )
    print(summary.to_string(index=False))
    print("Saved:", output)
    print("Private oracle:", oracle_path)


if __name__ == "__main__":
    main()
