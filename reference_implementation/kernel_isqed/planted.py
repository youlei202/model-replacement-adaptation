from __future__ import annotations

import hashlib
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .adapters import (
    fit_adapter,
    predict_coordinates,
    rff_features,
    transform_features,
)
from .audit import AuditCell
from .codec import decode_state, encode_state
from .controlled import load_world_opening
from .contracts import LAMBDA_GRID
from .deployment import OperatingPoint, enumerate_operating_points
from .io import PROJECT_ROOT, STATES_DIR
from .response import coordinates_to_probabilities, total_variation
from .statistics import simultaneous_bit_information_summary


# Bits are planted in a public adapter-response basis, not in coordinate
# directions that the adapter cannot represent. Semantic-TV calibration gives
# strength the same interpretation in every audit cell and adapter.
PLANTED_SCOPE_ID = "response_geometry_capacity_control"
PLANTED_TV_STRENGTHS = (0.10, 0.25, 0.40)
PLANTED_ADAPTERS = ("convex", "rff", "krr")
PLANTED_BUDGETS = (2 ** 10, 2 ** 14, 2 ** 18)
PRIMARY_TV_STRENGTH = 0.25
PRIMARY_BUDGET = 2 ** 18
PRIMARY_WORLD_COUNT = 512
DESCRIPTIVE_WORLD_COUNT = 32
DELTA_CALIBRATION_DRAWS = 8
DELTA_CALIBRATION_STEPS = 32
DELTA_CALIBRATION_MAX = 1024.0
SIMULTANEOUS_ALPHA = 0.05
PERSUASIVE_PRIMARY_POINTS = 14
STRONG_BIT_ERROR_THRESHOLD = 0.40
MIN_RESPONSE_GAIN = 0.05
PLANTED_LAW = "adapter_response_basis_rademacher"
INFERENCE_RULE = "world_mean_hoeffding_bonferroni_binary_entropy"
CONFIG_SELECTION_RULE = "capacity_first_rank_precision_regularization"
RECOVERY_RULE = "matched_baseline_basis_projection"


@dataclass(frozen=True)
class PlantedGeometry:
    basis: np.ndarray
    output_dimension: int
    flattened: bool

    @property
    def bit_shape(self) -> Tuple[int, ...]:
        if self.flattened:
            return (int(self.basis.shape[1]),)
        return (int(self.basis.shape[1]), int(self.output_dimension))

    @property
    def bit_count(self) -> int:
        return int(np.prod(self.bit_shape))

    @property
    def digest(self) -> str:
        payload = np.ascontiguousarray(self.basis, dtype="<f8").tobytes()
        return hashlib.sha256(payload).hexdigest()

    def perturbation(self, bits: np.ndarray) -> np.ndarray:
        values = self.basis @ bits
        if self.flattened:
            return values.reshape(-1, self.output_dimension)
        return values

    def scores(self, response_difference: np.ndarray) -> np.ndarray:
        values = np.asarray(response_difference, dtype=np.float64)
        if self.flattened:
            values = values.reshape(-1)
        return self.basis.T @ values


def world_count(tv_strength: float, budget: int) -> int:
    return (
        PRIMARY_WORLD_COUNT
        if tv_strength == PRIMARY_TV_STRENGTH and budget == PRIMARY_BUDGET
        else DESCRIPTIVE_WORLD_COUNT
    )


def commitment_scope(
    cell_id: str,
    adapter: str,
    tv_strength: float,
    budget: int,
    draw_id: int,
) -> str:
    # The law scope and adapter prevent cross-scope opening reuse and identify
    # the adapter-specific public response-basis law.
    return "planted:%s:%s:%s:%s:%s:%d" % (
        PLANTED_SCOPE_ID,
        cell_id,
        adapter,
        tv_strength,
        budget,
        draw_id,
    )


def _codec_precision_bits(codec: str) -> int:
    if codec.startswith("q"):
        return int(codec[1:])
    if codec.startswith("native_fp"):
        return int(codec[len("native_fp") :])
    raise ValueError("unknown planted-state codec: %s" % codec)


def select_capacity_point(
    points: Sequence[OperatingPoint], bit_cap: int
) -> OperatingPoint | None:
    """Select capacity without observing planted calibration or primary worlds."""

    feasible = [point for point in points if point.encoded.complete_bits <= bit_cap]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda point: (
            int(point.decoded.rank),
            _codec_precision_bits(point.encoded.codec),
            -int(point.decoded.lambda_index),
            int(point.encoded.complete_bits),
            -float(point.validation_error),
            point.encoded.codec,
        ),
    )


def response_geometry(
    point: OperatingPoint, cell: AuditCell
) -> PlantedGeometry:
    adapter = point.decoded.adapter
    output_dimension = int(cell.target_coordinates["fit"].shape[1])
    if adapter == "convex":
        selected = cell.peer_coordinates["fit"][
            :, np.asarray(point.context.peer_indices, dtype=int), :
        ]
        design = selected.transpose(0, 2, 1).reshape(-1, selected.shape[1])
        tangent = design[:, :-1] - design[:, -1:]
        basis, singular_values, _ = np.linalg.svd(tangent, full_matrices=False)
        tolerance = np.finfo(float).eps * max(tangent.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        if rank < 1:
            raise RuntimeError("convex planted response basis has zero rank")
        return PlantedGeometry(basis[:, :rank], output_dimension, True)
    if adapter == "rff":
        features = transform_features(
            point.context, cell.peer_coordinates["fit"]
        )
        mapped = rff_features(point.context, features, point.decoded.rank)
        centered = mapped - mapped.mean(axis=0, keepdims=True)
        basis, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        lam = float(LAMBDA_GRID[point.decoded.lambda_index])
        gains = singular_values ** 2 / (
            singular_values ** 2 + len(centered) * lam
        )
        rank = int(np.sum(gains >= MIN_RESPONSE_GAIN))
        constant = np.ones((len(mapped), 1), dtype=np.float64) / np.sqrt(len(mapped))
        return PlantedGeometry(
            np.column_stack([constant, basis[:, :rank]]),
            output_dimension,
            False,
        )
    if adapter == "krr":
        if point.context.eigvals is None or point.context.eigvecs is None:
            raise RuntimeError("KRR planted response basis is unavailable")
        rank = min(int(point.encoded.rank), len(point.context.eigvals))
        eigenvalues = point.context.eigvals[:rank]
        lam = float(LAMBDA_GRID[point.decoded.lambda_index])
        gains = eigenvalues / (
            eigenvalues + len(point.context.fit_features) * lam
        )
        positive = int(np.sum(gains >= MIN_RESPONSE_GAIN))
        if positive < 1:
            raise RuntimeError("KRR planted response basis has zero rank")
        return PlantedGeometry(
            point.context.eigvecs[:, :positive], output_dimension, False
        )
    raise ValueError("unsupported planted-state adapter: %s" % adapter)


def _calibration_seed(
    cell_id: str, adapter: str, tv_strength: float, draw_id: int
) -> int:
    label = "%s:%s:%s:%s:%d" % (
        PLANTED_SCOPE_ID,
        cell_id,
        adapter,
        tv_strength,
        draw_id,
    )
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _calibration_tv(
    cell: AuditCell,
    adapter: str,
    geometry: PlantedGeometry,
    tv_strength: float,
    delta: float,
) -> float:
    natural = cell.target_coordinates["fit"]
    natural_probability = coordinates_to_probabilities(natural)
    shifts = []
    for draw_id in range(DELTA_CALIBRATION_DRAWS):
        rng = np.random.RandomState(
            _calibration_seed(cell.cell_id, adapter, tv_strength, draw_id)
        )
        bits = rng.choice((-1.0, 1.0), size=geometry.bit_shape)
        planted_probability = coordinates_to_probabilities(
            natural + delta * geometry.perturbation(bits)
        )
        shifts.append(
            float(
                np.mean(
                    total_variation(natural_probability, planted_probability)
                )
            )
        )
    return float(np.mean(shifts))


def calibrate_delta(
    cell: AuditCell,
    adapter: str,
    geometry: PlantedGeometry,
    tv_strength: float,
) -> Tuple[float, float]:
    """Choose a cell-level delta using Y0 before any registered world."""

    if not 0.0 < tv_strength < 1.0:
        raise ValueError("planted semantic-TV strength must lie in (0, 1)")
    lower = 0.0
    upper = 1.0
    upper_tv = _calibration_tv(
        cell, adapter, geometry, tv_strength, upper
    )
    while upper_tv < tv_strength and upper < DELTA_CALIBRATION_MAX:
        lower = upper
        upper = min(2.0 * upper, DELTA_CALIBRATION_MAX)
        upper_tv = _calibration_tv(
            cell, adapter, geometry, tv_strength, upper
        )
    if upper_tv < tv_strength:
        raise RuntimeError(
            "semantic-TV calibration failed for %s at strength %s"
            % (cell.cell_id, tv_strength)
        )
    for _ in range(DELTA_CALIBRATION_STEPS):
        midpoint = 0.5 * (lower + upper)
        if (
            _calibration_tv(
                cell, adapter, geometry, tv_strength, midpoint
            )
            < tv_strength
        ):
            lower = midpoint
        else:
            upper = midpoint
    return float(upper), _calibration_tv(
        cell, adapter, geometry, tv_strength, upper
    )


def _unavailable_summary(
    cell: AuditCell,
    adapter: str,
    tv_strength: float,
    delta: float,
    calibration_tv: float,
    budget: int,
    tier: str,
    simultaneous_family: int,
    geometry_digest: str = "",
) -> Dict[str, object]:
    return {
        "row_kind": "summary",
        "world_law_scope": PLANTED_SCOPE_ID,
        "audit_cell": cell.cell_id,
        "target_id": cell.target_id,
        "task_id": cell.task_id,
        "adapter": adapter,
        "tv_strength": tv_strength,
        "delta": delta,
        "budget_cap": budget,
        "draw_id": -1,
        "commitment_id": "",
        "bit_error": np.nan,
        "raw_y0_bit_error": np.nan,
        "bit_error_upper_bound": np.nan,
        "plugin_information_per_bit": np.nan,
        "information_lower_bound_per_bit": np.nan,
        "natural_perturbation_tv": np.nan,
        "calibration_tv": calibration_tv,
        "complete_coded_bits": np.nan,
        "planted_bit_count": np.nan,
        "bits_per_planted_bit": np.nan,
        "world_count": 0,
        "registered_points": simultaneous_family,
        "simultaneous_alpha": SIMULTANEOUS_ALPHA,
        "inference_rule": INFERENCE_RULE,
        "configuration_rule": CONFIG_SELECTION_RULE,
        "recovery_rule": RECOVERY_RULE,
        "planted_law": PLANTED_LAW,
        "geometry_digest": geometry_digest,
        "inference_tier": tier,
        "codec": "",
        "rank": np.nan,
        "lambda_index": np.nan,
        "bitstream_digest": "",
        "state_path": "",
        "accounting_status": "not_applicable",
        "status": "no_feasible_configuration",
    }


def freeze_planted_geometry(
    cells: Sequence[AuditCell],
    workers: int = 1,
    registered_points: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if workers < 1:
        raise ValueError("planted freeze workers must be at least one")
    if registered_points is None:
        registered_points = len(cells) * len(PLANTED_ADAPTERS)
        if registered_points != 27:
            raise RuntimeError(
                "primary response-geometry family must contain 9 cells x 3 adapters"
            )
    if workers > 1:
        payloads = [(cell, registered_points) for cell in cells]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(payloads))
        ) as executor:
            pieces = list(executor.map(_freeze_planted_cell_worker, payloads))
        result_frames = [piece[0] for piece in pieces]
        oracle_frames = [piece[1] for piece in pieces]
        return (
            _annotate_primary_evidence(
                pd.concat(result_frames, ignore_index=True)
            ),
            pd.concat(oracle_frames, ignore_index=True),
        )

    rows: List[Dict[str, object]] = []
    oracle_rows: List[Dict[str, object]] = []
    simultaneous_family = registered_points

    for cell in cells:
        natural_points: Dict[Tuple[str, int], OperatingPoint | None] = {}
        natural_predictions: Dict[Tuple[str, int], np.ndarray] = {}
        geometries: Dict[Tuple[str, int], PlantedGeometry] = {}
        for adapter in PLANTED_ADAPTERS:
            points = enumerate_operating_points(cell, adapter, (0, 1, 2), False)
            for budget in PLANTED_BUDGETS:
                point = select_capacity_point(points, budget)
                natural_points[(adapter, budget)] = point
                if point is not None:
                    geometries[(adapter, budget)] = response_geometry(point, cell)
                    natural_predictions[(adapter, budget)] = predict_coordinates(
                        point.decoded,
                        point.context,
                        cell.peer_coordinates["fit"],
                    )

        for adapter in PLANTED_ADAPTERS:
            for tv_strength in PLANTED_TV_STRENGTHS:
                for budget in PLANTED_BUDGETS:
                    natural = natural_points[(adapter, budget)]
                    tier = (
                        "simultaneous_primary"
                        if tv_strength == PRIMARY_TV_STRENGTH
                        and budget == PRIMARY_BUDGET
                        else "descriptive"
                    )
                    if natural is None:
                        rows.append(
                            _unavailable_summary(
                                cell,
                                adapter,
                                tv_strength,
                                np.nan,
                                np.nan,
                                budget,
                                tier,
                                simultaneous_family,
                            )
                        )
                        continue

                    geometry = geometries[(adapter, budget)]
                    delta, calibration_tv = calibrate_delta(
                        cell, adapter, geometry, tv_strength
                    )

                    losses: List[float] = []
                    raw_losses: List[float] = []
                    shifts: List[float] = []
                    coded_bits: List[int] = []
                    planted_bit_count = geometry.bit_count
                    baseline = natural_predictions[(adapter, budget)]
                    for draw_id in range(world_count(tv_strength, budget)):
                        scope = commitment_scope(
                            cell.cell_id,
                            adapter,
                            tv_strength,
                            budget,
                            draw_id,
                        )
                        opening = load_world_opening(scope)
                        rng = np.random.RandomState(opening.seed % (2 ** 32))
                        bits = rng.choice(
                            (-1.0, 1.0),
                            size=geometry.bit_shape,
                        )
                        planted_target = (
                            cell.target_coordinates["fit"]
                            + delta * geometry.perturbation(bits)
                        )
                        natural_probability = coordinates_to_probabilities(
                            cell.target_coordinates["fit"]
                        )
                        planted_probability = coordinates_to_probabilities(
                            planted_target
                        )
                        shift = float(
                            np.mean(
                                total_variation(
                                    natural_probability, planted_probability
                                )
                            )
                        )
                        planted_state = fit_adapter(
                            adapter,
                            natural.context,
                            cell.peer_coordinates["fit"],
                            planted_target,
                            natural.state.lambda_index,
                            natural.state.rank,
                        )
                        encoded = encode_state(
                            planted_state,
                            natural.context,
                            natural.encoded.codec,
                            natural.encoded.rank,
                        )
                        if encoded.complete_bits > budget:
                            raise RuntimeError(
                                "planted state exceeds its registered budget"
                            )
                        path = (
                            STATES_DIR
                            / "frozen"
                            / "response_geometry"
                            / cell.cell_id
                            / (encoded.bitstream_digest + ".bin")
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(encoded.bitstream)
                        decoded = decode_state(encoded, natural.context)
                        recovered = predict_coordinates(
                            decoded,
                            natural.context,
                            cell.peer_coordinates["fit"],
                        )
                        recovered_bits = np.where(
                            geometry.scores(recovered - baseline) >= 0.0,
                            1.0,
                            -1.0,
                        )
                        raw_recovered_bits = np.where(
                            geometry.scores(
                                recovered - cell.target_coordinates["fit"]
                            )
                            >= 0.0,
                            1.0,
                            -1.0,
                        )
                        bit_error = float(np.mean(recovered_bits != bits))
                        raw_bit_error = float(
                            np.mean(raw_recovered_bits != bits)
                        )
                        losses.append(bit_error)
                        raw_losses.append(raw_bit_error)
                        shifts.append(shift)
                        coded_bits.append(int(encoded.complete_bits))
                        oracle_rows.append(
                            {
                                "world_law_scope": PLANTED_SCOPE_ID,
                                "audit_cell": cell.cell_id,
                                "adapter": adapter,
                                "tv_strength": tv_strength,
                                "delta": delta,
                                "budget_cap": budget,
                                "draw_id": draw_id,
                                "seed_hex": "%064x" % opening.seed,
                                "planted_bits": np.packbits(bits > 0)
                                .tobytes()
                                .hex(),
                                "commitment_id": opening.commitment.commitment_id,
                                "commitment_value": str(
                                    opening.commitment.value
                                ),
                                "blinding": str(opening.commitment.blinding),
                            }
                        )
                        rows.append(
                            {
                                "row_kind": "world",
                                "world_law_scope": PLANTED_SCOPE_ID,
                                "audit_cell": cell.cell_id,
                                "target_id": cell.target_id,
                                "task_id": cell.task_id,
                                "adapter": adapter,
                                "tv_strength": tv_strength,
                                "delta": delta,
                                "budget_cap": budget,
                                "draw_id": draw_id,
                                "commitment_id": opening.commitment.commitment_id,
                                "bit_error": bit_error,
                                "raw_y0_bit_error": raw_bit_error,
                                "bit_error_upper_bound": np.nan,
                                "plugin_information_per_bit": np.nan,
                                "information_lower_bound_per_bit": np.nan,
                                "natural_perturbation_tv": shift,
                                "calibration_tv": calibration_tv,
                                "complete_coded_bits": encoded.complete_bits,
                                "planted_bit_count": planted_bit_count,
                                "bits_per_planted_bit": encoded.complete_bits
                                / planted_bit_count,
                                "world_count": 1,
                                "registered_points": simultaneous_family,
                                "simultaneous_alpha": SIMULTANEOUS_ALPHA,
                                "inference_rule": INFERENCE_RULE,
                                "configuration_rule": CONFIG_SELECTION_RULE,
                                "recovery_rule": RECOVERY_RULE,
                                "planted_law": PLANTED_LAW,
                                "geometry_digest": geometry.digest,
                                "inference_tier": tier,
                                "codec": encoded.codec,
                                "rank": encoded.rank,
                                "lambda_index": decoded.lambda_index,
                                "bitstream_digest": encoded.bitstream_digest,
                                "state_path": str(path.relative_to(PROJECT_ROOT)),
                                "accounting_status": "not_evaluated",
                                "status": "completed",
                            }
                        )

                    if len(set(coded_bits)) != 1:
                        raise RuntimeError(
                            "a fixed planted configuration changed coded length"
                        )
                    inference = simultaneous_bit_information_summary(
                        losses, simultaneous_family, SIMULTANEOUS_ALPHA
                    )
                    lower_bound = (
                        inference["information_lower_bound_per_bit"]
                        if tier == "simultaneous_primary"
                        else np.nan
                    )
                    bits_per_planted_bit = coded_bits[0] / planted_bit_count
                    accounting_status = (
                        "accounting_interval_diagnostic"
                        if np.isfinite(lower_bound)
                        and lower_bound > min(1.0, bits_per_planted_bit) + 1e-12
                        else "passed"
                    )
                    rows.append(
                        {
                            "row_kind": "summary",
                            "world_law_scope": PLANTED_SCOPE_ID,
                            "audit_cell": cell.cell_id,
                            "target_id": cell.target_id,
                            "task_id": cell.task_id,
                            "adapter": adapter,
                            "tv_strength": tv_strength,
                            "delta": delta,
                            "budget_cap": budget,
                            "draw_id": -1,
                            "commitment_id": "",
                            "bit_error": inference["bit_error"],
                            "raw_y0_bit_error": float(np.mean(raw_losses)),
                            "bit_error_upper_bound": inference[
                                "bit_error_upper_bound"
                            ],
                            "plugin_information_per_bit": inference[
                                "plugin_information_per_bit"
                            ],
                            "information_lower_bound_per_bit": lower_bound,
                            "natural_perturbation_tv": float(np.mean(shifts)),
                            "calibration_tv": calibration_tv,
                            "complete_coded_bits": coded_bits[0],
                            "planted_bit_count": planted_bit_count,
                            "bits_per_planted_bit": bits_per_planted_bit,
                            "world_count": len(losses),
                            "registered_points": simultaneous_family,
                            "simultaneous_alpha": SIMULTANEOUS_ALPHA,
                            "inference_rule": INFERENCE_RULE,
                            "configuration_rule": CONFIG_SELECTION_RULE,
                            "recovery_rule": RECOVERY_RULE,
                            "planted_law": PLANTED_LAW,
                            "geometry_digest": geometry.digest,
                            "inference_tier": tier,
                            "codec": natural.encoded.codec,
                            "rank": natural.encoded.rank,
                            "lambda_index": natural.decoded.lambda_index,
                            "bitstream_digest": natural.encoded.bitstream_digest,
                            "state_path": "",
                            "accounting_status": accounting_status,
                            "status": "completed",
                        }
                    )
    return _annotate_primary_evidence(pd.DataFrame(rows)), pd.DataFrame(oracle_rows)


def _annotate_primary_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["minimum_success"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result["strong_point"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result["primary_family_pass_count"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result["primary_family_success"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    primary = (
        (result["row_kind"] == "summary")
        & (result["inference_tier"] == "simultaneous_primary")
    )
    minimum = primary & (result["bit_error_upper_bound"] < 0.5) & (
        result["information_lower_bound_per_bit"] > 0.0
    )
    strong = minimum & (result["bit_error"] <= STRONG_BIT_ERROR_THRESHOLD)
    pass_count = int(minimum.sum())
    result.loc[primary, "minimum_success"] = minimum[primary].astype(bool)
    result.loc[primary, "strong_point"] = strong[primary].astype(bool)
    result.loc[primary, "primary_family_pass_count"] = pass_count
    result.loc[primary, "primary_family_success"] = (
        pass_count >= PERSUASIVE_PRIMARY_POINTS
    )
    return result


def _freeze_planted_cell_worker(
    payload: Tuple[AuditCell, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cell, registered_points = payload
    return freeze_planted_geometry(
        [cell], workers=1, registered_points=registered_points
    )
