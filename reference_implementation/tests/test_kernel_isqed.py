from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.utils import tag_run_mode
from experiments.exp13_peer_roster_replication import expected_run_keys as roster_keys
from experiments.exp14_target_role_replication import expected_run_keys as role_keys
from kernel_isqed.adapters import fit_adapter, predict_coordinates
from kernel_isqed.audit import AuditCell, get_cells
from kernel_isqed.codec import decode_state, encode_state
from kernel_isqed.contracts import (
    CROSS_ROLE_PEERS,
    PEERS,
    ROLE_ROTATION_TARGETS,
)
from kernel_isqed.response import coordinates_to_probabilities, helmert_matrix, total_variation
from kernel_isqed.planted import (
    PlantedGeometry,
    calibrate_delta,
    response_geometry,
    select_capacity_point,
)
from kernel_isqed.deployment import OperatingPoint
from kernel_isqed.statistics import simultaneous_bit_information_summary


def test_peer_roster_replication_key_grid_is_fixed() -> None:
    grid = tuple(
        ("target_%d__task_%d" % (target, task), "target_%d" % target, "task_%d" % task)
        for target in range(3)
        for task in range(3)
    )
    rows = roster_keys(grid)
    assert len(rows) == 324
    counts = pd.DataFrame(rows).groupby(["roster_id", "analysis"]).size().to_dict()
    assert set(counts.values()) == {27, 54, 81}


def test_target_role_replication_key_grid_is_fixed() -> None:
    grid = tuple(
        ("secondary_%d__task_%d" % (target, task), "secondary_%d" % target, "task_%d" % task)
        for target in range(3)
        for task in range(3)
    )
    rows = role_keys(grid)
    assert len(rows) == 324
    counts = pd.DataFrame(rows).groupby(["roster_id", "analysis"]).size().to_dict()
    assert set(counts.values()) == {27, 54, 81}


def test_target_role_rosters_are_disjoint() -> None:
    targets = {model.model_id for model in ROLE_ROTATION_TARGETS}
    assert targets.isdisjoint(model.model_id for model in PEERS)
    assert targets.isdisjoint(model.model_id for model in CROSS_ROLE_PEERS)
    assert {model.model_id for model in PEERS}.isdisjoint(
        model.model_id for model in CROSS_ROLE_PEERS
    )


def audit_cell_fixture(
    fit_size: int = 24, validation_size: int = 12, iid_size: int = 12, ood_size: int = 8,
):
    sizes = {
        "fit": fit_size,
        "validation": validation_size,
        "iid_test": iid_size,
        "ood_test": ood_size,
    }
    rng = np.random.RandomState(19)
    item_ids = {}
    peers = {}
    targets = {}
    coefficient = rng.normal(size=(3, 1))
    for split, size in sizes.items():
        item_ids[split] = np.asarray(["%s_%03d" % (split, index) for index in range(size)])
        peers[split] = rng.normal(size=(size, 3, 1))
        targets[split] = np.einsum("npd,pd->nd", peers[split], coefficient)
    return AuditCell("fixture/target", "rewardbench", item_ids, peers, targets)


def test_helmert_contract() -> None:
    for option_count in (2, 4, 10):
        matrix = helmert_matrix(option_count)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(option_count - 1), atol=1e-12)
        np.testing.assert_allclose(matrix @ np.ones(option_count), 0.0, atol=1e-12)


def test_probability_round_trip_shape() -> None:
    rng = np.random.RandomState(7)
    coordinates = rng.normal(size=(8, 9))
    probabilities = coordinates_to_probabilities(coordinates)
    assert probabilities.shape == (8, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_fixed_peer_context_across_targets() -> None:
    reference = audit_cell_fixture()
    cells = [
        reference,
        AuditCell(
            "fixture/second",
            reference.task_id,
            reference.item_ids,
            reference.peer_coordinates,
            reference.target_coordinates,
        ),
    ]
    for split in ("fit", "validation", "iid_test", "ood_test"):
        for cell in cells[1:]:
            np.testing.assert_allclose(cell.peer_coordinates[split], cells[0].peer_coordinates[split])


def test_state_codecs_decode_in_clean_contract() -> None:
    cell = audit_cell_fixture()
    context = cell.context((0, 1, 2))
    settings = {
        "convex": (0, 0),
        "linear": (8, 0),
        "rff": (8, 16),
        "krr": (8, 0),
    }
    for adapter, (lambda_index, rank) in settings.items():
        state = fit_adapter(
            adapter, context, cell.peer_coordinates["fit"], cell.target_coordinates["fit"], lambda_index, rank,
        )
        codec_rank = len(context.fit_features) if adapter == "krr" else state.rank
        for codec in ("q4", "native_fp32"):
            encoded = encode_state(state, context, codec, codec_rank)
            decoded = decode_state(encoded, context)
            prediction = predict_coordinates(decoded, context, cell.peer_coordinates["iid_test"])
            assert np.isfinite(prediction).all()
            assert encoded.complete_bits + encoded.byte_padding_bits == len(encoded.bitstream) * 8
            if encoded.byte_padding_bits:
                assert encoded.bitstream[-1] & ((1 << encoded.byte_padding_bits) - 1) == 0


def test_clone_total_variation_is_zero() -> None:
    cell = audit_cell_fixture()
    probabilities = coordinates_to_probabilities(cell.peer_coordinates["iid_test"][:, 0, :])
    np.testing.assert_array_equal(total_variation(probabilities, probabilities), 0.0)


def test_missing_response_cache_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_cells(tmp_path / "missing_real_responses.parquet")


def test_fast_run_is_not_tagged_as_full() -> None:
    frame = pd.DataFrame([{"status": "completed"}])
    assert tag_run_mode(frame, fast=True)["run_mode"].iloc[0] == "fast"


def test_planted_delta_is_calibrated_in_semantic_tv() -> None:
    cell = audit_cell_fixture()
    item_count, output_dimension = cell.target_coordinates["fit"].shape
    basis = np.ones((item_count, 1), dtype=np.float64) / np.sqrt(item_count)
    geometry = PlantedGeometry(basis, output_dimension, False)
    delta, observed = calibrate_delta(cell, "rff", geometry, 0.25)
    assert delta > 0.0
    assert abs(observed - 0.25) < 1e-6


def test_planted_capacity_selection_prefers_rank_before_natural_error() -> None:
    class Point:
        def __init__(self, rank, codec, bits, lambda_index, error):
            self.decoded = type(
                "Decoded", (), {"rank": rank, "lambda_index": lambda_index}
            )()
            self.encoded = type(
                "Encoded", (), {"codec": codec, "complete_bits": bits}
            )()
            self.validation_error = error

    low_error = Point(16, "native_fp64", 900, 0, 0.01)
    high_capacity = Point(32, "q8", 950, 0, 0.20)
    assert select_capacity_point([low_error, high_capacity], 1024) is high_capacity


def test_simultaneous_bit_information_reports_upper_error() -> None:
    summary = simultaneous_bit_information_summary([0.25] * 512, 27)
    assert summary["bit_error"] == 0.25
    assert 0.25 < summary["bit_error_upper_bound"] < 0.5
    assert summary["information_lower_bound_per_bit"] > 0.0


def test_planted_response_basis_bits_survive_fit_and_codec() -> None:
    original = audit_cell_fixture()
    targets = {
        split: peers.mean(axis=1)
        for split, peers in original.peer_coordinates.items()
    }
    cell = AuditCell(
        original.target_id,
        original.task_id,
        original.item_ids,
        original.peer_coordinates,
        targets,
    )
    context = cell.context((0, 1, 2))
    rng = np.random.RandomState(23)
    for adapter, rank in (("convex", 0), ("rff", 16), ("krr", 0)):
        state = fit_adapter(
            adapter,
            context,
            cell.peer_coordinates["fit"],
            cell.target_coordinates["fit"],
            0,
            rank,
        )
        codec_rank = len(context.fit_features) if adapter == "krr" else state.rank
        encoded = encode_state(state, context, "native_fp64", codec_rank)
        decoded = decode_state(encoded, context)
        point = OperatingPoint(state, context, encoded, decoded, 0.0)
        geometry = response_geometry(point, cell)
        baseline = predict_coordinates(
            decoded, context, cell.peer_coordinates["fit"]
        )
        bits = rng.choice((-1.0, 1.0), size=geometry.bit_shape)
        planted = cell.target_coordinates["fit"] + 0.01 * geometry.perturbation(bits)
        planted_state = fit_adapter(
            adapter,
            context,
            cell.peer_coordinates["fit"],
            planted,
            state.lambda_index,
            state.rank,
        )
        planted_encoded = encode_state(
            planted_state, context, "native_fp64", codec_rank
        )
        planted_decoded = decode_state(planted_encoded, context)
        recovered = predict_coordinates(
            planted_decoded, context, cell.peer_coordinates["fit"]
        )
        recovered_bits = np.where(
            geometry.scores(recovered - baseline) >= 0.0, 1.0, -1.0
        )
        np.testing.assert_array_equal(recovered_bits, bits)
