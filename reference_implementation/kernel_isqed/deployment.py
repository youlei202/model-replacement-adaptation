from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .adapters import (
    AdapterContext,
    AdapterState,
    configuration_grid,
    fit_adapter,
    mean_tv,
    predict_coordinates,
)
from .audit import AuditCell
from .codec import EncodedState, decode_state, encode_state
from .contracts import LAMBDA_GRID, QUANTIZED_BITS, SPECTRAL_RANKS


@dataclass
class OperatingPoint:
    state: AdapterState
    context: AdapterContext
    encoded: EncodedState
    decoded: AdapterState
    validation_error: float


def codec_names(fast: bool = False) -> Tuple[str, ...]:
    if fast:
        return ("q2", "q8", "native_fp32")
    quantized = tuple("q%d" % value for value in QUANTIZED_BITS)
    return quantized + ("native_fp16", "native_fp32", "native_fp64")


def configuration_points(adapter: str, fast: bool = False) -> List[Tuple[int, int]]:
    grid = configuration_grid(adapter)
    if not fast or len(grid) <= 4:
        return grid
    indices = sorted(set((0, len(grid) // 2, len(grid) - 1)))
    return [grid[index] for index in indices]


def codec_ranks(state: AdapterState, context: AdapterContext, fast: bool = False,) -> Tuple[int, ...]:
    if state.adapter == "krr":
        size = len(context.fit_features)
        values = [rank for rank in SPECTRAL_RANKS if rank < size] + [size]
        if fast and len(values) > 2:
            values = [values[0], values[-1]]
        return tuple(values)
    if state.adapter in ("linear", "rff", "mlp"):
        return (int(state.rank),)
    return (0,)


def enumerate_operating_points(
    cell: AuditCell, adapter: str, peer_indices: Sequence[int], fast: bool = False,
) -> List[OperatingPoint]:
    context = cell.context(peer_indices)
    points = []
    for lambda_index, rank in configuration_points(adapter, fast):
        state = fit_adapter(
            adapter, context, cell.peer_coordinates["fit"], cell.target_coordinates["fit"], lambda_index, rank,
        )
        for codec_name in codec_names(fast):
            for codec_rank in codec_ranks(state, context, fast):
                try:
                    encoded = encode_state(state, context, codec_name, codec_rank)
                    decoded = decode_state(encoded, context)
                    prediction = predict_coordinates(decoded, context, cell.peer_coordinates["validation"],)
                    if not np.isfinite(prediction).all():
                        raise ValueError("decoded state produced non-finite predictions")
                    validation_error = mean_tv(cell.target_coordinates["validation"], prediction,)
                except ValueError:
                    continue
                points.append(OperatingPoint(state, context, encoded, decoded, validation_error,))
    return points


def select_under_budget(points: Sequence[OperatingPoint], bit_cap: int,) -> Optional[OperatingPoint]:
    feasible = [point for point in points if point.encoded.complete_bits <= bit_cap]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda point: (
            point.validation_error,
            point.encoded.complete_bits,
            point.decoded.rank,
            -float(LAMBDA_GRID[point.decoded.lambda_index]),
            point.encoded.codec,
        ),
    )


def select_budget_point(
    cell: AuditCell, adapter: str, peer_indices: Sequence[int], bit_cap: int, fast: bool = False,
) -> Optional[OperatingPoint]:
    points = enumerate_operating_points(cell, adapter, peer_indices, fast)
    return select_under_budget(points, bit_cap)
