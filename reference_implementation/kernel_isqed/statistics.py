from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 0.0), 1.0)
    if value in (0.0, 1.0):
        return 0.0
    return -value * math.log(value, 2) - (1.0 - value) * math.log(1.0 - value, 2)


def fano_lower_bound(world_count: int, decoding_error: float) -> float:
    entropy = math.log(world_count, 2)
    bound = entropy - binary_entropy(decoding_error) - decoding_error * math.log(world_count - 1, 2)
    return max(0.0, float(bound))


def truth_table_rate_distortion(size: int, distortion: float) -> float:
    return float(size) * (1.0 - binary_entropy(distortion))


def simultaneous_bit_information_lower_bound(
    losses: Iterable[float], registered_points: int, alpha: float = 0.05,
) -> float:
    return simultaneous_bit_information_summary(
        losses, registered_points, alpha
    )["information_lower_bound_per_bit"]


def simultaneous_bit_information_summary(
    losses: Iterable[float], registered_points: int, alpha: float = 0.05,
) -> dict[str, float]:
    values = np.asarray(list(losses), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("simultaneous bit-information inference requires world losses")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("world losses must be finite values in [0, 1]")
    if registered_points < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid simultaneous-inference family")
    radius = math.sqrt(math.log(registered_points / alpha) / (2.0 * len(values)))
    upper = min(1.0, float(values.mean()) + radius)
    lower_bound = 1.0 - binary_entropy(upper) if upper < 0.5 else 0.0
    plugin = 1.0 - binary_entropy(min(float(values.mean()), 0.5))
    return {
        "bit_error": float(values.mean()),
        "bit_error_upper_bound": upper,
        "hoeffding_radius": radius,
        "plugin_information_per_bit": plugin,
        "information_lower_bound_per_bit": lower_bound,
        "registered_points": float(registered_points),
        "alpha": float(alpha),
        "world_count": float(len(values)),
    }
