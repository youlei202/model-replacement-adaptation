"""Cheap calibration is a first-class member of every pair's common catalog."""
from __future__ import annotations

import torch
from .models import FrozenAnchor, TrainableCalibration


def initial_calibrations(weights: torch.Tensor):
    """Identical starting menu for every pair; selection belongs to inner data."""
    positive = weights.clamp_min(0)
    if positive.sum() <= 0:
        positive = torch.ones_like(positive)
    return {
        "kl_logit_mixture": TrainableCalibration(positive, "logit", temperature=True),
        "kl_probability_mixture": TrainableCalibration(positive, "probability", temperature=False),
    }


def direct_anchor(peer_position: int, peer_count: int, *, device=None, dtype=torch.float32):
    if not 0 <= peer_position < peer_count:
        raise ValueError("Selected peer position outside interface")
    weights = torch.zeros(peer_count, device=device, dtype=dtype)
    weights[peer_position] = 1
    return FrozenAnchor(weights, "logit")


__all__ = ["FrozenAnchor", "TrainableCalibration", "initial_calibrations", "direct_anchor"]
