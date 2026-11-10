"""Validation-only operating-point selection. Test data cannot define the frontier."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class ProfiledCandidate:
    system_id: str
    validation_loss: float
    measured_cost: float
    actual_state_bytes: int
    cost_scope: str  # complete_pipeline or adapter_microbenchmark
    quality_gap_upper_validation: float | None = None

    def __post_init__(self) -> None:
        if (not self.system_id or not isfinite(self.validation_loss) or self.validation_loss < 0
                or not isfinite(self.measured_cost) or self.measured_cost <= 0
                or self.actual_state_bytes < 0):
            raise ValueError('Invalid candidate/profile')
        if self.cost_scope not in ('complete_pipeline', 'adapter_microbenchmark'):
            raise ValueError('Cost measurement scope must be explicit')
        if self.quality_gap_upper_validation is not None and not isfinite(self.quality_gap_upper_validation):
            raise ValueError('Nonfinite validation quality bound')


def select_min_cost(candidates: Sequence[ProfiledCandidate], *,
                    quality_threshold: float | None = None,
                    noninferiority_margin: float | None = None) -> ProfiledCandidate:
    if (quality_threshold is None) == (noninferiority_margin is None):
        raise ValueError('Choose either an absolute threshold or a predeclared NI rule')
    feasible = []
    for c in candidates:
        if c.cost_scope != 'complete_pipeline':
            raise ValueError('Adapter-only timings cannot define end-to-end savings')
        if quality_threshold is not None and c.validation_loss <= quality_threshold:
            feasible.append(c)
        if noninferiority_margin is not None:
            if c.quality_gap_upper_validation is None:
                raise ValueError('Need validation quality bound for NI selection')
            if c.quality_gap_upper_validation <= noninferiority_margin:
                feasible.append(c)
    if not feasible:
        raise ValueError('No eligible system; do not loosen threshold using test results')
    return min(feasible, key=lambda c: (c.measured_cost, c.validation_loss, c.actual_state_bytes, c.system_id))


def nondominated(candidates: Sequence[ProfiledCandidate]) -> list[ProfiledCandidate]:
    out = []
    for c in candidates:
        if not any(d.validation_loss <= c.validation_loss and d.measured_cost <= c.measured_cost
                   and (d.validation_loss < c.validation_loss or d.measured_cost < c.measured_cost)
                   for d in candidates):
            out.append(c)
    return sorted(out, key=lambda c: (c.measured_cost, c.validation_loss, c.system_id))
