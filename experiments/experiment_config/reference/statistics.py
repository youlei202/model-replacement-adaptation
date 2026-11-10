"""Paired cluster-level utilities. Adapt the population weighting explicitly."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import NormalDist
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    independent_clusters: int
    nominal_familywise_alpha: float
    comparisons: int
    weighting: str


def paired_cluster_interval(difference: Sequence[float], clusters: Sequence[str], *,
                            draws: int = 10000, seed: int = 81724032,
                            alpha: float = .05, comparisons: int = 1,
                            weighting: str = 'cluster_equal') -> Interval:
    d = np.asarray(difference, dtype=float)
    g = np.asarray(clusters, dtype=str)
    if d.ndim != 1 or g.shape != d.shape or not len(d) or not np.isfinite(d).all():
        raise ValueError('Finite aligned one-dimensional paired differences required')
    if draws < 100 or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError('Invalid inference settings')
    unique, ix = np.unique(g, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('At least two independent clusters required')
    sums = np.bincount(ix, weights=d)
    counts = np.bincount(ix)
    means = sums / counts
    if weighting not in ('cluster_equal', 'item_equal'):
        raise ValueError('Declare cluster_equal or item_equal weighting')
    estimate = float(means.mean() if weighting == 'cluster_equal' else d.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for start in range(0, draws, 128):
        take = rng.integers(len(unique), size=(min(128, draws - start), len(unique)))
        values[start:start + len(take)] = (
            means[take].mean(1) if weighting == 'cluster_equal'
            else sums[take].sum(1) / counts[take].sum(1))
    tail = alpha / (2 * comparisons)
    lo, hi = np.quantile(values, [tail, 1 - tail])
    return Interval(estimate, float(lo), float(hi), len(unique), alpha,
                    comparisons, weighting)


def average_budget_differences(matrix: np.ndarray, weights: Sequence[float] | None = None) -> np.ndarray:
    """Rows must be the SAME aligned queries; columns are registered budgets.
    Bootstrap the returned vector, not a flattened query x budget matrix.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Nonempty finite query-by-budget matrix required')
    w = np.ones(a.shape[1]) / a.shape[1] if weights is None else np.asarray(weights, float)
    if w.shape != (a.shape[1],) or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid budget weights')
    return a @ w


def paired_power_n(sd: float, effect: float, *, alpha: float = .05,
                   comparisons: int = 4, power: float = .90) -> int:
    """Normal-approximation planning using independent paired cluster means.
    Not an observed-power calculation or a finite-sample guarantee.
    """
    if not isfinite(sd) or sd <= 0 or not isfinite(effect) or effect <= 0:
        raise ValueError('Positive finite SD and detectable effect required')
    if not 0 < alpha < 1 or not .5 < power < 1 or comparisons < 1:
        raise ValueError('Invalid power-design settings')
    z_alpha = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    z_power = NormalDist().inv_cdf(power)
    return max(2, ceil(((z_alpha + z_power) * sd / effect) ** 2))
