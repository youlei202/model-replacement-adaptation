"""Exact full-support metrics, computed a batch of positions at a time.

Top-k tail bounds are provided only for diagnostics. Renormalizing retained
coordinates would change the target distribution and is intentionally absent.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


def exact_metrics(target_logits: np.ndarray, replacement_logits: np.ndarray,
                  batch_positions: int = 32) -> dict[str, np.ndarray]:
    t, p = np.asarray(target_logits), np.asarray(replacement_logits)
    if t.ndim != 2 or t.shape != p.shape or min(t.shape) < 1:
        raise ValueError('Matched nonempty positions-by-full-vocabulary arrays required')
    if batch_positions < 1:
        raise ValueError('Positive batch size required')
    result = {k: np.empty(len(t), dtype=float) for k in ('tv', 'kl', 'argmax_disagreement')}
    for s in range(0, len(t), batch_positions):
        a = np.asarray(t[s:s + batch_positions], dtype=np.float64)
        b = np.asarray(p[s:s + batch_positions], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite logits; resolve shared support/masking explicitly')
        la = a - logsumexp(a, axis=1, keepdims=True)
        lb = b - logsumexp(b, axis=1, keepdims=True)
        pa, pb = np.exp(la), np.exp(lb)
        result['tv'][s:s + len(a)] = .5 * np.abs(pa - pb).sum(1)
        result['kl'][s:s + len(a)] = np.maximum(0., (pa * (la - lb)).sum(1))
        result['argmax_disagreement'][s:s + len(a)] = a.argmax(1) != b.argmax(1)
    return result


def tv_tail_bounds(target_kept: np.ndarray, prediction_kept: np.ndarray,
                   target_tail: np.ndarray, prediction_tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kept IDs must be the SAME union support; inputs retain full normalization.
    TV(full) is in [partial + .5|p_tail-q_tail|, partial + .5(p_tail+q_tail)].
    """
    a, b = np.asarray(target_kept, float), np.asarray(prediction_kept, float)
    x, y = np.asarray(target_tail, float), np.asarray(prediction_tail, float)
    if a.ndim != 2 or a.shape != b.shape or x.shape != (len(a),) or y.shape != x.shape:
        raise ValueError('Incompatible kept-support and tail shapes')
    if any(not np.isfinite(z).all() or (z < 0).any() for z in (a, b, x, y)):
        raise ValueError('Finite nonnegative probabilities required')
    if not np.allclose(a.sum(1) + x, 1, atol=1e-8, rtol=0) or not np.allclose(b.sum(1) + y, 1, atol=1e-8, rtol=0):
        raise ValueError('Inputs must retain original full-distribution mass')
    partial = .5 * np.abs(a - b).sum(1)
    return partial + .5 * np.abs(x - y), partial + .5 * (x + y)
