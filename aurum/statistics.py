"""Paired document inference, with simultaneous contrasts across frozen targets.

The centered Gaussian multiplier max-t interval is asymptotic. Whole documents
are the independent observations; repeated budgets are averaged within them.
"""
import numpy as np


def simultaneous_intervals(values, strata, *, alpha=.05, draws=10000, seed=20261110):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    labels = np.asarray(strata)
    if len(labels) != len(x) or not np.isfinite(x).all():
        raise ValueError('Finite aligned document observations required')
    mean = np.zeros(x.shape[1]); variance = np.zeros(x.shape[1]); influence = np.zeros_like(x)
    for label in np.unique(labels):
        mask = labels == label; n = mask.sum()
        if n < 2:
            raise ValueError('At least two documents per source required')
        weight = n / len(x)
        sample = x[mask]; average = sample.mean(0)
        mean += weight * average
        variance += weight**2 * sample.var(0, ddof=1) / n
        influence[mask] = weight * (sample - average) / np.sqrt(n * (n - 1))
    se = np.sqrt(variance); nonzero = se > 1e-14
    rng = np.random.default_rng(seed); maxima = []
    for offset in range(0, draws, 256):
        z = rng.standard_normal((min(256, draws - offset), len(x))) @ influence
        maxima.extend(np.max(np.abs(z[:, nonzero] / se[nonzero]), axis=1) if nonzero.any() else np.zeros(len(z)))
    critical = float(np.quantile(maxima, 1 - alpha))
    return {'mean': mean, 'se': se, 'lower': mean - critical * se, 'upper': mean + critical * se,
            'critical': critical, 'alpha': alpha, 'draws': draws, 'seed': seed,
            'method': 'source-stratified centered Gaussian multiplier max-|t|; asymptotic simultaneous intervals',
            'family_size': x.shape[1], 'nondegenerate_contrasts': int(nonzero.sum()), 'documents': len(x)}
