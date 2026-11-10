"""Direct bounded optimization of tiny calibrators, shared by every peer pair.

Direct coefficients avoid the vanishing mixture-weight gradient of a softmax
initialized at a single peer. Both target-dependent coefficients are charged.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .losses import objective
from .models import FrozenAnchor


def optimize_coefficients(data, initial, *, kind='logit', loss='kl', single=None,
                          maxiter=60, device='cpu', batch_size=128):
    if data.role not in ('fit', 'inner_train'):
        raise ValueError('Calibrators use fit labels only')
    initial = np.asarray(initial, dtype=np.float64)
    if kind == 'probability':
        x0 = np.array([initial[0] / initial.sum()])
        bounds = [(0., 1.)]
    elif single is not None:
        x0 = np.array([max(initial[single], .1)])
        bounds = [(0.02, 4.)]
    else:
        x0 = initial.copy()
        bounds = [(0., 4.), (0., 4.)]
    history = []
    started = time.perf_counter()

    def calculate(values):
        parameters = torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)
        total = 0.
        gradient = np.zeros_like(values)
        for offset in range(0, len(data), batch_size):
            indices = np.arange(offset, min(offset + batch_size, len(data)))
            peers, _, target = data.batch(indices, device)
            if kind == 'probability':
                probabilities = torch.softmax(peers, -1)
                q = parameters[0] * probabilities[:, 0] + (1 - parameters[0]) * probabilities[:, 1]
                log_q = q.clamp_min(torch.finfo(q.dtype).tiny).log()
            elif single is not None:
                log_q = torch.log_softmax(parameters[0] * peers[:, single], -1)
            else:
                log_q = torch.log_softmax(parameters[0] * peers[:, 0] + parameters[1] * peers[:, 1], -1)
            value = objective(target, log_q, loss)
            value.backward()
            proportion = len(indices) / len(data)
            total += float(value.detach()) * proportion
            gradient += parameters.grad.detach().cpu().numpy().astype(float) * proportion
            parameters.grad = None
        history.append({'loss': total, 'coefficients': values.tolist(), 'gradient': gradient.tolist()})
        return total, gradient

    result = minimize(calculate, x0, jac=True, bounds=bounds, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30})
    if kind == 'probability':
        weights = np.array([result.x[0], 1 - result.x[0]], dtype=np.float32)
    elif single is not None:
        weights = np.zeros(2, np.float32); weights[single] = result.x[0]
    else:
        weights = result.x.astype(np.float32)
    return FrozenAnchor(torch.tensor(weights, device=device), kind), {
        'kind': kind, 'objective': loss, 'single_peer_position': single,
        'initial': initial.tolist(), 'weights': weights.tolist(), 'success': bool(result.success),
        'message': str(result.message), 'iterations': int(result.nit), 'evaluations': len(history),
        'train_objective': float(result.fun), 'seconds': time.perf_counter() - started,
        'history': history, 'selection_labels': 'inner_train_only',
        'note': 'All fit endpoints retained even when optimizer reports numerical line-search termination.',
    }
