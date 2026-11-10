from __future__ import annotations

import numpy as np

from isqed.core import Intervention, ModelUnit


class LinearStructuralModel(ModelUnit):
    """Linear structural model with optional additive Gaussian noise."""

    def __init__(self, dim, beta=None, noise_std=0.0):
        super().__init__(name="LinearSynthetic")
        self.dim = dim
        if beta is None:
            self.beta = np.random.randn(dim)
            self.beta /= np.linalg.norm(self.beta)
        else:
            self.beta = np.asarray(beta)
        self.noise_std = noise_std

    def _forward(self, input_vec):
        clean_y = input_vec @ self.beta
        noise = np.random.randn(*clean_y.shape) * self.noise_std
        return clean_y + noise


class NoiseIntervention(Intervention):
    """Scale a synthetic input by the intervention strength."""

    def apply(self, x, theta):
        return x * theta
