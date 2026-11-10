"""AURUM target-free predictors with a literally nested cheap anchor.

The basis and feature transformation are selected-peer-only public context. A
target-trained transformation is not public and must not be passed as such.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class FrozenAnchor(nn.Module):
    """Weighted logits or an arithmetic mixture of full peer probabilities."""

    def __init__(self, weights: Tensor, kind: str = "logit") -> None:
        super().__init__()
        weights = torch.as_tensor(weights)
        if not weights.is_floating_point():
            weights = weights.float()
        if weights.ndim != 1 or len(weights) == 0 or not torch.isfinite(weights).all():
            raise ValueError("Finite nonempty anchor weight vector required")
        if kind not in ("logit", "probability"):
            raise ValueError("Unknown anchor kind")
        if kind == "probability":
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Probability weights must be nonnegative with positive sum")
            weights = weights / weights.sum()
        self.kind = kind
        self.register_buffer("weights", weights.detach().clone())

    def logits(self, peers: Tensor) -> Tensor:
        if peers.ndim != 3 or peers.shape[1] != len(self.weights):
            raise ValueError("Expected peers[items, selected_peers, vocabulary]")
        if self.kind == "logit":
            # Matches the ordered scalar-weighted legacy expression for q=2.
            return sum(self.weights[j] * peers[:, j] for j in range(len(self.weights)))
        log_weights = torch.where(self.weights > 0, self.weights.log(), -torch.inf)
        return torch.logsumexp(torch.log_softmax(peers, -1) + log_weights[None, :, None], 1)

    def forward(self, peers: Tensor, features: Tensor | None = None) -> Tensor:
        # Both an anchor and its residual child normalize these SAME scores.
        # Thus adding an exact zero gives bitwise equal log probabilities,
        # without a data-dependent branch that would kill the initial gradient.
        return torch.log_softmax(self.logits(peers), -1)


class TrainableCalibration(nn.Module):
    """Simplex weights plus an optional positive logit inverse temperature.

    Temperature is absorbed into exported logit coefficients, so it is not
    charged twice. The initial sum is retained, unlike the reference default.
    """

    def __init__(self, weights: Tensor, kind: str = "logit", temperature: bool = True) -> None:
        super().__init__()
        weights = torch.as_tensor(weights)
        if not weights.is_floating_point():
            weights = weights.float()
        if (weights.ndim != 1 or len(weights) == 0 or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Nonnegative finite positive-sum calibration weights required")
        if kind not in ("logit", "probability"):
            raise ValueError("Unknown calibration kind")
        self.kind = kind
        self.mix_logits = nn.Parameter((weights / weights.sum()).clamp_min(1e-8).log())
        initial_temperature = weights.sum().log() if temperature and kind == "logit" else weights.new_zeros(())
        self.log_inverse_temperature = nn.Parameter(initial_temperature,
                                                    requires_grad=temperature and kind == "logit")

    def coefficients(self) -> Tensor:
        weights = torch.softmax(self.mix_logits, 0)
        if self.kind == "logit":
            weights = weights * self.log_inverse_temperature.clamp(-4, 4).exp()
        return weights

    def logits(self, peers: Tensor) -> Tensor:
        weights = self.coefficients()
        if peers.ndim != 3 or peers.shape[1] != len(weights):
            raise ValueError("Expected [items, selected_peers, vocabulary]")
        if self.kind == "logit":
            return sum(weights[j] * peers[:, j] for j in range(len(weights)))
        return torch.logsumexp(torch.log_softmax(peers, -1) + weights.log()[None, :, None], 1)

    def forward(self, peers: Tensor, features: Tensor | None = None) -> Tensor:
        return torch.log_softmax(self.logits(peers), -1)

    def frozen(self) -> FrozenAnchor:
        return FrozenAnchor(self.coefficients().detach(), self.kind)


class AnchoredResidual(nn.Module):
    """Linear or factored coefficient map into a fixed public output basis."""

    def __init__(self, anchor: FrozenAnchor, basis: Tensor, feature_dim: int,
                 factor_rank: int | None = None, seed: int = 20261110) -> None:
        super().__init__()
        if (basis.ndim != 2 or basis.shape[1] < 1 or feature_dim < 1
                or not torch.isfinite(basis).all()):
            raise ValueError("Finite basis[vocabulary, rank] and positive feature_dim required")
        if anchor.weights.device != basis.device or anchor.weights.dtype != basis.dtype:
            raise ValueError("Anchor and basis must use the same device and floating dtype")
        self.anchor = anchor
        self.register_buffer("basis", basis.detach())
        self.feature_dim = int(feature_dim)
        self.factor_rank = factor_rank
        rank = basis.shape[1]
        self.bias = nn.Parameter(basis.new_zeros(rank))
        if factor_rank is None:
            self.linear = nn.Parameter(basis.new_zeros(feature_dim, rank))
        else:
            if not 1 <= factor_rank <= min(feature_dim, rank):
                raise ValueError("Invalid factor rank")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            down = torch.randn(feature_dim, factor_rank, generator=generator,
                               dtype=basis.dtype) / math.sqrt(feature_dim)
            self.down = nn.Parameter(down.to(basis.device))
            self.up = nn.Parameter(basis.new_zeros(factor_rank, rank))

    def projected_coefficients(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Wrong feature dimensions")
        low = features @ self.linear if self.factor_rank is None else (features @ self.down) @ self.up
        return low + self.bias

    def correction(self, features: Tensor) -> Tensor:
        return self.projected_coefficients(features) @ self.basis.T

    def logits(self, peers: Tensor, features: Tensor) -> Tensor:
        if peers.shape[0] != features.shape[0] or peers.shape[-1] != self.basis.shape[0]:
            raise ValueError("Peer, feature and basis dimensions disagree")
        return self.anchor.logits(peers) + self.correction(features)

    def forward(self, peers: Tensor, features: Tensor) -> Tensor:
        return torch.log_softmax(self.logits(peers, features), -1)


class AnchoredMLP(nn.Module):
    def __init__(self, anchor: FrozenAnchor, basis: Tensor, feature_dim: int,
                 hidden: int = 32, seed: int = 20261110) -> None:
        super().__init__()
        if hidden < 1 or feature_dim < 1 or basis.ndim != 2:
            raise ValueError("Positive layer dimensions required")
        self.anchor = anchor
        self.register_buffer("basis", basis.detach())
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden)
        devices = [basis.device.index or 0] if basis.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(feature_dim, hidden, device=basis.device, dtype=basis.dtype)
            self.output = nn.Linear(hidden, basis.shape[1], device=basis.device, dtype=basis.dtype)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def projected_coefficients(self, features: Tensor) -> Tensor:
        return self.output(torch.tanh(self.hidden(features)))

    def correction(self, features: Tensor) -> Tensor:
        return self.projected_coefficients(features) @ self.basis.T

    def logits(self, peers: Tensor, features: Tensor) -> Tensor:
        return self.anchor.logits(peers) + self.correction(features)

    def forward(self, peers: Tensor, features: Tensor) -> Tensor:
        return torch.log_softmax(self.logits(peers, features), -1)


class AnchoredResponseGate(nn.Module):
    """Response-conditioned two-peer correction; both peer calls remain required."""

    def __init__(self, anchor: FrozenAnchor, feature_dim: int, strength: float = 1.) -> None:
        super().__init__()
        if len(anchor.weights) != 2 or feature_dim < 1 or not math.isfinite(strength) or strength <= 0:
            raise ValueError("Two peers, positive dimensions and gate strength required")
        self.anchor = anchor
        self.feature_dim = int(feature_dim)
        self.strength = float(strength)
        self.gate = nn.Linear(feature_dim, 1, device=anchor.weights.device, dtype=anchor.weights.dtype)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def logits(self, peers: Tensor, features: Tensor) -> Tensor:
        delta = self.strength * torch.tanh(self.gate(features)) * (peers[:, 0] - peers[:, 1])
        return self.anchor.logits(peers) + delta

    def forward(self, peers: Tensor, features: Tensor) -> Tensor:
        return torch.log_softmax(self.logits(peers, features), -1)
