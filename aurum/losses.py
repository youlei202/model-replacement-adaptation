"""Full-support probability losses; no independent vocabulary-chunk softmax."""
from __future__ import annotations

import torch
from torch import Tensor


def _check_shapes(target: Tensor, prediction: Tensor) -> None:
    if target.ndim != 2 or target.shape != prediction.shape or len(target) == 0 or target.shape[1] < 2:
        raise ValueError("Matching nonempty [items, full_vocabulary] tensors required")


def per_item(target_logits: Tensor, prediction_log_probs: Tensor, *, check_finite: bool = True) -> dict[str, Tensor]:
    _check_shapes(target_logits, prediction_log_probs)
    if check_finite and (not torch.isfinite(target_logits).all() or not torch.isfinite(prediction_log_probs).all()):
        raise ValueError("Nonfinite target logits or predicted log probabilities")
    log_p = torch.log_softmax(target_logits.detach(), -1)
    p = log_p.exp()
    q = prediction_log_probs.exp()
    target_argmax = target_logits.detach().argmax(-1)
    prediction_argmax = prediction_log_probs.argmax(-1)
    return {
        "tv": .5 * (p - q).abs().sum(-1),
        "kl": (p * (log_p - prediction_log_probs)).sum(-1),
        "argmax_disagreement": (target_argmax != prediction_argmax).to(p.dtype),
        "target_argmax": target_argmax,
        "prediction_argmax": prediction_argmax,
        "probability_sum_error": (q.sum(-1) - 1).abs(),
    }


def objective(target_logits: Tensor, log_q: Tensor, name: str = "kl", *, basis: Tensor | None = None) -> Tensor:
    _check_shapes(target_logits, log_q)
    aliases = {"full_support_forward_kl": "kl", "legacy_projected_logit_mse": "projected_logit_mse"}
    name = aliases.get(name, name)
    if name in ("projected_logit_mse", "full_logit_mse"):
        target = target_logits.detach() - target_logits.detach().mean(-1, keepdim=True)
        pred = log_q - log_q.mean(-1, keepdim=True)
        difference = target - pred
        if name == "projected_logit_mse":
            if basis is None or basis.ndim != 2 or basis.shape[0] != target.shape[1]:
                raise ValueError("Projected MSE requires the public output basis")
            difference = difference @ basis
        return difference.square().mean()
    log_p = torch.log_softmax(target_logits.detach(), -1)
    p = log_p.exp()
    if name == "kl":
        return (p * (log_p - log_q)).sum(-1).mean()
    if name == "tv":
        return (.5 * (p - log_q.exp()).abs().sum(-1)).mean()
    if name == "tv_kl":
        return (.5 * (p - log_q.exp()).abs().sum(-1) + .05 * (p * (log_p - log_q)).sum(-1)).mean()
    raise ValueError(f"Unknown objective {name!r}")


def projected_objective(model, peers: Tensor, features: Tensor, target_logits: Tensor) -> Tensor:
    """Legacy projected residual objective without materializing output logits.

    The residual basis need not be exactly orthonormal after FP32 storage, so its
    small Gram matrix is included rather than silently assuming an identity.
    """
    basis = model.basis
    target_low = (target_logits.detach() - model.anchor.logits(peers).detach()) @ basis
    correction_low = model.projected_coefficients(features) @ (basis.T @ basis)
    return (target_low - correction_low).square().mean()
