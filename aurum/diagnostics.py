"""Bounded, fit-only mechanism diagnostics; their outputs are not candidates.

The per-query coefficient experiment sees a teacher for every fitted query. It
tests output-basis representability and optimization, not deployable prediction
or a statistically independent generalization bound. No validation/confirmation
rows are accepted. All optimizers retain the zero/initial predictor in the trace.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .losses import objective, per_item
from .train import ArrayData, _placement


def _fit_sample(data: ArrayData, n_examples: int, seed: int) -> ArrayData:
    if data.role not in ("fit", "inner_train"):
        raise ValueError("Mechanism diagnostics require explicitly labeled fit/inner_train rows")
    if n_examples < 1:
        raise ValueError("Positive diagnostic sample size required")
    ids = np.random.default_rng(seed).choice(len(data), min(len(data), n_examples), replace=False)
    return data.subset(np.sort(ids), role="inner_train")


def _metrics(target, log_q):
    with torch.no_grad():
        values = per_item(target, log_q)
        return {key: float(value.double().mean()) for key, value in values.items()
                if key not in ("target_argmax", "prediction_argmax")}


def _bounded_fit(model, peers, features, target, *, steps, lr, loss, eval_every,
                 gradient_clip, callback=None):
    if steps < 1 or lr <= 0 or not math.isfinite(lr) or eval_every < 1 or gradient_clip <= 0:
        raise ValueError("Invalid diagnostic optimization budget")
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError("A trainable diagnostic model is required")
    optimizer = torch.optim.Adam([p for _, p in params], lr=lr)
    start = time.perf_counter()
    with torch.no_grad():
        before = _metrics(target, model(peers, features))
    best = dict(before)
    best_step = 0
    best_params = {name: p.detach().clone() for name, p in params}
    history = [{"step": 0, **before, "seconds": 0., "is_best_tv": True}]
    first_gradients = None
    last_gradients = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = objective(target, model(peers, features), loss)
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite diagnostic objective")
        value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in params):
            raise FloatingPointError("Nonfinite diagnostic gradient")
        if first_gradients is None:
            first_gradients = {name: None if p.grad is None else float(p.grad.norm()) for name, p in params}
        if step == steps:
            last_gradients = {name: None if p.grad is None else float(p.grad.norm()) for name, p in params}
        torch.nn.utils.clip_grad_norm_([p for _, p in params], gradient_clip)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                now = _metrics(target, model(peers, features))
            improved = now["tv"] < best["tv"]
            if improved:
                best, best_step = now, step
                best_params = {name: p.detach().clone() for name, p in params}
            row = {"step": step, **now, "objective_before_step": float(value.detach()),
                   "seconds": time.perf_counter() - start, "is_best_tv": improved}
            history.append(row)
            if callback:
                callback(row)
    final = dict(history[-1])
    with torch.no_grad():
        for name, parameter in params:
            parameter.copy_(best_params[name])
    return {"before": before, "best": best, "final": final, "best_step": best_step,
            "completed_steps": steps, "loss": loss, "lr": lr, "history": history,
            "first_gradient_norms": first_gradients, "final_pre_update_gradient_norms": last_gradients,
            "trainable_scalar_count": sum(p.numel() for _, p in params),
            "seconds": time.perf_counter() - start,
            "absolute_fit_tv_gain": before["tv"] - best["tv"],
            "relative_fit_tv_gain": (before["tv"] - best["tv"]) / max(before["tv"], 1e-30)}


def tiny_fit_diagnostic(model: nn.Module, data: ArrayData, *, n_examples: int = 32,
                        steps: int = 300, lr: float = .01, loss: str = "kl",
                        seed: int = 20261110, eval_every: int = 10,
                        gradient_clip: float = 5., callback=None) -> dict:
    """Can the given map fit 16--32 examples at all? Never mutates the model.

    This is deliberately training-set memorization, with the exact same rows
    used to optimize and report fit. It must not enter a deployment catalog or
    be reported as held-out quality. A small gain alone does not distinguish
    output-basis limitations from optimizer or feature-map limitations.
    """
    sample = _fit_sample(data, n_examples, seed)
    diagnostic_model = copy.deepcopy(model)
    device, dtype = _placement(diagnostic_model)
    peers, features, target = sample.batch(np.arange(len(sample)), device, dtype)
    result = _bounded_fit(diagnostic_model, peers, features, target, steps=steps, lr=lr,
                          loss=loss, eval_every=eval_every, gradient_clip=gradient_clip, callback=callback)
    result.update(kind="tiny_fit_memorization", data_role="inner_fit_diagnostic_only",
                  n_examples=len(sample), source_row_indices=sample.indices.tolist(), seed=seed,
                  same_rows_for_optimization_and_report=True, deployable_candidate=False,
                  original_model_mutated=False, fit_source_role=data.role)
    if features.shape[1]:
        result["feature_conditioning"] = feature_conditioning(features)
    return result


class _PerQueryCoefficients(nn.Module):
    def __init__(self, anchor_scores, basis):
        super().__init__()
        self.register_buffer("anchor_scores", anchor_scores.detach())
        self.register_buffer("basis", basis.detach())
        self.coefficients = nn.Parameter(basis.new_zeros(len(anchor_scores), basis.shape[1]))

    def forward(self, peers, features=None):
        return torch.log_softmax(self.anchor_scores + self.coefficients @ self.basis.T, -1)


def per_query_basis_oracle(anchor, basis: torch.Tensor, data: ArrayData, *,
                           n_examples: int = 32, steps: int = 400, lr: float = .5,
                           loss: str = "kl", seed: int = 20261110,
                           eval_every: int = 10, gradient_clip: float = 5.,
                           include_coefficients: bool = False, callback=None) -> dict:
    """Achieved fit with an independent basis coefficient vector per query.

    Despite the conventional name "oracle", finite optimization does NOT give
    a certified lower bound on TV. A gain demonstrates representable teacher
    corrections; a null is interpretable only together with solver convergence
    and wider/probability-basis diagnostics. These query-specific target-derived
    vectors are not free public features or a prediction method for unseen data.
    """
    sample = _fit_sample(data, n_examples, seed)
    if basis.ndim != 2 or basis.shape[0] != data.vocab:
        raise ValueError("The diagnostic basis must use the full target support")
    peers, features, target = sample.batch(np.arange(len(sample)), basis.device, basis.dtype)
    with torch.no_grad():
        scores = anchor.logits(peers)
    model = _PerQueryCoefficients(scores, basis)
    result = _bounded_fit(model, peers, features, target, steps=steps, lr=lr,
                          loss=loss, eval_every=eval_every, gradient_clip=gradient_clip, callback=callback)
    result.update(kind="per_query_basis_coefficient_capacity_diagnostic", data_role="inner_fit_diagnostic_only",
                  n_examples=len(sample), source_row_indices=sample.indices.tolist(), seed=seed,
                  basis_rank=basis.shape[1], same_rows_for_optimization_and_report=True,
                  deployable_candidate=False, certified_lower_bound=False,
                  target_used_for_each_query=True, fit_source_role=data.role,
                  coefficient_l2_norm=float(model.coefficients.detach().norm()),
                  coefficient_max_absolute=float(model.coefficients.detach().abs().max()),
                  interpretation="Achieved representability/optimization diagnostic; not held-out error or a TV lower bound")
    if include_coefficients:
        result["best_fit_coefficients"] = model.coefficients.detach().cpu().tolist()
    return result


def feature_conditioning(features, indices=None, *, relative_tolerance: float = 1e-8) -> dict:
    """Small Gram spectra, rank and scale summaries for peer-only features.

    Infinite condition numbers are represented by None, keeping JSON strict.
    The diagnostic never modifies or refits the supplied feature transform.
    """
    if relative_tolerance <= 0 or not math.isfinite(relative_tolerance):
        raise ValueError("Positive finite rank tolerance required")
    if isinstance(features, torch.Tensor):
        values = features.detach()
        if indices is not None:
            values = values[torch.as_tensor(indices, dtype=torch.long, device=values.device)]
        values = values.double().cpu()
    else:
        values = np.asarray(features if indices is None else features[indices])
        values = torch.from_numpy(np.array(values, dtype=np.float64, copy=True))
    if values.ndim != 2 or min(values.shape) < 1 or not torch.isfinite(values).all():
        raise ValueError("Finite nonempty feature matrix required")
    centered = values - values.mean(0)
    scale = centered.square().mean(0).sqrt()
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(values)).clamp_min(0).flip(0)
    largest = float(eigenvalues[0])
    floor = largest * relative_tolerance
    positive = eigenvalues[eigenvalues > floor]
    smallest = float(eigenvalues[-1])
    condition = largest / smallest if smallest > floor else None
    return {"items": len(values), "dimensions": values.shape[1],
            "centered_effective_rank": len(positive), "relative_rank_tolerance": relative_tolerance,
            "centered_gram_condition_number": condition,
            "centered_gram_eigenvalues": eigenvalues.tolist(),
            "rank_deficient_at_tolerance": len(positive) < values.shape[1],
            "feature_std_min": float(scale.min()), "feature_std_median": float(scale.median()),
            "feature_std_max": float(scale.max()), "feature_max_absolute": float(values.abs().max()),
            "constant_columns": int((scale == 0).sum()),
            "interpretation": "A small-sample diagnostic can be rank limited by its item count"}


def learning_summary(training_record: dict, *, anchor_fit_tv: float | None = None,
                     anchor_stop_tv: float | None = None, outer_tv: float | None = None) -> dict:
    """Describe observable optimization/generalization signals, without a verdict."""
    history = training_record.get("history", [])
    if not history:
        raise ValueError("A nonempty training history is required")
    stop_values = [row["inner_stop_tv"] for row in history]
    fit_tv = training_record.get("selected_fit_means", {}).get("tv")
    stop_tv = training_record.get("inner_stop_tv_after", min(stop_values))
    first_gradients = training_record.get("first_step_gradient_norms", {})
    live = [v for v in first_gradients.values() if v is not None and v > 0]
    result = {"checkpoint_evaluations": len(history), "chosen_epoch": training_record.get("chosen_epoch"),
              "completed_epochs": training_record.get("completed_epochs"),
              "first_gradient_nonzero_tensors": len(live),
              "first_gradient_tensor_count": len(first_gradients),
              "inner_stop_tv_first": stop_values[0], "inner_stop_tv_last": stop_values[-1],
              "inner_stop_tv_selected": stop_tv, "fit_tv_selected": fit_tv,
              "inner_stop_improvement_from_initial": stop_values[0] - stop_tv,
              "outer_development_tv": outer_tv,
              "selected_fit_to_inner_stop_gap": None if fit_tv is None else stop_tv - fit_tv,
              "initial_checkpoint_retained": training_record.get("chosen_epoch") == 0,
              "stop_reason": training_record.get("stop_reason")}
    if anchor_fit_tv is not None and fit_tv is not None:
        result["fit_gain_over_anchor"] = anchor_fit_tv - fit_tv
    if anchor_stop_tv is not None:
        result["inner_stop_gain_over_anchor"] = anchor_stop_tv - stop_tv
    if fit_tv is not None and outer_tv is not None:
        result["fit_to_outer_development_gap"] = outer_tv - fit_tv
    return result
