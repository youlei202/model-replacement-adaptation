"""Streaming development-only optimization with inner-TV checkpoint selection.

The runner supplies explicit document-cluster splits. Outer validation is scored
after fitting, never passed to this routine as a stopping set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from .losses import objective, per_item, projected_objective


class ArrayData:
    """Aligned read-only banks; only the requested minibatch is copied to GPU."""

    def __init__(self, peers, features, target_logits, indices=None, *, role="development"):
        self.peers = peers
        self.features = features
        self.target_logits = target_logits
        self.role = str(role)
        shape = target_logits.shape
        if len(shape) != 2 or shape[0] == 0 or shape[1] < 2:
            raise ValueError("Nonempty target bank[items,vocabulary] required")
        if isinstance(peers, (tuple, list)):
            if not peers or any(tuple(p.shape) != tuple(shape) for p in peers):
                raise ValueError("Peer banks must align with target items and output support")
            self.peer_count = len(peers)
        else:
            if len(peers.shape) != 3 or peers.shape[0] != shape[0] or peers.shape[2] != shape[1]:
                raise ValueError("Stacked peers must have shape[items,peers,vocabulary]")
            self.peer_count = peers.shape[1]
        if features is not None and (len(features.shape) != 2 or features.shape[0] != shape[0]):
            raise ValueError("Features must align with item rows")
        self.indices = np.arange(shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if (self.indices.ndim != 1 or len(self.indices) == 0 or len(set(self.indices.tolist())) != len(self.indices)
                or self.indices.min() < 0 or self.indices.max() >= shape[0]):
            raise ValueError("Unique, valid, nonempty source row indices required")
        self.feature_dim = 0 if features is None else features.shape[1]
        self.vocab = shape[1]

    def __len__(self):
        return len(self.indices)

    def subset(self, local_indices, *, role=None):
        return ArrayData(self.peers, self.features, self.target_logits,
                         self.indices[np.asarray(local_indices, dtype=np.int64)], role=role or self.role)

    @staticmethod
    def _take(source, ids, device, dtype):
        if isinstance(source, Tensor):
            idx = torch.as_tensor(ids, dtype=torch.long, device=source.device)
            return source.detach().index_select(0, idx).to(device=device, dtype=dtype)
        # Advanced indexing copies a memmap view and avoids read-only tensor warnings.
        values = np.array(source[ids], copy=True, order="C")
        return torch.from_numpy(values).to(device=device, dtype=dtype)

    def batch(self, local_indices, device, dtype=torch.float32):
        ids = self.indices[np.asarray(local_indices, dtype=np.int64)]
        if isinstance(self.peers, (tuple, list)):
            peers = torch.stack([self._take(p, ids, device, dtype) for p in self.peers], dim=1)
        else:
            peers = self._take(self.peers, ids, device, dtype)
        features = (torch.empty((len(ids), 0), device=device, dtype=dtype) if self.features is None
                    else self._take(self.features, ids, device, dtype))
        target = self._take(self.target_logits, ids, device, dtype)
        return peers, features, target


@dataclass
class TrainConfig:
    loss: str = "kl"
    epochs: int = 30
    lr: float = .001
    batch_size: int = 32
    eval_batch_size: int = 64
    eval_every: int = 1
    patience: int = 10
    seed: int = 20261110
    weight_decay: float = 0.
    residual_penalty: float = 0.
    gradient_clip: float = 5.
    min_delta: float = 0.
    max_steps: int | None = None

    def validate(self):
        if (self.epochs < 1 or self.lr <= 0 or self.batch_size < 1 or self.eval_batch_size < 1
                or self.eval_every < 1 or self.patience < 0 or self.weight_decay < 0
                or self.residual_penalty < 0 or self.gradient_clip <= 0 or self.min_delta < 0
                or self.max_steps is not None and self.max_steps < 1):
            raise ValueError("Invalid optimization configuration")


def _placement(model):
    tensors = list(model.parameters()) + list(model.buffers())
    reference = next((p for p in tensors if p is not None and p.is_floating_point()), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    return reference.device, reference.dtype


def evaluate(model: nn.Module, data: ArrayData, *, batch_size: int = 64,
             device=None, return_per_item: bool = True) -> dict:
    if batch_size < 1:
        raise ValueError("Positive evaluation minibatch required")
    inferred_device, dtype = _placement(model)
    device = inferred_device if device is None else torch.device(device)
    before = model.training
    model.eval()
    values = {}
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            peer, feat, target = data.batch(np.arange(start, min(start + batch_size, len(data))), device, dtype)
            log_q = model(peer, feat)
            for key, value in per_item(target, log_q).items():
                values.setdefault(key, []).append(value.detach().cpu().numpy())
    model.train(before)
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    means = {key: float(np.asarray(value, dtype=np.float64).mean())
             for key, value in arrays.items() if key not in ("target_argmax", "prediction_argmax")}
    result = {"means": means, "items": len(data), "role": data.role,
              "max_probability_sum_error": float(arrays["probability_sum_error"].max())}
    # Direct keys make simple experiment runners concise; means remains explicit.
    result.update(means)
    if return_per_item:
        result["per_item"] = arrays
        result["source_row_indices"] = data.indices.copy()
    return result


def fit(model: nn.Module, train_data: ArrayData, stop_data: ArrayData,
        config: TrainConfig | None = None, *, callback: Callable | None = None,
        **config_overrides) -> dict:
    """Optimize on train and restore the best inner-stop TV state, including epoch0.

    ``callback(epoch, history_row, model, is_best)`` is called after each inner
    evaluation so the runner can save deployable encodings and all diagnostics.
    The returned history retains worsening epochs; a zero best update is a tie.
    """
    if config is None:
        config = TrainConfig(**config_overrides)
    elif config_overrides:
        config = TrainConfig(**{**asdict(config), **config_overrides})
    config.validate()
    if stop_data.role in ("validation", "outer_validation", "test", "confirmation"):
        raise ValueError("Stopping must use inner development, not outer validation or confirmation")
    if train_data.target_logits is stop_data.target_logits and np.intersect1d(train_data.indices, stop_data.indices).size:
        raise ValueError("Train and inner stopping rows overlap")
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable target parameters")
    device, dtype = _placement(model)
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    start_time = time.perf_counter()
    history = []
    initial = evaluate(model, stop_data, batch_size=config.eval_batch_size, return_per_item=False)
    best_tv = initial["tv"]
    best_epoch = 0
    best_step = 0
    best_parameters = {name: p.detach().clone() for name, p in parameters}
    initial_row = {"epoch": 0, "step": 0, "inner_stop_tv": initial["tv"],
                   "inner_stop_kl": initial["kl"], "inner_stop_argmax_disagreement": initial["argmax_disagreement"],
                   "fit_objective": None, "is_best": True, "seconds": time.perf_counter() - start_time}
    history.append(initial_row)
    if callback:
        callback(0, initial_row, model, True)
    first_gradients = None
    steps = 0
    stale = 0
    stop_reason = "epochs_exhausted"
    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.
        seen = 0
        for offset in range(0, len(train_data), config.batch_size):
            if offset == 0:
                permutation = rng.permutation(len(train_data))
            selected = permutation[offset:offset + config.batch_size]
            peers, features, target = train_data.batch(selected, device, dtype)
            if config.loss in ("projected_logit_mse", "legacy_projected_logit_mse"):
                if not hasattr(model, "projected_coefficients"):
                    raise ValueError("Projected residual objective requires a public-basis residual model")
                value = projected_objective(model, peers, features, target)
            else:
                value = objective(target, model(peers, features), config.loss)
            if config.residual_penalty:
                # Penalize retained map output in the public coefficient space,
                # not an unregistered rescaling of the probability objective.
                if not hasattr(model, "projected_coefficients"):
                    raise ValueError("Coefficient penalty requires a basis residual")
                value = value + config.residual_penalty * model.projected_coefficients(features).square().mean()
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite objective")
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in parameters):
                raise FloatingPointError("Nonfinite gradient")
            if first_gradients is None:
                first_gradients = {name: None if p.grad is None else float(p.grad.detach().norm())
                                   for name, p in parameters}
            torch.nn.utils.clip_grad_norm_([p for _, p in parameters], config.gradient_clip)
            optimizer.step()
            weighted_loss += float(value.detach()) * len(selected)
            seen += len(selected)
            steps += 1
            if config.max_steps is not None and steps >= config.max_steps:
                stop_reason = "max_steps"
                break
        force = epoch == config.epochs or stop_reason == "max_steps"
        if epoch % config.eval_every == 0 or force:
            measured = evaluate(model, stop_data, batch_size=config.eval_batch_size, return_per_item=False)
            improved = measured["tv"] < best_tv - config.min_delta
            if improved:
                best_tv = measured["tv"]
                best_epoch = epoch
                best_step = steps
                best_parameters = {name: p.detach().clone() for name, p in parameters}
                stale = 0
            else:
                stale += 1
            row = {"epoch": epoch, "step": steps, "inner_stop_tv": measured["tv"],
                   "inner_stop_kl": measured["kl"], "inner_stop_argmax_disagreement": measured["argmax_disagreement"],
                   "fit_objective": weighted_loss / seen, "is_best": improved,
                   "seconds": time.perf_counter() - start_time}
            history.append(row)
            if callback:
                callback(epoch, row, model, improved)
            if config.patience and stale >= config.patience:
                stop_reason = "inner_patience"
                break
        if stop_reason == "max_steps":
            break
    last_fit = evaluate(model, train_data, batch_size=config.eval_batch_size, return_per_item=False)
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(best_parameters[name])
    model.eval()
    best_fit = evaluate(model, train_data, batch_size=config.eval_batch_size, return_per_item=False)
    return {"scope": "inner-development checkpoint selection; outer validation and confirmation unopened",
            "config": asdict(config), "inner_stop_tv_before": initial["tv"], "inner_stop_tv_after": best_tv,
            "chosen_epoch": best_epoch, "chosen_step": best_step, "completed_epochs": epoch,
            "completed_steps": steps, "history": history, "first_step_gradient_norms": first_gradients,
            "last_fit_means": last_fit["means"], "selected_fit_means": best_fit["means"],
            "stop_reason": stop_reason, "seconds": time.perf_counter() - start_time,
            "training_items": len(train_data), "inner_stop_items": len(stop_data),
            "public_buffers_not_copied_into_target_checkpoint": True}
