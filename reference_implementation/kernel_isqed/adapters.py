from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor

from isqed.geometry import DISCOSolver

from .contracts import GLOBAL_SEED, LAMBDA_GRID, MLP_LAMBDA_INDICES, MLP_WIDTHS, RFF_DIMENSIONS
from .response import coordinates_to_probabilities, total_variation


@dataclass
class AdapterContext:
    peer_indices: Tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    sigma: float
    fit_features: np.ndarray
    output_dimension: int
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    rff_weight: Optional[np.ndarray] = None
    rff_phase: Optional[np.ndarray] = None


@dataclass
class AdapterState:
    adapter: str
    peer_indices: Tuple[int, ...]
    lambda_index: int = 0
    rank: int = 0
    weights: Optional[np.ndarray] = None
    coefficient: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    hidden_weight: Optional[np.ndarray] = None
    hidden_bias: Optional[np.ndarray] = None
    output_weight: Optional[np.ndarray] = None
    output_bias: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def peer_subsets(q: int, peer_count: int = 3) -> Tuple[Tuple[int, ...], ...]:
    if q < 1 or q > peer_count:
        raise ValueError("q must be between one and peer_count")
    return tuple(combinations(range(peer_count), q))


def flatten_peer_coordinates(peer_coordinates: np.ndarray, peer_indices: Sequence[int]) -> np.ndarray:
    values = np.asarray(peer_coordinates, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("peer coordinates must have shape [items, peers, coordinates]")
    return values[:, np.asarray(peer_indices, dtype=int), :].reshape(values.shape[0], -1)


def make_context(
    fit_peer_coordinates: np.ndarray,
    peer_indices: Sequence[int],
    rff_seed: int = GLOBAL_SEED,
    max_rff_dimension: int = max(RFF_DIMENSIONS),
) -> AdapterContext:
    features = flatten_peer_coordinates(fit_peer_coordinates, peer_indices)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (features - mean) / scale
    sigma = median_bandwidth(standardized)
    kernel = rbf_kernel(standardized, standardized, sigma)
    eigvals, eigvecs = np.linalg.eigh(kernel)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rng = np.random.RandomState(rff_seed)
    weight = rng.normal(0.0, 1.0 / sigma, size=(standardized.shape[1], max_rff_dimension))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=max_rff_dimension)
    output_dimension = int(fit_peer_coordinates.shape[-1])
    return AdapterContext(
        tuple(int(value) for value in peer_indices),
        mean,
        scale,
        sigma,
        standardized,
        output_dimension,
        eigvals,
        eigvecs,
        weight,
        phase,
    )


def transform_features(context: AdapterContext, peer_coordinates: np.ndarray) -> np.ndarray:
    values = flatten_peer_coordinates(peer_coordinates, context.peer_indices)
    return (values - context.mean) / context.scale


def median_bandwidth(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    rng = np.random.RandomState(GLOBAL_SEED)
    count = min(20000, len(values) * (len(values) - 1) // 2)
    left = rng.randint(0, len(values), size=max(count * 2, 1))
    right = rng.randint(0, len(values), size=max(count * 2, 1))
    keep = left != right
    distances = np.sqrt(np.sum((values[left[keep]] - values[right[keep]]) ** 2, axis=1))
    median = float(np.median(distances[:count])) if len(distances) else 1.0
    return max(median, 1e-8)


def rbf_kernel(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distance = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.exp(-distance / (2.0 * sigma * sigma))


def rff_features(context: AdapterContext, features: np.ndarray, dimension: int) -> np.ndarray:
    if context.rff_weight is None or context.rff_phase is None:
        raise ValueError("context does not contain public RFF features")
    weight = context.rff_weight[:, :dimension]
    phase = context.rff_phase[:dimension]
    return np.sqrt(2.0 / dimension) * np.cos(features @ weight + phase)


def _ridge(features: np.ndarray, targets: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    mean_x = features.mean(axis=0)
    mean_y = targets.mean(axis=0)
    centered_x = features - mean_x
    centered_y = targets - mean_y
    system = centered_x.T @ centered_x + len(features) * lam * np.eye(features.shape[1])
    coefficient = np.linalg.solve(system, centered_x.T @ centered_y)
    bias = mean_y - mean_x @ coefficient
    return coefficient, bias


def _convex_weights(peer_coordinates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(peer_coordinates, dtype=np.float64)
    item_count, peer_count, coordinate_count = values.shape
    design = values.transpose(0, 2, 1).reshape(item_count * coordinate_count, peer_count)
    target = targets.reshape(-1)

    def objective(weights: np.ndarray) -> float:
        residual = design @ weights - target
        return float(residual @ residual)

    disco_distance, disco_weights = DISCOSolver.solve_weights_and_distance(target, design)
    disco_weights = np.asarray(disco_weights, dtype=np.float64)
    if np.isfinite(disco_distance) and np.isfinite(disco_weights).all():
        return _project_simplex(disco_weights)

    initial = np.ones(peer_count, dtype=np.float64) / peer_count
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * peer_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if result.success and np.isfinite(result.x).all():
        return np.asarray(result.x, dtype=np.float64)
    weights = initial.copy()
    spectral = float(np.linalg.norm(design, 2))
    step = 1.0 / max(2.0 * spectral * spectral, 1e-12)
    for _ in range(5000):
        gradient = 2.0 * design.T @ (design @ weights - target)
        candidate = _project_simplex(weights - step * gradient)
        if np.max(np.abs(candidate - weights)) < 1e-12:
            break
        weights = candidate
    return weights


def _project_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = sorted_values - cumulative / indices > 0
    rho = int(indices[valid][-1])
    threshold = cumulative[rho - 1] / rho
    return np.maximum(values - threshold, 0.0)


def fit_adapter(
    adapter: str,
    context: AdapterContext,
    fit_peer_coordinates: np.ndarray,
    fit_targets: np.ndarray,
    lambda_index: int = 0,
    rank: int = 0,
) -> AdapterState:
    from .frozen import require_pretest_fitting

    require_pretest_fitting()
    lam = float(LAMBDA_GRID[lambda_index])
    if adapter == "convex":
        selected = fit_peer_coordinates[:, np.asarray(context.peer_indices, dtype=int), :]
        weights = _convex_weights(selected, fit_targets)
        return AdapterState(adapter, context.peer_indices, weights=weights)
    features = transform_features(context, fit_peer_coordinates)
    if adapter == "linear":
        coefficient, bias = _ridge(features, fit_targets, lam)
        return AdapterState(
            adapter, context.peer_indices, lambda_index, features.shape[1], coefficient=coefficient, bias=bias
        )
    if adapter == "rff":
        dimension = int(rank)
        mapped = rff_features(context, features, dimension)
        coefficient, bias = _ridge(mapped, fit_targets, lam)
        return AdapterState(adapter, context.peer_indices, lambda_index, dimension, coefficient=coefficient, bias=bias)
    if adapter == "krr":
        if context.eigvals is None or context.eigvecs is None:
            raise ValueError("KRR context is missing its eigendecomposition")
        projected = context.eigvecs.T @ fit_targets
        alpha = context.eigvecs @ (projected / (context.eigvals[:, None] + len(features) * lam))
        return AdapterState(adapter, context.peer_indices, lambda_index, len(features), alpha=alpha)
    if adapter == "mlp":
        hidden_width = int(rank)
        feature_mean = features.mean(axis=0)
        feature_scale = features.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        standardized = (features - feature_mean) / feature_scale
        estimator = MLPRegressor(
            hidden_layer_sizes=(hidden_width,),
            activation="tanh",
            solver="adam",
            alpha=len(features) * lam,
            batch_size=min(256, len(features)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=GLOBAL_SEED,
            tol=1e-5,
        )
        estimator.fit(standardized, fit_targets)
        if len(estimator.coefs_) != 2 or len(estimator.intercepts_) != 2:
            raise RuntimeError("the compact adapter must remain a two-layer MLP")
        hidden_weight = np.asarray(estimator.coefs_[0], dtype=np.float64)
        hidden_weight = hidden_weight / feature_scale.reshape(-1, 1)
        hidden_bias = np.asarray(estimator.intercepts_[0], dtype=np.float64)
        hidden_bias = hidden_bias - feature_mean @ hidden_weight
        return AdapterState(
            adapter,
            context.peer_indices,
            lambda_index,
            hidden_width,
            hidden_weight=hidden_weight,
            hidden_bias=hidden_bias,
            output_weight=np.asarray(estimator.coefs_[1], dtype=np.float64),
            output_bias=np.asarray(estimator.intercepts_[1], dtype=np.float64),
            metadata={
                "activation": "tanh",
                "solver": "adam",
                "random_state": GLOBAL_SEED,
                "input_standardization": "folded_into_first_layer",
            },
        )
    raise ValueError("unknown adapter %s" % adapter)


def predict_coordinates(state: AdapterState, context: AdapterContext, peer_coordinates: np.ndarray) -> np.ndarray:
    if state.adapter == "mean_all":
        return peer_coordinates[:, np.asarray(state.peer_indices, dtype=int), :].mean(axis=1)
    if state.adapter == "peer":
        return peer_coordinates[:, state.peer_indices[0], :]
    if state.adapter == "convex":
        if state.weights is None:
            raise ValueError("convex state has no weights")
        selected = peer_coordinates[:, np.asarray(state.peer_indices, dtype=int), :]
        return np.einsum("npd,p->nd", selected, state.weights)
    features = transform_features(context, peer_coordinates)
    if state.adapter == "linear":
        return features @ state.coefficient + state.bias
    if state.adapter == "rff":
        mapped = rff_features(context, features, state.rank)
        return mapped @ state.coefficient + state.bias
    if state.adapter == "krr":
        kernel = rbf_kernel(features, context.fit_features, context.sigma)
        return kernel @ state.alpha
    if state.adapter == "mlp":
        hidden = np.tanh(features @ state.hidden_weight + state.hidden_bias)
        return hidden @ state.output_weight + state.output_bias
    raise ValueError("unknown adapter %s" % state.adapter)


def mean_tv(target_coordinates: np.ndarray, predicted_coordinates: np.ndarray) -> float:
    actual = coordinates_to_probabilities(target_coordinates)
    predicted = coordinates_to_probabilities(predicted_coordinates)
    return float(np.mean(total_variation(actual, predicted)))


def configuration_grid(adapter: str) -> List[Tuple[int, int]]:
    if adapter == "convex":
        return [(0, 0)]
    if adapter == "linear":
        return [(index, 0) for index in range(len(LAMBDA_GRID))]
    if adapter == "rff":
        return [(index, dimension) for dimension in RFF_DIMENSIONS for index in range(len(LAMBDA_GRID))]
    if adapter == "krr":
        return [(index, 0) for index in range(len(LAMBDA_GRID))]
    if adapter == "mlp":
        return [(index, width) for width in MLP_WIDTHS for index in MLP_LAMBDA_INDICES]
    raise ValueError("unknown adapter %s" % adapter)


def select_native_configuration(
    adapter: str,
    context: AdapterContext,
    fit_peer_coordinates: np.ndarray,
    fit_targets: np.ndarray,
    validation_peer_coordinates: np.ndarray,
    validation_targets: np.ndarray,
) -> Tuple[AdapterState, float]:
    from .frozen import require_pretest_fitting

    require_pretest_fitting()
    candidates = []
    for lambda_index, rank in configuration_grid(adapter):
        state = fit_adapter(adapter, context, fit_peer_coordinates, fit_targets, lambda_index, rank)
        predicted = predict_coordinates(state, context, validation_peer_coordinates)
        error = mean_tv(validation_targets, predicted)
        tie = (error, rank or 0, -float(LAMBDA_GRID[lambda_index]), "%s_%d_%d" % (adapter, lambda_index, rank))
        candidates.append((tie, state))
    tie, state = min(candidates, key=lambda value: value[0])
    return state, float(tie[0])
