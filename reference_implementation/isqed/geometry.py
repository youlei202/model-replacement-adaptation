from __future__ import annotations

import cvxpy as cp
import numpy as np


class DISCOSolver:
    """Project a target response onto the convex hull of peer responses."""

    @staticmethod
    def solve_weights_and_distance(
        target_vec: np.ndarray, peer_matrix: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return the projection distance and simplex weights."""
        target = np.asarray(target_vec).reshape(-1)
        peers = np.asarray(peer_matrix)
        if peers.ndim != 2:
            raise ValueError("peer_matrix must have shape (dimension, peers)")
        dimension, peer_count = peers.shape
        if target.size != dimension:
            raise ValueError(
                "shape mismatch: target %s and peers %s"
                % (target.shape, peers.shape)
            )

        weights = cp.Variable(peer_count)
        objective = cp.Minimize(cp.norm(target - peers @ weights, 2))
        problem = cp.Problem(objective, [weights >= 0, cp.sum(weights) == 1])

        for solver in (cp.ECOS, cp.SCS, None):
            try:
                if solver is None:
                    problem.solve()
                else:
                    problem.solve(solver=solver)
            except cp.error.SolverError:
                continue
            if weights.value is not None:
                return float(problem.value), np.asarray(weights.value).reshape(-1)

        return float("inf"), np.full(peer_count, 1.0 / peer_count)

    @staticmethod
    def compute_pier(
        target_vec: np.ndarray, peer_matrix: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Return the residual vector for a fixed set of peer weights."""
        target = np.asarray(target_vec).reshape(-1)
        return target - np.asarray(peer_matrix) @ np.asarray(weights)
