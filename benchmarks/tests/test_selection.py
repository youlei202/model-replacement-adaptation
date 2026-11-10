"""Run with: .venv-benchmarks/bin/python -m unittest discover -s benchmarks/tests -v"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selection_core import (CODECS, LAMBDAS, decode, encode, fit_weights, helmert, make_context,
                            paired_cluster_interval, predict, probabilities, rbf, state_bits)
from run_selection import (BUDGETS, PEERS, TARGETS, arrays, coverage, evaluate_cell, fit_cell,
                           read_responses, selection_policies, sha, validate_freeze)


class EncodingsTest(unittest.TestCase):
    def test_nnls_retains_scale_and_actual_fp64_bits(self):
        for q, bits in ((2, 304), (3, 368)):
            weights = np.arange(1, q + 1, dtype=float)[:, None]
            encoded = encode(weights, "nnls", tuple(range(q)), "native_fp64")
            restored, matrix = decode(encoded.blob)
            self.assertEqual(encoded.bits, bits)
            self.assertEqual(restored.bits, bits)
            np.testing.assert_array_equal(weights, matrix)
            self.assertNotEqual(matrix.sum(), 1)

    def test_all_codecs_roundtrip_metadata_padding_and_simplex(self):
        rng = np.random.default_rng(23)
        for family, matrix in (("krr", rng.normal(size=(7, 3))), ("convex", np.array([[.17], [.28], [.55]])), ("nnls", np.array([[.8], [1.2], [.03]]))):
            for codec in CODECS:
                with self.subTest(family=family, codec=codec):
                    encoded = encode(matrix, family, (0, 1, 2), codec, 4)
                    restored, result = decode(encoded.blob)
                    self.assertEqual(encoded.bits, state_bits(family, codec, *matrix.shape))
                    self.assertEqual(encoded.digest, restored.digest)
                    self.assertGreaterEqual(encoded.bits - encoded.meaningful_bits, 0)
                    self.assertLess(encoded.bits - encoded.meaningful_bits, 8)
                    self.assertTrue(np.isfinite(result).all())
                    if family == "convex":
                        self.assertAlmostEqual(result.sum(), 1)
                        self.assertTrue((result >= 0).all())
                    if codec == "native_fp64":
                        np.testing.assert_allclose(matrix, result, rtol=0, atol=1e-15)
        damaged = bytearray(encode(np.ones((1, 1)), "krr", (0, 1), "q3").blob)
        damaged[-1] |= 1
        with self.assertRaisesRegex(ValueError, "padding"):
            decode(bytes(damaged))


class FittingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(131)
        self.fit = rng.normal(size=(24, 3, 3))
        self.validation = rng.normal(size=(11, 3, 3))
        self.target = .2 * self.fit[:, 0] + 1.4 * self.fit[:, 2]

    def test_nnls_fit_recovers_unconstrained_sum(self):
        w = fit_weights(self.fit, self.target, (0, 2), "nnls")
        np.testing.assert_allclose(w, [.2, 1.4], atol=1e-12)

    def test_convex_face_solver_matches_independent_constrained_optimizer(self):
        pair = (0, 1, 2)
        w = fit_weights(self.fit, self.target, pair, "convex")
        design = self.fit.transpose(0, 2, 1).reshape(-1, 3)
        y = self.target.reshape(-1)
        objective = lambda z: np.square(design @ z - y).sum()
        result = minimize(objective, np.ones(3) / 3, method="SLSQP", bounds=[(0, 1)] * 3,
                          constraints={"type": "eq", "fun": lambda z: z.sum() - 1}, options={"ftol": 1e-12})
        self.assertTrue(result.success)
        self.assertAlmostEqual(objective(w), objective(result.x), places=8)

    def test_spectral_prediction_matches_direct_kernel_ridge(self):
        context = make_context(self.fit, self.validation, (0, 2))
        li = 8
        spectral = (context.eigenvectors.T @ self.target) / (context.eigenvalues[:, None] + len(self.fit) * LAMBDAS[li])
        encoded = encode(spectral, "krr", (0, 2), "native_fp64", li)
        decoded, matrix = decode(encoded.blob)
        output = predict(decoded, matrix, self.validation, context.validation_projection)
        gram = rbf(context.features, context.features, context.sigma)
        alpha = np.linalg.solve(gram + len(self.fit) * LAMBDAS[li] * np.eye(len(self.fit)), self.target)
        validation_features = (self.validation[:, (0, 2), :].reshape(len(self.validation), -1) - context.mean) / context.scale
        direct = rbf(validation_features, context.features, context.sigma) @ alpha
        np.testing.assert_allclose(output, direct, atol=1e-9)

    def test_helmert_preserves_softmax(self):
        logits = np.array([[10., 12., 15., 9.]])
        coordinates = logits @ helmert(3).T
        p = np.exp(logits - logits.max())
        np.testing.assert_allclose(probabilities(coordinates), p / p.sum(), atol=1e-15)


class ProtocolTest(unittest.TestCase):
    def test_missing_response_is_not_silently_removed(self):
        manifest = pd.DataFrame({"task_id": ["hellaswag"] * 2, "item_id": ["a", "b"], "prompt_hash": ["x", "y"]})
        response = manifest.iloc[:1].assign(model_id="peer")
        missing = coverage(manifest, response, ("peer",))
        self.assertEqual(missing.item_id.tolist(), ["b"])

    def test_conflicting_cache_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)
            row = {"model_id": "peer", "task_id": "hellaswag", "item_id": "a", "prompt_hash": "h", "revision": "r", "coord_0": 1.}
            pd.DataFrame([row]).to_parquet(p / "a.parquet")
            pd.DataFrame([row | {"coord_0": 2.}]).to_parquet(p / "b.parquet")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_responses([p], ("peer",))
            (p / "b.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "pinned checkpoint"):
                read_responses([p], ("peer",), expected_revisions={"peer": "different"})

    def test_cluster_interval_keeps_item_weighted_point_estimate(self):
        difference = np.array([1., 1., 1., -1.])
        mean, low, high = paired_cluster_interval(difference, np.array(["a", "a", "a", "b"]), draws=200)
        self.assertEqual(mean, .5)  # Equal weighting of cluster means would give 0.
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_freeze_before_test_and_checkpoint_integrity(self):
        rng = np.random.default_rng(93)
        manifest_rows, response_rows = [], []
        for split, count in (("fit", 16), ("validation", 9), ("iid_test", 10)):
            peers = rng.normal(size=(count, 3, 3))
            target = .3 * peers[:, 0] + .8 * peers[:, 2]
            for i in range(count):
                item = f"{split}_{i:03d}"
                base = {"seed": 123, "task_id": "hellaswag", "item_id": item, "item_cluster": item, "domain": "test", "gold": i % 4, "prompt_hash": item + "hash", "split": split}
                manifest_rows.append(base)
                for m, values in zip(PEERS + (TARGETS[0],), [peers[:, 0], peers[:, 1], peers[:, 2], target]):
                    response_rows.append({"model_id": m, "task_id": "hellaswag", "item_id": item, "prompt_hash": item + "hash", "revision": "synthetic", **{f"coord_{j}": float(values[i, j]) for j in range(3)}})
        manifest, responses = pd.DataFrame(manifest_rows), pd.DataFrame(response_rows)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            # The fit stage must succeed without any test responses available.
            no_test = responses[~responses.item_id.str.startswith("iid_test")]
            path = fit_cell(root, manifest, no_test, 123, "hellaswag", TARGETS[0], "synthetic_contract")
            freeze = validate_freeze(path, "synthetic_contract")
            self.assertFalse((path.parent / "item_losses.parquet").exists())
            self.assertFalse(freeze["test_evaluated"])
            policies = [r for r in freeze["selections"] if r["selection_kind"] == "policy"]
            self.assertEqual(len(policies), 2 * 3 * len(BUDGETS))
            for row in policies:
                self.assertLessEqual(row["actual_bits"], row["budget_bits"])
            before = sha(path)
            evaluate_cell(path, manifest, responses, "synthetic_contract", bootstrap_draws=100)
            self.assertEqual(before, sha(path))
            evaluate_cell(path, manifest, responses, "synthetic_contract", bootstrap_draws=100)
            gains = pd.read_csv(path.parent / "paired_gains.csv")
            self.assertEqual(len(gains), 2 * 2 * len(BUDGETS))
            direct = pd.read_csv(path.parent / "paired_direct_contrasts.csv")
            self.assertEqual(len(direct), 69)
            self.assertEqual(set(direct.comparison), {"all_q2_peer_pairs", "nnls_minus_krr_same_peers", "validation_selected_nnls_minus_krr_q2"})
            self.assertTrue((path.parent / "evaluation_complete.json").exists())
            binary = next((path.parent / "states").glob("*.bin"))
            binary.write_bytes(binary.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "modified"):
                validate_freeze(path, "synthetic_contract")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        unittest.main()
