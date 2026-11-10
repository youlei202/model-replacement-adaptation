"""Scientific invariants for the AURUM anchor/loss intervention.

Run with the existing torch environment (no test dependency installation):
  .venv-inference/bin/python -m unittest aurum.tests.test_modeling -v
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aurum.codec import ARRAY, HEADER, decode, encode, export_model
from aurum.inference import DecodedPredictor, predict_decoded
from aurum.losses import objective, per_item, projected_objective
from aurum.models import (
    AnchoredMLP, AnchoredResidual, AnchoredResponseGate, FrozenAnchor,
    TrainableCalibration,
)
from aurum.train import ArrayData, TrainConfig, evaluate, fit


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(924)
        self.peers = torch.randn(12, 2, 23)
        self.features = torch.randn(12, 6)
        self.basis = torch.linalg.qr(torch.randn(23, 4)).Q
        self.target = torch.randn(12, 23)

    def make_models(self, kind="logit"):
        anchor = FrozenAnchor(torch.tensor([.3, .8]), kind)
        return [
            anchor,
            AnchoredResidual(copy.deepcopy(anchor), self.basis, 6),
            AnchoredResidual(copy.deepcopy(anchor), self.basis, 6, factor_rank=2),
            AnchoredMLP(copy.deepcopy(anchor), self.basis, 6, hidden=5),
            AnchoredResponseGate(copy.deepcopy(anchor), 6, strength=.7),
        ]

    def test_zero_is_exact_anchor_and_first_gradient_is_live(self):
        for kind in ("logit", "probability"):
            models = self.make_models(kind)
            expected = models[0](self.peers)
            for model in models[1:]:
                with self.subTest(kind=kind, family=type(model).__name__):
                    actual = model(self.peers, self.features)
                    self.assertTrue(torch.equal(actual, expected))
                    objective(self.target, actual, "kl").backward()
                    live = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
                    self.assertGreater(live, 0.)
                    if isinstance(model, AnchoredResidual) and model.factor_rank:
                        self.assertEqual(float(model.down.grad.abs().sum()), 0.)
                        self.assertGreater(float(model.up.grad.abs().sum()), 0.)

    def test_exact_full_support_forward_kl_and_teacher_detachment(self):
        teacher = self.target.clone().requires_grad_()
        predicted = self.peers[:, 0].clone().requires_grad_()
        log_q = predicted.log_softmax(-1)
        actual = objective(teacher, log_q, "kl")
        expected = torch.nn.functional.kl_div(log_q, teacher.detach().softmax(-1), reduction="batchmean")
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)
        # Probability mass in every vocabulary entry contributes, including tail.
        tail_q = log_q.detach().clone()
        tail_q[:, -1] -= .2
        self.assertGreater(float(objective(teacher, tail_q, "kl")), float(actual))
        for name in ("tv", "tv_kl"):
            self.assertTrue(torch.isfinite(objective(teacher, log_q, name)))

    def test_nonorthogonal_projected_objective_matches_materialized_map(self):
        basis = self.basis * torch.tensor([.9, 1.1, .4, 2.])
        model = AnchoredResidual(FrozenAnchor(torch.tensor([.3, .7])), basis, 6)
        with torch.no_grad():
            model.linear.normal_(std=.1)
        expected = ((self.target - model.logits(self.peers, self.features)) @ basis).square().mean()
        torch.testing.assert_close(projected_objective(model, self.peers, self.features, self.target), expected)

    def test_codec_roundtrip_and_actual_state_accounting_all_families(self):
        for kind in ("logit", "probability"):
            for model in self.make_models(kind):
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(torch.randn_like(parameter) * .03)
                for codec in ("fp32", "fp16", "int8", "int4"):
                    with self.subTest(kind=kind, family=type(model).__name__, codec=codec):
                        blob = export_model(model, (0, 2), codec=codec, basis_id=17, vocab=23, retired_target=4)
                        state = decode(blob)
                        actual = predict_decoded(state, self.peers, self.features, self.basis)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(12))
                        if codec == "fp32":
                            torch.testing.assert_close(actual, model(self.peers, self.features), atol=1e-6, rtol=1e-6)
                        self.assertEqual(state["actual_bytes"], len(blob))
                        self.assertEqual(state["actual_bits"], 8 * len(blob))
                        n = sum(a.size for a in state["arrays"])
                        if codec == "fp32":
                            self.assertEqual(len(blob), HEADER.size + len(state["arrays"]) * ARRAY.size + 4 * n)
                        if not isinstance(model, FrozenAnchor):
                            self.assertEqual(state["basis_id"], 17)
        cheap = export_model(self.make_models()[0], (0, 2), vocab=23)
        residual = export_model(self.make_models()[1], (0, 2))
        self.assertLess(len(cheap), len(residual))

    def test_corruption_and_deployment_scope_fail_explicitly(self):
        model = self.make_models()[1]
        with self.assertRaises(ValueError):
            export_model(model, (0, 2), retired_target=2)
        blob = export_model(model, (0, 2))
        for corrupted in (blob[:8], blob[:-1], blob + b"x", b"NOPE" + blob[4:]):
            with self.assertRaises(ValueError):
                decode(corrupted)
        with self.assertRaises(ValueError):
            predict_decoded(decode(blob), self.peers[:, :1], self.features, self.basis)
        with self.assertRaises(ValueError):
            predict_decoded(decode(blob), self.peers, self.features[:, :-1], self.basis)
        with self.assertRaises(ValueError):
            predict_decoded(decode(blob), self.peers, self.features, self.basis[:, :-1])
        with self.assertRaises(ValueError):
            encode("anchor_logit", (2, 0), [np.array([1., 0.])])

    def test_readonly_memmaps_alignment_and_decoded_batch_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "peers.npy"
            np.save(path, self.peers.numpy())
            peers = np.load(path, mmap_mode="r")
            data = ArrayData(peers, self.features.numpy(), self.target.numpy(), indices=[9, 3, 0], role="validation")
            model = self.make_models()[1]
            state = decode(export_model(model, (0, 2)))
            replay = DecodedPredictor(state, self.basis)
            small_batch = evaluate(replay, data, batch_size=1)
            whole_batch = evaluate(replay, data, batch_size=3)
            np.testing.assert_array_equal(small_batch["source_row_indices"], [9, 3, 0])
            np.testing.assert_allclose(small_batch["per_item"]["tv"], whole_batch["per_item"]["tv"], atol=1e-6)
            np.testing.assert_array_equal(peers, self.peers.numpy())
            self.assertFalse(peers.flags.writeable)

    def test_fitting_refuses_outer_stopping_and_overlapping_rows(self):
        data = ArrayData(self.peers, self.features, self.target)
        model = self.make_models()[1]
        with self.assertRaises(ValueError):
            fit(model, data.subset(range(6)), data.subset(range(6, 12), role="validation"))
        with self.assertRaises(ValueError):
            fit(model, data.subset(range(7)), data.subset(range(6, 12), role="inner_stop"))

    def test_linear_probability_fit_learns_heldout_signal_and_keeps_history(self):
        n = 160
        peers = torch.randn(n, 2, 23) * .5
        features = torch.randn(n, 6)
        anchor = FrozenAnchor(torch.tensor([.3, .7]))
        planted = torch.randn(6, 4) * .5
        target = anchor.logits(peers) + (features @ planted) @ self.basis.T
        data = ArrayData(peers, features, target)
        train = data.subset(range(110), role="inner_train")
        stop = data.subset(range(110, 160), role="inner_stop")
        model = AnchoredResidual(anchor, self.basis, 6)
        initial = evaluate(model, stop)["tv"]
        visited = []
        result = fit(model, train, stop, TrainConfig(epochs=60, lr=.04, batch_size=22, patience=12),
                     callback=lambda epoch, row, _model, best: visited.append((epoch, best)))
        self.assertLess(evaluate(model, stop)["tv"], initial * .1)
        self.assertEqual(len(visited), len(result["history"]))
        self.assertEqual(result["history"][0]["epoch"], 0)
        self.assertGreater(result["first_step_gradient_norms"]["linear"], 0.)
        self.assertEqual(evaluate(model, stop)["tv"], result["inner_stop_tv_after"])
        self.assertTrue(result["public_buffers_not_copied_into_target_checkpoint"])

    def test_cheap_probability_calibration_gets_same_optimization_support(self):
        peers = torch.randn(100, 2, 23)
        truth = FrozenAnchor(torch.tensor([.75, .25]), "probability")
        target = truth(peers)
        data = ArrayData(peers, None, target)
        model = TrainableCalibration(torch.tensor([.1, .9]), "probability", temperature=False)
        train, stop = data.subset(range(70)), data.subset(range(70, 100), role="inner_stop")
        start = evaluate(model, stop)["tv"]
        result = fit(model, train, stop, TrainConfig(epochs=60, lr=.08, batch_size=35, patience=10))
        self.assertLess(result["inner_stop_tv_after"], .1 * start)
        decoded = decode(export_model(model, (0, 2), vocab=23))
        torch.testing.assert_close(predict_decoded(decoded, peers), model(peers), atol=1e-6, rtol=1e-6)

    def test_exact_anchor_remains_available_after_training(self):
        anchor = FrozenAnchor(torch.tensor([.3, .7]))
        target = anchor.logits(self.peers)
        data = ArrayData(self.peers, self.features, target)
        model = AnchoredResidual(anchor, self.basis, 6)
        result = fit(model, data.subset(range(8)), data.subset(range(8, 12), role="inner_stop"),
                     TrainConfig(epochs=2, lr=.1, batch_size=4))
        self.assertEqual(result["chosen_epoch"], 0)
        self.assertTrue(torch.equal(model(self.peers, self.features), anchor(self.peers)))


if __name__ == "__main__":
    unittest.main()
