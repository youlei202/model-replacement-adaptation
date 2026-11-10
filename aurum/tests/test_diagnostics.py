import copy
import unittest

import torch

from aurum.diagnostics import (feature_conditioning, learning_summary,
                                   per_query_basis_oracle, tiny_fit_diagnostic)
from aurum.models import AnchoredResidual, FrozenAnchor
from aurum.train import ArrayData


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.peers = torch.randn(32, 2, 23) * .5
        self.features = torch.randn(32, 8)
        self.basis = torch.linalg.qr(torch.randn(23, 4)).Q
        self.anchor = FrozenAnchor(torch.tensor([.4, .6]))
        self.coefficients = self.features @ torch.randn(8, 4) * .3
        self.target = self.anchor.logits(self.peers) + self.coefficients @ self.basis.T
        self.data = ArrayData(self.peers, self.features, self.target, role="inner_train")

    def test_tiny_fit_exposes_learnability_without_mutating_candidate(self):
        model = AnchoredResidual(self.anchor, self.basis, 8)
        before = copy.deepcopy(model.state_dict())
        result = tiny_fit_diagnostic(model, self.data, n_examples=16, steps=100, lr=.04)
        self.assertLess(result["best"]["tv"], result["before"]["tv"] * .1)
        self.assertEqual(result["n_examples"], 16)
        self.assertFalse(result["deployable_candidate"])
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_oracle_demonstrates_representability_and_marks_teacher_access(self):
        result = per_query_basis_oracle(self.anchor, self.basis, self.data, n_examples=16,
                                        steps=100, lr=.1, include_coefficients=True)
        self.assertLess(result["best"]["tv"], result["before"]["tv"] * .1)
        self.assertFalse(result["certified_lower_bound"])
        self.assertFalse(result["deployable_candidate"])
        self.assertTrue(result["target_used_for_each_query"])
        self.assertEqual(len(result["best_fit_coefficients"]), 16)
        self.assertGreater(result["first_gradient_norms"]["coefficients"], 0.)

    def test_diagnostics_refuse_validation_and_confirmation(self):
        model = AnchoredResidual(self.anchor, self.basis, 8)
        for role in ("validation", "outer_validation", "confirmation", "test", "development"):
            data = ArrayData(self.peers, self.features, self.target, role=role)
            with self.assertRaises(ValueError):
                tiny_fit_diagnostic(model, data)
            with self.assertRaises(ValueError):
                per_query_basis_oracle(self.anchor, self.basis, data)

    def test_feature_spectrum_identifies_duplicate_and_constant_features(self):
        x = torch.cat([self.features[:, :2], self.features[:, :1], torch.ones(32, 1)], dim=1)
        result = feature_conditioning(x)
        self.assertEqual(result["centered_effective_rank"], 2)
        self.assertEqual(result["constant_columns"], 1)
        self.assertIsNone(result["centered_gram_condition_number"])
        self.assertTrue(result["rank_deficient_at_tolerance"])

    def test_learning_summary_distinguishes_fit_from_stopping_gain(self):
        result = learning_summary({"history": [{"inner_stop_tv": .3}, {"inner_stop_tv": .32}],
                                   "chosen_epoch": 0, "completed_epochs": 1,
                                   "selected_fit_means": {"tv": .1}, "inner_stop_tv_after": .3,
                                   "first_step_gradient_norms": {"map": .2, "factor": 0.}},
                                  anchor_fit_tv=.2, anchor_stop_tv=.3, outer_tv=.31)
        self.assertEqual(result["first_gradient_nonzero_tensors"], 1)
        self.assertEqual(result["fit_gain_over_anchor"], .1)
        self.assertEqual(result["inner_stop_gain_over_anchor"], 0.)
        self.assertTrue(result["initial_checkpoint_retained"])


if __name__ == "__main__":
    unittest.main()
