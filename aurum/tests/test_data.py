"""Tests for data-selection boundaries and exact legacy anchor refitting."""
import numpy as np
import pytest

from aurum.data import RichDevelopmentCache, fit_legacy_weights


def test_refit_anchor_uses_only_explicit_inner_training_rows():
    rng = np.random.default_rng(123)
    peers = (rng.normal(size=(19, 13)).astype('f4'), rng.normal(size=(19, 13)).astype('f4'))
    target = (0.25 * peers[0] + 0.75 * peers[1]).astype('f4')
    train_indices = np.arange(0, 19, 2)
    first = fit_legacy_weights(peers, target, train_indices, batch_size=3)
    target[1::2] += 1000
    second = fit_legacy_weights(peers, target, train_indices, batch_size=7)
    np.testing.assert_allclose(first['convex'], [0.25, 0.75], atol=1e-6)
    np.testing.assert_allclose(first['nnls'], [0.25, 0.75], atol=1e-6)
    np.testing.assert_array_equal(first['convex'], second['convex'])
    np.testing.assert_array_equal(first['nnls'], second['nnls'])
    assert first['training_indices_sha256'] == second['training_indices_sha256']


def test_development_bridge_rejects_test_roles_before_opening_data():
    for role in ('test', 'test_pool', 'pilot', 'confirmation'):
        with pytest.raises(ValueError, match='permits fit and validation only'):
            RichDevelopmentCache._role(role)


def test_refit_anchor_rejects_duplicate_rows():
    peers = (np.eye(3, dtype='f4'), np.ones((3, 3), dtype='f4'))
    with pytest.raises(ValueError, match='Invalid training indices'):
        fit_legacy_weights(peers, peers[0], [0, 0])
