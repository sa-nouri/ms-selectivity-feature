"""Decoder pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from msfeature.decode import cross_validate, make_classifier


def test_make_classifier_has_scaler_and_svm():
    clf = make_classifier()
    assert "scaler" in clf.named_steps
    assert "svm" in clf.named_steps


def test_cross_validate_recovers_separable_classes():
    """Linearly separable data should give near-perfect accuracy."""
    rng = np.random.default_rng(0)
    n_per = 50
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.1, size=(n_per, 5)),
            rng.normal(loc=2.0, scale=0.1, size=(n_per, 5)),
        ]
    )
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    res = cross_validate(X, y, n_splits=5)
    assert res.mean_score > 0.95


def test_cross_validate_reports_fold_count():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (rng.uniform(size=40) > 0.5).astype(int)
    res = cross_validate(X, y, n_splits=4)
    assert res.fold_scores.shape == (4,)
    assert len(res.n_test_per_fold) == 4


def test_cross_validate_rejects_size_mismatch():
    with pytest.raises(ValueError):
        cross_validate(np.zeros((10, 3)), np.zeros(11))


def test_cross_validate_rejects_single_class():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="2 classes"):
        cross_validate(X, y, n_splits=5)


def test_cross_validate_rejects_too_few_per_class():
    X = np.random.default_rng(0).normal(size=(8, 3))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1])  # only 1 of class 1
    with pytest.raises(ValueError, match="smallest class"):
        cross_validate(X, y, n_splits=5)


def test_cross_validate_rejects_empty_X():
    with pytest.raises(ValueError, match="non-empty"):
        cross_validate(np.zeros((0, 3)), np.zeros(0))
