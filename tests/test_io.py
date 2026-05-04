"""Loader robustness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from msfeature.io import load_session


def _write_sample_mat(path: Path, *, n_trials: int = 4, n_samples: int = 50) -> None:
    sess = {
        "EyeX": np.zeros((n_trials, n_samples)),
        "EyeY": np.zeros((n_trials, n_samples)),
        "trial_isface": np.array([[i % 2] for i in range(n_trials)], dtype=np.uint8),
        "trial_stimulus_code": np.array(
            [[i + 1] for i in range(n_trials)], dtype=np.uint8
        ),
    }
    savemat(str(path), {"sess": sess})


def test_load_round_trips(tmp_path):
    path = tmp_path / "ok.mat"
    _write_sample_mat(path, n_trials=4, n_samples=50)
    ds = load_session(path)
    assert ds.n_trials == 4 and ds.n_samples == 50
    np.testing.assert_array_equal(ds.is_face, [False, True, False, True])
    np.testing.assert_array_equal(ds.stimulus_code, [1, 2, 3, 4])


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "does_not_exist.mat")


def test_load_missing_sess(tmp_path):
    path = tmp_path / "no_sess.mat"
    savemat(str(path), {"other": np.zeros(3)})
    with pytest.raises(KeyError, match="no 'sess'"):
        load_session(path)


def test_load_missing_field_lists_what_is_missing(tmp_path):
    path = tmp_path / "missing_field.mat"
    savemat(str(path), {"sess": {"EyeX": np.zeros((2, 5)), "EyeY": np.zeros((2, 5))}})
    with pytest.raises(AttributeError) as exc:
        load_session(path)
    msg = str(exc.value)
    assert "trial_isface" in msg or "trial_stimulus_code" in msg


def test_load_shape_mismatch_raises(tmp_path):
    path = tmp_path / "mismatched.mat"
    sess = {
        "EyeX": np.zeros((4, 50)),
        "EyeY": np.zeros((4, 50)),
        "trial_isface": np.zeros((3, 1), dtype=np.uint8),  # wrong trial count
        "trial_stimulus_code": np.ones((4, 1), dtype=np.uint8),
    }
    savemat(str(path), {"sess": sess})
    with pytest.raises(ValueError, match="trial-count"):
        load_session(path)
