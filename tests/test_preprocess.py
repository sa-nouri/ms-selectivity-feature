"""Preprocessing: blink masking, glitch removal, drift correction, low-pass."""

from __future__ import annotations

import numpy as np

from msfeature.preprocess import (
    correct_drift,
    lowpass,
    mark_blinks,
    preprocess_trial,
    remove_glitches,
)


def test_mark_blinks_flags_nan_only_by_default():
    x = np.array([0.0, 1.0, np.nan, 2.0])
    y = np.array([0.0, np.nan, 0.0, 0.0])
    np.testing.assert_array_equal(mark_blinks(x, y), [False, True, True, False])


def test_mark_blinks_with_pupil_loss_value():
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0])
    mask = mark_blinks(x, y, pupil_loss_value=0.0, nan_as_blink=False)
    np.testing.assert_array_equal(mask, [True, True, False])


def test_remove_glitches_replaces_short_spikes():
    n = 50
    x = np.zeros(n)
    y = np.zeros(n)
    x[10] = 5.0  # one-sample spike
    cleaned_x, _ = remove_glitches(x, y, z_threshold=3.0)
    assert abs(cleaned_x[10]) < 1.0


def test_remove_glitches_preserves_real_movements():
    """A 50-sample step should NOT be flagged as a glitch."""
    n = 100
    x = np.zeros(n)
    y = np.zeros(n)
    x[40:] = 1.0  # large but persistent step (real saccade)
    cleaned_x, _ = remove_glitches(x, y, max_run_samples=2)
    # The end of the trace should still be at ~1.0
    assert cleaned_x[-1] > 0.95


def test_remove_glitches_uses_robust_scale():
    """A few large glitches should not raise the scale enough to hide
    themselves. With SD-based scale, three large glitches in a 50-sample
    trace inflate sigma so much they all fall below the threshold."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 0.005, n)
    y = rng.normal(0, 0.005, n)
    x[50] = 5.0
    x[100] = -4.0
    x[150] = 6.0
    cleaned_x, _ = remove_glitches(x, y, z_threshold=4.0)
    assert abs(cleaned_x[50]) < 0.5
    assert abs(cleaned_x[100]) < 0.5
    assert abs(cleaned_x[150]) < 0.5


def test_remove_glitches_handles_nan_samples():
    """A NaN sample is treated as a glitch and interpolated over.
    Uses noisy fixational data — on perfectly noise-free input MAD/SD
    of the diff signal collapses to ~0 and any non-zero diff looks like
    a glitch, but real eye-tracking traces always carry sample noise."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0.0, 0.005, n)
    y = rng.normal(0.0, 0.005, n)
    x[100] = np.nan
    cleaned_x, _ = remove_glitches(x, y)
    assert np.isfinite(cleaned_x[100])


def test_remove_glitches_rejects_shape_mismatch():
    import pytest
    with pytest.raises(ValueError):
        remove_glitches(np.zeros(10), np.zeros(11))


def test_correct_drift_removes_linear_in_time_trend():
    fs = 1000.0
    n = 1000
    t = np.arange(n) / fs
    x = 0.5 * t + np.sin(t * 6)
    y = -0.3 * t + np.cos(t * 6)
    cx, cy = correct_drift(x, y, dt=1 / fs)
    sx = np.polyfit(t, cx, 1)[0]
    sy = np.polyfit(t, cy, 1)[0]
    assert abs(sx) < 1e-9
    assert abs(sy) < 1e-9


def test_correct_drift_handles_short_input():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    cx, cy = correct_drift(x, y, dt=1e-3)
    np.testing.assert_array_equal(cx, x)
    np.testing.assert_array_equal(cy, y)


def test_lowpass_attenuates_high_frequency():
    fs = 2000.0
    t = np.arange(2000) / fs
    high = np.sin(2 * np.pi * 500 * t)
    low = np.sin(2 * np.pi * 5 * t)
    x = high + low
    y = np.zeros_like(x)
    fx, _ = lowpass(x, y, sampling_rate_hz=fs, cutoff_hz=50.0, order=2)
    assert np.std(fx) < np.std(x)
    assert np.std(fx) > 0.5  # slow component preserved


def test_lowpass_skips_nan_input():
    """filtfilt would propagate NaN globally; the wrapper should pass
    through unchanged so downstream code can handle the NaN."""
    n = 1000
    x = np.zeros(n)
    y = np.zeros(n)
    x[500] = np.nan
    fx, fy = lowpass(x, y, sampling_rate_hz=2000.0, cutoff_hz=200.0)
    assert np.isnan(fx[500])
    assert np.all(fy == 0)


def test_preprocess_trial_returns_blink_mask_from_original():
    x = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.zeros_like(x)
    cx, cy, mask = preprocess_trial(x, y, sampling_rate_hz=1000.0)
    np.testing.assert_array_equal(mask, [False, False, True, False] + [False] * 4)
    assert cx.shape == x.shape and cy.shape == y.shape
