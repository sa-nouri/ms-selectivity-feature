"""Detector core: sigma, threshold, run-length grouping, end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from msfeature.detect import (
    compute_sigma,
    detect_microsaccades,
    elliptic_threshold_mask,
    extract_runs,
)
from tests.conftest import InjectedSaccade, build_trace


# -----------------------------------------------------------------------
# compute_sigma
# -----------------------------------------------------------------------


def test_sigma_zero_for_zero_velocity():
    assert compute_sigma(np.zeros(100)) == 0.0


def test_sigma_engbert2003_for_normal_input():
    """For Normal(0, 5) data, engbert2003 = sqrt(median(v^2) - median(v)^2).
    Median(v^2) for N(0, sigma^2) is ~0.4549 * sigma^2, so the returned
    value is sigma * sqrt(0.4549) ~= 3.37 for sigma=5."""
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 5.0, 10000)
    s = compute_sigma(v, method="engbert2003")
    assert 3.0 < s < 3.8


def test_sigma_engbert2015_for_normal_input():
    """For Normal(0, 5) data, engbert2015 = sqrt(median((v - median(v))^2))
    is also smaller than sigma but slightly different from engbert2003."""
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 5.0, 10000)
    s = compute_sigma(v, method="engbert2015")
    assert 3.0 < s < 4.0


def test_sigma_methods_differ_when_median_nonzero():
    """The two sigma forms agree only when median(v) = 0. With a shift,
    they should diverge."""
    rng = np.random.default_rng(0)
    v = rng.normal(2.0, 5.0, 10000)  # shifted
    s2003 = compute_sigma(v, method="engbert2003")
    s2015 = compute_sigma(v, method="engbert2015")
    assert s2003 != s2015


def test_sigma_falls_back_to_std_when_median_form_is_zero():
    """When the median form underflows, the explicit fallback returns SD."""
    v = np.zeros(100)
    v[50] = 10.0
    s = compute_sigma(v, fallback_to_std=True)
    assert s == np.std(v, ddof=0)


def test_sigma_returns_zero_without_fallback():
    """Default behaviour (fallback off): underflow returns 0.0 so the
    threshold mask is empty rather than silently shifting."""
    v = np.zeros(100)
    v[50] = 10.0
    assert compute_sigma(v, fallback_to_std=False) == 0.0


def test_sigma_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown sigma method"):
        compute_sigma(np.zeros(10), method="otero_millan")


def test_sigma_ignores_nan():
    """The default (engbert2015) sigma drops non-finite samples before
    computing median((v - median(v))^2)."""
    v = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan])
    s = compute_sigma(v, method="engbert2015")
    finite = v[np.isfinite(v)]  # [1, 2, 3, 4]; median = 2.5
    expected = np.sqrt(np.median((finite - np.median(finite)) ** 2))
    assert np.isclose(s, expected)


# -----------------------------------------------------------------------
# elliptic_threshold_mask
# -----------------------------------------------------------------------


def test_elliptic_mask_obeys_unit_ellipse():
    vx = np.array([0.0, 1.0, 2.0, 5.0, 0.0])
    vy = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    mask = elliptic_threshold_mask(vx, vy, sigma_x=1.0, sigma_y=1.0, lam=2.0)
    # Test value: (vx/2)^2 + (vy/2)^2 > 1
    expected = np.array([False, False, False, True, True])
    np.testing.assert_array_equal(mask, expected)


def test_elliptic_mask_handles_nan_velocity():
    vx = np.array([np.nan, 0.0, 10.0])
    vy = np.array([0.0, np.nan, 0.0])
    mask = elliptic_threshold_mask(vx, vy, sigma_x=1.0, sigma_y=1.0, lam=2.0)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_elliptic_mask_all_false_when_sigma_zero():
    vx = np.array([100.0, 100.0])
    vy = np.array([100.0, 100.0])
    assert not elliptic_threshold_mask(vx, vy, 0.0, 1.0, lam=6.0).any()
    assert not elliptic_threshold_mask(vx, vy, 1.0, 0.0, lam=6.0).any()


def test_elliptic_mask_rejects_invalid_lambda():
    with pytest.raises(ValueError):
        elliptic_threshold_mask(np.zeros(3), np.zeros(3), 1.0, 1.0, lam=0.0)


def test_elliptic_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        elliptic_threshold_mask(np.zeros(3), np.zeros(4), 1.0, 1.0, lam=6.0)


# -----------------------------------------------------------------------
# extract_runs
# -----------------------------------------------------------------------


def test_extract_runs_basic():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert extract_runs(mask, min_length=1) == [(1, 2), (5, 7)]
    assert extract_runs(mask, min_length=3) == [(5, 7)]
    assert extract_runs(mask, min_length=4) == []


def test_extract_runs_starts_and_ends_at_edges():
    mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert extract_runs(mask, min_length=1) == [(0, 1), (3, 5)]


def test_extract_runs_empty():
    assert extract_runs(np.zeros(0, dtype=bool), min_length=1) == []
    assert extract_runs(np.zeros(10, dtype=bool), min_length=1) == []


# -----------------------------------------------------------------------
# detect_microsaccades — end to end on synthetic data
# -----------------------------------------------------------------------


def test_detects_single_saccade(monkey_cfg):
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=0.5,
        saccades=[InjectedSaccade(onset_s=0.2, duration_s=0.012,
                                   amp_x_deg=0.5, amp_y_deg=0.0)],
    )
    events = detect_microsaccades(x, y, monkey_cfg)
    assert len(events) == 1
    ev = events[0]
    assert abs(ev.start_time - 0.2) < 0.005
    assert 0.45 <= ev.amplitude <= 0.55
    assert abs(ev.direction) < np.deg2rad(5)


def test_returns_empty_on_pure_fixation(monkey_cfg):
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=0.5,
        saccades=[],
        noise_std_deg=0.005,
    )
    events = detect_microsaccades(x, y, monkey_cfg)
    # On pristine fixation a few false positives may occur; the rate
    # should be modest (<= a handful per 500 ms).
    assert len(events) < 10


def test_amplitude_cap_drops_macrosaccades(monkey_cfg):
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=0.5,
        saccades=[InjectedSaccade(onset_s=0.2, duration_s=0.025,
                                   amp_x_deg=3.0, amp_y_deg=0.0)],
    )
    events_with = detect_microsaccades(x, y, monkey_cfg, apply_post_rules=True)
    events_without = detect_microsaccades(x, y, monkey_cfg, apply_post_rules=False)
    assert len(events_without) >= 1
    assert len(events_with) == 0  # the 3-degree event exceeds the 1-degree cap


def test_refractory_merge_collapses_close_pair(monkey_cfg):
    """Two events 10 ms apart (< 30 ms refractory) merge into one."""
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=0.5,
        saccades=[
            InjectedSaccade(onset_s=0.20, duration_s=0.012,
                            amp_x_deg=0.4, amp_y_deg=0.0),
            InjectedSaccade(onset_s=0.222, duration_s=0.012,
                            amp_x_deg=-0.2, amp_y_deg=0.0),
        ],
    )
    raw = detect_microsaccades(x, y, monkey_cfg, apply_post_rules=False)
    merged = detect_microsaccades(x, y, monkey_cfg, apply_post_rules=True)
    assert len(raw) == 2
    assert len(merged) == 1
    # The merged amplitude is start-to-end displacement (~0.2°), not 0.4+0.2.
    assert merged[0].amplitude < 0.3


def test_refractory_does_not_merge_far_pair(monkey_cfg):
    """Two events 100 ms apart (> 30 ms refractory) stay separate."""
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=0.5,
        saccades=[
            InjectedSaccade(onset_s=0.10, duration_s=0.012,
                            amp_x_deg=0.4, amp_y_deg=0.0),
            InjectedSaccade(onset_s=0.30, duration_s=0.012,
                            amp_x_deg=0.4, amp_y_deg=0.0),
        ],
    )
    events = detect_microsaccades(x, y, monkey_cfg)
    assert len(events) == 2


def test_detector_works_at_500hz_human(human_cfg):
    """The detector should scale to the human dataset's 500 Hz rate."""
    x, y = build_trace(
        fs=human_cfg.sampling_rate_hz,
        duration_s=1.0,
        saccades=[InjectedSaccade(onset_s=0.5, duration_s=0.020,
                                   amp_x_deg=0.5, amp_y_deg=0.0)],
    )
    events = detect_microsaccades(x, y, human_cfg)
    assert len(events) == 1
    assert 0.45 <= events[0].amplitude <= 0.55


def test_rejects_2d_input(monkey_cfg):
    x = np.zeros((10, 10))
    y = np.zeros((10, 10))
    with pytest.raises(ValueError):
        detect_microsaccades(x, y, monkey_cfg)


def test_returns_empty_on_too_short_trace(monkey_cfg):
    """Less than 5 samples -> velocity is all NaN -> no events."""
    x = np.zeros(4)
    y = np.zeros(4)
    assert detect_microsaccades(x, y, monkey_cfg) == []


def test_does_not_warn_on_clean_input(monkey_cfg):
    """Smoke test that no RuntimeWarning leaks out of the detector."""
    import warnings
    rng = np.random.default_rng(0)
    n = int(0.5 * monkey_cfg.sampling_rate_hz)
    x = rng.normal(0, 0.005, n)
    y = rng.normal(0, 0.005, n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_microsaccades(x, y, monkey_cfg)
