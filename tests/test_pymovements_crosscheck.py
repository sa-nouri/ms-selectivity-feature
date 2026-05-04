"""Cross-validation against pymovements' Engbert-Kliegl implementation.

These tests are skipped if pymovements is not installed (it's a test-time
dependency only). They feed identical pre-computed velocities into both
detectors and confirm the events match sample-for-sample on synthetic
microsaccades.
"""

from __future__ import annotations

import numpy as np
import pytest

from msfeature.detect import detect_microsaccades
from msfeature.velocity import ek_velocity_2d
from tests.conftest import InjectedSaccade, build_trace

pm = pytest.importorskip("pymovements")


def test_matches_pymovements_on_5_synthetic_events(monkey_cfg):
    saccades = [
        InjectedSaccade(0.10, 0.012, 0.30, 0.00),
        InjectedSaccade(0.30, 0.014, 0.00, 0.45),
        InjectedSaccade(0.50, 0.012, -0.40, 0.20),
        InjectedSaccade(0.70, 0.018, 0.25, -0.30),
        InjectedSaccade(0.90, 0.010, -0.20, -0.20),
    ]
    x, y = build_trace(
        fs=monkey_cfg.sampling_rate_hz,
        duration_s=1.0,
        saccades=saccades,
        seed=42,
    )

    ours = detect_microsaccades(x, y, monkey_cfg, apply_post_rules=False)
    our_onsets = sorted(e.start_idx for e in ours)

    vx, vy = ek_velocity_2d(x, y, monkey_cfg.dt)
    vel = np.column_stack([np.nan_to_num(vx), np.nan_to_num(vy)])
    pm_events = pm.events.microsaccades(
        velocities=vel,
        minimum_duration=monkey_cfg.min_duration_samples,
        threshold="engbert2015",
        threshold_factor=monkey_cfg.velocity_threshold_lambda,
    )
    pm_onsets = sorted(pm_events.frame["onset"].to_list())

    assert len(our_onsets) == len(pm_onsets) == len(saccades)
    for a, b in zip(our_onsets, pm_onsets):
        assert abs(a - b) <= 1, f"onset mismatch: ours={a} pm={b}"
