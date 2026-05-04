"""Post-detection rules: refractory merge, amplitude cap, outlier rejection."""

from __future__ import annotations

import numpy as np

from msfeature.events import (
    Microsaccade,
    cap_amplitude,
    cap_duration,
    merge_refractory,
    recompute_amplitude_direction,
    reject_amplitude_outliers,
)


def _ev(start_idx, end_idx, amp=0.5, peak=30.0, dt=0.5e-3):
    return Microsaccade(
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=start_idx * dt,
        end_time=end_idx * dt,
        duration=(end_idx - start_idx) * dt,
        amplitude=amp,
        peak_velocity=peak,
        direction=0.0,
    )


def test_merge_collapses_within_refractory():
    events = [_ev(100, 120), _ev(125, 145)]  # gap = 4 samples
    merged = merge_refractory(events, refractory_samples=10, dt=0.5e-3)
    assert len(merged) == 1
    assert merged[0].start_idx == 100 and merged[0].end_idx == 145


def test_merge_preserves_far_apart():
    events = [_ev(100, 120), _ev(200, 220)]
    merged = merge_refractory(events, refractory_samples=10, dt=0.5e-3)
    assert len(merged) == 2


def test_merge_carries_max_peak_velocity():
    events = [_ev(100, 120, peak=20.0), _ev(122, 140, peak=80.0)]
    merged = merge_refractory(events, refractory_samples=10, dt=0.5e-3)
    assert merged[0].peak_velocity == 80.0


def test_merge_empty():
    assert merge_refractory([], refractory_samples=10, dt=0.5e-3) == []


def test_merge_does_not_cascade_into_super_event():
    """A long chain of close events must not cascade into a single
    multi-hundred-ms event — `max_merged_duration_s` caps it."""
    dt = 1e-3  # 1 kHz
    # 10 events, each 6 samples long, 5 samples apart -> if cascade-merged
    # we'd get one 105-sample event (~105 ms), which is implausible.
    events = [_ev(i * 11, i * 11 + 5, dt=dt) for i in range(10)]
    merged = merge_refractory(events, refractory_samples=10, dt=dt,
                              max_merged_duration_s=0.030)
    # The cap should split the chain into multiple short merged events,
    # none of them longer than ~30 ms.
    assert len(merged) > 1
    for ev in merged:
        assert ev.duration <= 0.031  # tolerance for endpoint inclusion


def test_recompute_amplitude_direction():
    n = 200
    x = np.zeros(n)
    y = np.zeros(n)
    x[10:20] = 0.3  # 0.3 deg horizontal step
    y[10:20] = 0.4  # 0.4 deg vertical step
    ev = _ev(start_idx=5, end_idx=15)  # spans the step
    out = recompute_amplitude_direction([ev], x, y)
    # Displacement from idx 5 (0,0) to idx 15 (0.3, 0.4) -> amp = 0.5
    assert np.isclose(out[0].amplitude, 0.5)
    assert np.isclose(out[0].direction, np.arctan2(0.4, 0.3))


def test_cap_amplitude_drops_above_threshold():
    events = [_ev(0, 10, amp=0.3), _ev(20, 30, amp=1.5), _ev(40, 50, amp=0.9)]
    out = cap_amplitude(events, max_amplitude=1.0)
    assert [e.amplitude for e in out] == [0.3, 0.9]


def test_outlier_rejection_drops_high_tail():
    amps = [0.10, 0.12, 0.11, 0.13, 0.10, 0.12, 0.11, 0.13, 1.5]  # last is the outlier
    events = [_ev(i * 100, i * 100 + 10, amp=a) for i, a in enumerate(amps)]
    out = reject_amplitude_outliers(events, n_sd=2.0)
    assert [round(e.amplitude, 2) for e in out] == amps[:-1]


def test_cap_duration_drops_long_events():
    short = _ev(0, 10)        # 0.005 s at 0.5 ms dt
    medium = _ev(0, 30)       # 0.015 s
    long_ = _ev(0, 300)       # 0.150 s
    out = cap_duration([short, medium, long_], max_duration_s=0.050)
    assert [e.duration for e in out] == [short.duration, medium.duration]


def test_outlier_rejection_no_op_on_uniform():
    events = [_ev(i * 100, i * 100 + 10, amp=0.5) for i in range(5)]
    out = reject_amplitude_outliers(events, n_sd=2.0)
    assert len(out) == 5
