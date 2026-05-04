"""Feature extraction: binned rate and trial descriptors."""

from __future__ import annotations

import numpy as np

from msfeature.events import Microsaccade
from msfeature.features import (
    NOURI_RATE_BINS,
    RateBinSpec,
    binned_rate,
    shift_to_onset,
    stack_trial_features,
    trial_descriptors,
)


def _ev(start_time, amp=0.3, peak=30.0, direction=0.0, duration=0.012):
    return Microsaccade(
        start_idx=int(start_time * 2000),
        end_idx=int((start_time + duration) * 2000),
        start_time=start_time,
        end_time=start_time + duration,
        duration=duration,
        amplitude=amp,
        peak_velocity=peak,
        direction=direction,
    )


def test_nouri_window_has_correct_bin_count():
    # 100 ms to 600 ms, 15 ms bins -> floor(500 / 15) = 33 bins
    assert NOURI_RATE_BINS.n_bins == 33


def test_binned_rate_counts_event_starts_and_normalises():
    spec = RateBinSpec(window_start_s=0.0, window_end_s=0.060, bin_width_s=0.015)
    events = [_ev(0.005), _ev(0.020), _ev(0.022), _ev(0.050)]
    rates = binned_rate(events, spec)
    # Bin 0 [0, 15): 1 event ; Bin 1 [15, 30): 2 events
    # Bin 2 [30, 45): 0 ; Bin 3 [45, 60): 1
    expected_counts = np.array([1, 2, 0, 1])
    np.testing.assert_allclose(rates, expected_counts / 0.015)


def test_binned_rate_no_events():
    spec = RateBinSpec(0.0, 0.030, 0.015)
    rates = binned_rate([], spec)
    np.testing.assert_array_equal(rates, np.zeros(2))


def test_from_trial_length_clamps_to_whole_bins():
    spec = RateBinSpec.from_trial_length(
        n_samples=900, sampling_rate_hz=2000.0, bin_width_s=0.015
    )
    # 900 / 2000 = 0.45 s; floor(0.45 / 0.015) = 30 bins
    assert spec.window_start_s == 0.0
    assert spec.n_bins == 30
    assert np.isclose(spec.window_end_s, 0.45)


def test_trial_descriptors_empty():
    d = trial_descriptors([])
    assert d["n_events"] == 0
    assert np.isnan(d["mean_amplitude"])


def test_trial_descriptors_uses_unit_vector_for_direction():
    # Two events at +pi/2 and -pi/2 should average to zero, not zero arithmetic
    events = [_ev(0.1, direction=np.pi / 2), _ev(0.2, direction=-np.pi / 2)]
    d = trial_descriptors(events)
    assert np.isclose(d["mean_direction_x"], 0.0)
    assert np.isclose(d["mean_direction_y"], 0.0)


def test_stack_trial_features_shape():
    spec = RateBinSpec(0.0, 0.060, 0.015)
    per_trial = [[_ev(0.01)], [], [_ev(0.04), _ev(0.05)]]
    X = stack_trial_features(per_trial, spec)
    assert X.shape == (3, 4)


def test_stack_trial_features_handles_empty_input():
    spec = RateBinSpec(0.0, 0.060, 0.015)
    X = stack_trial_features([], spec)
    assert X.shape == (0, spec.n_bins)


def test_rate_bin_spec_rejects_invalid_inputs():
    import pytest
    with pytest.raises(ValueError):
        RateBinSpec(0.0, 0.0, 0.015)
    with pytest.raises(ValueError):
        RateBinSpec(0.0, 0.060, 0.0)
    with pytest.raises(ValueError):
        RateBinSpec(0.6, 0.1, 0.015)


def test_shift_to_onset_makes_pre_stim_negative():
    """Shifting an event at sample 200 by onset_sample=300 at 1 kHz puts
    its start_idx at -100 and start_time at -0.100 s."""
    ev = _ev(start_time=0.200, duration=0.012)
    out = shift_to_onset([ev], onset_sample=300, dt=1e-3)
    assert out[0].start_idx == ev.start_idx - 300
    assert np.isclose(out[0].start_time, ev.start_time - 0.300)
    assert np.isclose(out[0].end_time, ev.end_time - 0.300)
    # duration / amplitude / etc unchanged
    assert out[0].duration == ev.duration
    assert out[0].amplitude == ev.amplitude


def test_shift_to_onset_empty():
    assert shift_to_onset([], onset_sample=100, dt=1e-3) == []
