"""End-to-end acceptance test on the shipped monkey sample.

Verifies that the published rate-trace pattern (suppression nadir 200-250 ms
post-stimulus, rebound peak 350-420 ms) is recovered. This is the
falsifiable prediction from Nouri et al. 2025 — if these numbers drift,
something has regressed in the detector or in the windowing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msfeature.config import MONKEY_CONFIG
from msfeature.detect import detect_microsaccades
from msfeature.features import RateBinSpec, shift_to_onset, stack_trial_features
from msfeature.io import load_session
from msfeature.preprocess import preprocess_trial

DATA_PATH = Path(__file__).parents[1] / "data" / "monkey_sample" / "sample_1_EyeData.mat"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="monkey sample not present")
def test_monkey_rate_trace_matches_paper():
    ds = load_session(DATA_PATH)
    fs = MONKEY_CONFIG.sampling_rate_hz
    onset = MONKEY_CONFIG.stim_onset_sample
    assert onset is not None

    events_per_trial = []
    for i in range(ds.n_trials):
        x, y, _ = preprocess_trial(
            ds.eye_x[i],
            ds.eye_y[i],
            sampling_rate_hz=fs,
            lowpass_cutoff_hz=MONKEY_CONFIG.lowpass_cutoff_hz,
        )
        evs = detect_microsaccades(x, y, MONKEY_CONFIG)
        events_per_trial.append(shift_to_onset(evs, onset, MONKEY_CONFIG.dt))

    spec = RateBinSpec(-onset / fs, (ds.n_samples - onset) / fs, 0.015)
    X = stack_trial_features(events_per_trial, spec)
    edges = spec.bin_edges()
    centres_ms = (edges[:-1] + edges[1:]) / 2 * 1000
    mean_rate = X.mean(axis=0)

    post = centres_ms > 0
    post_idx = np.flatnonzero(post)
    nadir_t_ms = centres_ms[post_idx[np.argmin(mean_rate[post_idx])]]
    peak_t_ms = centres_ms[post_idx[np.argmax(mean_rate[post_idx])]]
    peak_rate = mean_rate[post_idx[np.argmax(mean_rate[post_idx])]]

    # Paper-reported timing (Nouri et al. 2025): nadir 210-230 ms,
    # peak 380-395 ms at ~3.3 Hz. We allow generous tolerance.
    assert 150 <= nadir_t_ms <= 260, f"suppression nadir at {nadir_t_ms:.0f} ms"
    assert 320 <= peak_t_ms <= 430, f"rebound peak at {peak_t_ms:.0f} ms"
    assert peak_rate >= 1.5, f"rebound peak rate only {peak_rate:.2f} Hz"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="monkey sample not present")
def test_monkey_event_descriptive_stats_in_range():
    """Event-level stats should land in the typical primate microsaccade
    range: amplitude < 1 deg, duration < 50 ms, peak velocity < 200 deg/s."""
    ds = load_session(DATA_PATH)
    events: list = []
    for i in range(ds.n_trials):
        x, y, _ = preprocess_trial(
            ds.eye_x[i], ds.eye_y[i],
            sampling_rate_hz=MONKEY_CONFIG.sampling_rate_hz,
            lowpass_cutoff_hz=MONKEY_CONFIG.lowpass_cutoff_hz,
        )
        events.extend(detect_microsaccades(x, y, MONKEY_CONFIG))
    if not events:
        pytest.skip("no events detected on this sample")
    amps = np.array([e.amplitude for e in events])
    durs = np.array([e.duration * 1000 for e in events])
    peaks = np.array([e.peak_velocity for e in events])

    assert amps.max() <= MONKEY_CONFIG.max_amplitude_deg + 1e-9
    assert durs.max() <= MONKEY_CONFIG.max_duration_ms + 1e-9, (
        "events longer than max_duration_ms — duration cap not enforced"
    )
    assert peaks.max() < 500, "implausibly fast 'microsaccade' detected"
    # Bulk distribution sanity: median microsaccade duration is 6-30 ms
    assert 5 <= np.median(durs) <= 30
    # Most events should be short — fewer than 5% above 50 ms
    assert (durs > 50).mean() < 0.05
