"""Shared pytest fixtures and synthetic-trace builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from msfeature.config import DatasetConfig


@dataclass
class InjectedSaccade:
    onset_s: float
    duration_s: float
    amp_x_deg: float
    amp_y_deg: float


def build_trace(
    fs: float,
    duration_s: float,
    saccades: list[InjectedSaccade],
    noise_std_deg: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic fixation trace with injected linear-ramp saccades.

    The drift to amp_{x,y} during the saccade then *persists* afterwards,
    matching how a real saccade displaces gaze position.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    x = rng.normal(0.0, noise_std_deg, n)
    y = rng.normal(0.0, noise_std_deg, n)
    for s in saccades:
        on = int(round(s.onset_s * fs))
        dur = max(2, int(round(s.duration_s * fs)))
        end = on + dur
        ramp = np.linspace(0.0, 1.0, dur)
        x[on:end] += ramp * s.amp_x_deg
        x[end:] += s.amp_x_deg
        y[on:end] += ramp * s.amp_y_deg
        y[end:] += s.amp_y_deg
    return x, y


@pytest.fixture
def monkey_cfg() -> DatasetConfig:
    return DatasetConfig(
        name="test_monkey",
        sampling_rate_hz=2000.0,
        velocity_threshold_lambda=6.0,
        min_duration_ms=6.0,
        max_amplitude_deg=1.0,
        refractory_merge_ms=30.0,
        amplitude_outlier_sd=2.5,
        sigma_method="engbert2015",
        lowpass_cutoff_hz=None,
    )


@pytest.fixture
def human_cfg() -> DatasetConfig:
    return DatasetConfig(
        name="test_human",
        sampling_rate_hz=500.0,
        velocity_threshold_lambda=6.0,
        min_duration_ms=12.0,
        max_amplitude_deg=1.0,
        refractory_merge_ms=30.0,
        amplitude_outlier_sd=2.5,
        sigma_method="engbert2015",
        lowpass_cutoff_hz=None,
    )
