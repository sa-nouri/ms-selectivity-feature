"""Engbert & Kliegl 2003 microsaccade detector.

The pipeline for a single eye trace:

    velocity ──┐
               ├─► sigma  ──► elliptic threshold ──► run-length grouping
    velocity ──┘                                       │
                                                       ▼
                                        Microsaccade events (raw)
                                                       │
                                                       ▼
                                  refractory merge → amplitude cap →
                                  outlier rejection → final events

This module exposes a single high-level function `detect_microsaccades`
that runs the whole pipeline. Lower-level building blocks (`compute_sigma`,
`elliptic_threshold_mask`, `extract_runs`) are exported for tests and for
callers who want to compose their own pipeline.
"""

from __future__ import annotations

import numpy as np

from .config import DatasetConfig
from .events import (
    Microsaccade,
    cap_amplitude,
    cap_duration,
    merge_refractory,
    recompute_amplitude_direction,
    reject_amplitude_outliers,
)
from .velocity import ek_velocity_2d


def compute_sigma(
    velocity: np.ndarray,
    method: str = "engbert2015",
    fallback_to_std: bool = False,
) -> float:
    """Per-axis robust scale estimate for the EK threshold.

    Two canonical forms exist in the literature, both called "EK sigma":

    'engbert2003' (verbatim Engbert & Kliegl 2003 Eq. 2):
        sigma = sqrt( median(v^2) - median(v)^2 )

    'engbert2015' (Engbert R Microsaccade Toolbox 0.9 + pymovements default):
        sigma = sqrt( median( (v - median(v))^2 ) )

    These differ in general; they coincide only when median(v) is exactly
    zero. The 2015 form is what is in active use in modern toolboxes and
    is the recommended default here.

    Args:
        velocity: 1-D velocity array; non-finite entries are dropped.
        method: which sigma form to use. Defaults to 'engbert2015'.
        fallback_to_std: when True, return mean-based SD if the median
            form underflows. The R toolbox raises in that case; we
            default to that strict behaviour (returns 0.0 instead of
            falling back, so callers get an empty mask rather than an
            unintended threshold change).

    Returns:
        sigma in the same units as `velocity` (typically deg/s).
    """
    v = np.asarray(velocity, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    if method == "engbert2003":
        diff = float(np.median(v * v) - np.median(v) ** 2)
    elif method == "engbert2015":
        med = float(np.median(v))
        diff = float(np.median((v - med) ** 2))
    else:
        raise ValueError(
            f"unknown sigma method {method!r}; "
            f"expected 'engbert2003' or 'engbert2015'"
        )
    if diff < 1e-10:
        return float(np.std(v, ddof=0)) if fallback_to_std else 0.0
    return float(np.sqrt(diff))


def elliptic_threshold_mask(
    vx: np.ndarray, vy: np.ndarray, sigma_x: float, sigma_y: float, lam: float
) -> np.ndarray:
    """Return boolean array, True where the EK elliptic test fires.

        (vx / (lam * sigma_x))^2 + (vy / (lam * sigma_y))^2 > 1

    NaN velocities (the 5-point edge truncation, or NaN positions) are
    treated as below threshold.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape")
    if sigma_x <= 0 or sigma_y <= 0:
        return np.zeros(vx.shape, dtype=bool)
    radius_x = lam * sigma_x
    radius_y = lam * sigma_y
    with np.errstate(invalid="ignore"):
        test = (vx / radius_x) ** 2 + (vy / radius_y) ** 2
    return np.where(np.isfinite(test), test > 1.0, False)


def extract_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Find runs of `True` of length >= `min_length` in a boolean array.

    Returns a list of (start_idx, end_idx) inclusive index pairs.
    """
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    # Locate transitions by padding with False on both sides and diffing.
    padded = np.concatenate([[False], m, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    runs = [
        (int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_length
    ]
    return runs


def _build_event(
    start: int,
    end: int,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
) -> Microsaccade:
    dx = float(x[end] - x[start])
    dy = float(y[end] - y[start])
    speed = np.hypot(vx[start : end + 1], vy[start : end + 1])
    if speed.size == 0 or not np.any(np.isfinite(speed)):
        peak = float("nan")
    else:
        with np.errstate(invalid="ignore"):
            peak = float(np.nanmax(speed))
    return Microsaccade(
        start_idx=int(start),
        end_idx=int(end),
        start_time=float(start * dt),
        end_time=float(end * dt),
        duration=float((end - start) * dt),
        amplitude=float(np.hypot(dx, dy)),
        peak_velocity=peak,
        direction=float(np.arctan2(dy, dx)),
    )


def detect_microsaccades(
    x: np.ndarray,
    y: np.ndarray,
    config: DatasetConfig,
    apply_post_rules: bool = True,
) -> list[Microsaccade]:
    """Detect microsaccades on a single trial (one eye).

    Args:
        x, y: 1-D position arrays in degrees, same length, sample-indexed
            (no timestamps required — sampling rate comes from `config`).
        config: dataset configuration providing dt, lambda, min duration,
            and post-rule parameters.
        apply_post_rules: when True, apply Nouri et al.'s post-detection
            rules (refractory merge, amplitude cap, outlier rejection).
            Set False to inspect the raw EK output.

    Returns:
        List of `Microsaccade` events sorted by start_idx.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")

    vx, vy = ek_velocity_2d(x, y, config.dt)
    sigma_x = compute_sigma(
        vx, method=config.sigma_method, fallback_to_std=config.sigma_fallback_to_std
    )
    sigma_y = compute_sigma(
        vy, method=config.sigma_method, fallback_to_std=config.sigma_fallback_to_std
    )
    mask = elliptic_threshold_mask(
        vx, vy, sigma_x, sigma_y, config.velocity_threshold_lambda
    )
    runs = extract_runs(mask, config.min_duration_samples)
    events = [_build_event(s, e, x, y, vx, vy, config.dt) for s, e in runs]
    events = cap_duration(events, config.max_duration_ms * 1e-3)

    if not apply_post_rules:
        return events

    events = merge_refractory(events, config.refractory_merge_samples, config.dt)
    events = recompute_amplitude_direction(events, x, y)
    events = cap_duration(events, config.max_duration_ms * 1e-3)
    events = cap_amplitude(events, config.max_amplitude_deg)
    events = reject_amplitude_outliers(events, config.amplitude_outlier_sd)
    return events
