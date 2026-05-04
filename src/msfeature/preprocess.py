"""Preprocessing for raw eye-position traces.

The detector assumes clean position data; this module is responsible for
producing it. Each step is a separate function so callers can compose
their own pipeline.

Order of operations recommended for the published datasets:

    1. mark_blinks         — flag samples where the tracker lost the eye
    2. remove_glitches     — replace 1-2 sample spikes with interpolation
    3. correct_drift       — remove slow time-locked baseline drift
    4. lowpass             — anti-alias / smooth before velocity differentiation

Note that blink masking is *flagging*, not removal — the trial-level mask
is propagated downstream so feature extractors can drop trials with
in-window blinks (per Nouri et al. methods).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def mark_blinks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pupil_loss_value: float | None = None,
    nan_as_blink: bool = True,
) -> np.ndarray:
    """Return a boolean mask, True where the sample is during a blink.

    Heuristic: blinks manifest as either NaN (interpolation gap) or a
    tracker-specific sentinel value (e.g. 0 in some EyeLink configs).
    Velocity-based blink detection is intentionally NOT used here, because
    that conflates blinks with saccades.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.zeros(x.shape, dtype=bool)
    if nan_as_blink:
        mask |= np.isnan(x) | np.isnan(y)
    if pupil_loss_value is not None:
        mask |= (x == pupil_loss_value) | (y == pupil_loss_value)
    return mask


def remove_glitches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    z_threshold: float = 6.0,
    max_run_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove short isolated tracking glitches by linear interpolation.

    A glitch is a run of <= `max_run_samples` consecutive samples whose
    first-difference robust z-score (per axis) exceeds `z_threshold`.
    Robust scale uses median absolute deviation (MAD) so the glitches
    themselves do not inflate the rejection threshold the way an
    SD-based score would. NaN samples are propagated as glitches.
    """
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    def _glitch_mask(p: np.ndarray) -> np.ndarray:
        finite = np.isfinite(p)
        if finite.sum() < 2:
            return ~finite
        d = np.empty(p.shape, dtype=float)
        d[0] = 0.0
        d[1:] = np.diff(p)
        # Robust scale: MAD scaled to be a consistent estimator of SD
        # for normal data. 1.4826 = 1 / Phi^-1(0.75).
        d_finite = d[np.isfinite(d)]
        if d_finite.size == 0:
            return ~finite
        mad = np.median(np.abs(d_finite - np.median(d_finite)))
        if mad == 0.0:
            # Fall back to SD if MAD collapses (degenerate data).
            scale = float(np.std(d_finite))
            if scale == 0.0:
                return ~finite
        else:
            scale = 1.4826 * float(mad)
        with np.errstate(invalid="ignore"):
            big = np.abs(d) > z_threshold * scale
        big = np.where(np.isfinite(d), big, True)  # NaN diff -> glitch
        return big | (~finite)

    raw = _glitch_mask(x) | _glitch_mask(y)
    if not raw.any():
        return x, y

    # Only replace runs short enough to be glitches, not real events.
    glitch = np.zeros(raw.shape, dtype=bool)
    in_run = False
    run_start = 0
    for i in range(raw.size):
        if raw[i] and not in_run:
            in_run = True
            run_start = i
        elif not raw[i] and in_run:
            in_run = False
            run_len = i - run_start
            if run_len <= max_run_samples:
                glitch[run_start:i] = True
    if in_run:
        run_len = raw.size - run_start
        if run_len <= max_run_samples:
            glitch[run_start:] = True

    if not glitch.any():
        return x, y

    idx = np.arange(x.size)
    good = ~glitch & np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return x, y
    x[glitch] = np.interp(idx[glitch], idx[good], x[good])
    y[glitch] = np.interp(idx[glitch], idx[good], y[good])
    return x, y


def correct_drift(
    x: np.ndarray, y: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a linear-in-time drift from each axis independently.

    The previous implementation regressed y on x, which conflates real
    eye geometry with drift. Drift is a slow change over time, so the
    correct regressor is the time axis.
    """
    n = x.size
    if n < 3:
        return x.copy(), y.copy()
    t = np.arange(n) * dt
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return x.copy(), y.copy()
    sx, ix = np.polyfit(t[finite], x[finite], 1)
    sy, iy = np.polyfit(t[finite], y[finite], 1)
    return x - (sx * t + ix), y - (sy * t + iy)


def lowpass(
    x: np.ndarray,
    y: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a zero-phase Butterworth low-pass filter to both axes.

    Skips filtering for traces too short for filtfilt's edge-padding
    requirement (`n > 3 * max(len(a), len(b))`) or for traces that
    contain non-finite samples (filtfilt would propagate NaN globally).
    """
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return x.copy(), y.copy()
    nyquist = sampling_rate_hz / 2.0
    normalized = cutoff_hz / nyquist
    if not 0 < normalized < 1:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} not in (0, {nyquist}) for fs={sampling_rate_hz}"
        )
    b, a = butter(order, normalized, btype="low")
    n = x.size
    if n <= 3 * max(len(a), len(b)):
        return x.copy(), y.copy()
    return filtfilt(b, a, x), filtfilt(b, a, y)


def preprocess_trial(
    x: np.ndarray,
    y: np.ndarray,
    sampling_rate_hz: float,
    *,
    lowpass_cutoff_hz: float | None = None,
    correct_drift_flag: bool = True,
    remove_glitches_flag: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the full preprocessing chain to one trial.

    Returns:
        Tuple of (x_clean, y_clean, blink_mask). The blink_mask is from
        the *original* signal (before interpolation) so the caller can
        decide whether to drop blink-containing trials.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    blink_mask = mark_blinks(x, y)
    if remove_glitches_flag:
        x, y = remove_glitches(x, y)
    if correct_drift_flag:
        x, y = correct_drift(x, y, dt=1.0 / sampling_rate_hz)
    if lowpass_cutoff_hz is not None:
        x, y = lowpass(x, y, sampling_rate_hz, lowpass_cutoff_hz)
    return x, y, blink_mask
