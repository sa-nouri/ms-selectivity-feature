"""Engbert & Kliegl 5-sample velocity estimator.

EK 2003 eq. 1:
    v_n = (x[n+2] + x[n+1] - x[n-1] - x[n-2]) / (6 * dt)

Applied independently to x and y. The first two and last two samples have
no defined velocity under this kernel; we set them to NaN so callers can
exclude them from threshold and event detection without confusing zero-
velocity edges with real fixations.
"""

from __future__ import annotations

import numpy as np


def ek_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute EK 2003 5-sample velocity for a 1-D position array.

    Args:
        positions: 1-D array of positions (any unit; output unit is unit/s).
        dt: sampling interval in seconds.

    Returns:
        Array of same length as `positions`, with the first two and last
        two entries set to NaN.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    p = np.asarray(positions, dtype=float)
    if p.ndim != 1:
        raise ValueError("positions must be 1-D")
    n = p.size
    v = np.full(n, np.nan, dtype=float)
    if n >= 5:
        v[2:-2] = (p[4:] + p[3:-1] - p[1:-3] - p[:-4]) / (6.0 * dt)
    return v


def ek_velocity_2d(
    x: np.ndarray, y: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-axis convenience wrapper for `ek_velocity`."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return ek_velocity(x, dt), ek_velocity(y, dt)
