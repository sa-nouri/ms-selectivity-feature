"""EK 5-point velocity tests."""

from __future__ import annotations

import numpy as np
import pytest

from msfeature.velocity import ek_velocity, ek_velocity_2d


def test_constant_position_zero_velocity():
    v = ek_velocity(np.full(20, 3.14), dt=1e-3)
    assert np.all(v[2:-2] == 0.0)
    assert np.all(np.isnan(v[:2]))
    assert np.all(np.isnan(v[-2:]))


def test_linear_ramp_constant_velocity():
    """A linear position ramp should yield a constant velocity equal to slope/dt."""
    dt = 1e-3
    n = 20
    slope_per_sample = 0.5
    p = np.arange(n) * slope_per_sample
    v = ek_velocity(p, dt=dt)
    expected = slope_per_sample / dt
    np.testing.assert_allclose(v[2:-2], expected)


def test_kernel_matches_paper_formula():
    """Verify against the literal EK 2003 Eq. 1 on a non-trivial signal."""
    dt = 0.5e-3
    p = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8])
    v = ek_velocity(p, dt=dt)
    for n in range(2, len(p) - 2):
        manual = (p[n + 2] + p[n + 1] - p[n - 1] - p[n - 2]) / (6 * dt)
        assert np.isclose(v[n], manual)


def test_short_arrays_return_all_nan():
    for n in range(0, 5):
        v = ek_velocity(np.zeros(n), dt=1e-3)
        assert v.shape == (n,)
        assert np.all(np.isnan(v))


def test_dt_must_be_positive():
    with pytest.raises(ValueError):
        ek_velocity(np.zeros(10), dt=0)
    with pytest.raises(ValueError):
        ek_velocity(np.zeros(10), dt=-1e-3)


def test_2d_wrapper_matches_per_axis():
    dt = 1e-3
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 30)
    y = rng.normal(0, 1, 30)
    vx, vy = ek_velocity_2d(x, y, dt=dt)
    np.testing.assert_array_equal(vx, ek_velocity(x, dt=dt))
    np.testing.assert_array_equal(vy, ek_velocity(y, dt=dt))


def test_2d_wrapper_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ek_velocity_2d(np.zeros(10), np.zeros(11), dt=1e-3)


def test_nan_in_position_propagates_to_velocity():
    """A NaN at index k contaminates v[n] for n in {k-2, k-1, k+1, k+2}.
    v[k] itself is *not* contaminated because the EK kernel does not
    use p[k] when computing v[k] — a non-obvious but useful property
    that limits NaN spread."""
    p = np.arange(10.0)
    p[5] = np.nan
    v = ek_velocity(p, dt=1e-3)
    assert np.isnan(v[3]) and np.isnan(v[4])
    assert np.isnan(v[6]) and np.isnan(v[7])
    assert np.isfinite(v[5])


def test_velocity_rejects_2d_input():
    with pytest.raises(ValueError):
        ek_velocity(np.zeros((4, 4)), dt=1e-3)
