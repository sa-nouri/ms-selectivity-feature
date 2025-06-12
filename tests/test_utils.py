import numpy as np

from src.utils import (
    compute_velocity,
    compute_amplitude,
    compute_partial_velocity,
    compute_velocity_consecutive,
)


def test_compute_velocity():
    # Test data
    x_positions = np.array([0, 1, 2, 3, 4])
    y_positions = np.array([0, 1, 2, 3, 4])
    timestamps = np.array([0, 1, 2, 3, 4])
    
    # Compute velocities
    velocities, sigma_vx, sigma_vy = compute_velocity(x_positions, y_positions, timestamps)
    
    # Check results
    assert len(velocities) == len(x_positions) - 1
    assert isinstance(sigma_vx, float)
    assert isinstance(sigma_vy, float)
    assert sigma_vx >= 0
    assert sigma_vy >= 0


def test_compute_amplitude():
    # Test data
    x_positions = np.array([0, 3, 4])
    y_positions = np.array([0, 4, 0])
    
    # Test amplitude between points 0 and 1 (should be 5)
    amplitude = compute_amplitude(x_positions, y_positions, 0, 1)
    assert np.isclose(amplitude, 5.0)
    
    # Test amplitude between points 1 and 2 (should be 4)
    amplitude = compute_amplitude(x_positions, y_positions, 1, 2)
    assert np.isclose(amplitude, 4.0)


def test_compute_partial_velocity():
    # Test data
    x_positions = np.array([0, 1, 2])
    y_positions = np.array([0, 1, 2])
    timestamps = np.array([0, 1, 2])
    
    # Test velocity between points 0 and 1
    vx, vy = compute_partial_velocity(x_positions, y_positions, timestamps, 0, 1)
    assert np.isclose(vx, 1.0)
    assert np.isclose(vy, 1.0)


def test_compute_velocity_consecutive():
    # Test data
    x_position = np.array([0, 1, 2, 3])
    y_position = np.array([0, 1, 2, 3])
    timestamp = np.array([0, 1, 2, 3])
    
    # Compute velocities
    x_velocity, y_velocity, pythagorean_velocity = compute_velocity_consecutive(
        x_position, y_position, timestamp
    )
    
    # Check results
    assert len(x_velocity) == len(x_position)
    assert len(y_velocity) == len(y_position)
    assert len(pythagorean_velocity) == len(x_position)
    assert np.all(x_velocity >= 0)
    assert np.all(y_velocity >= 0)
    assert np.all(pythagorean_velocity >= 0) 
