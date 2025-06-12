"""Tests for the saccade detection module."""

import numpy as np
import pytest
from src.detect_saccades import detect_saccades, validate_saccades


def test_detect_saccades():
    """Test saccade detection functionality."""
    # Create sample eye tracking data with saccades
    timestamps = np.arange(0, 1000, 1)  # 1 second of data at 1000 Hz
    x_positions = np.zeros(1000)
    y_positions = np.zeros(1000)

    # Add some saccades (horizontal eye movements)
    saccade_indices = [200, 400, 600, 800]
    for idx in saccade_indices:
        x_positions[idx : idx + 50] = 5  # Simulate rightward movement
        x_positions[idx + 50 : idx + 100] = -5  # Simulate leftward movement

    # Add some noise
    x_positions += np.random.normal(0, 0.1, 1000)
    y_positions += np.random.normal(0, 0.1, 1000)

    # Test saccade detection
    saccades = detect_saccades(x_positions, y_positions, timestamps)

    # Verify results
    assert len(saccades) == 4  # Should detect 4 saccades
    for saccade in saccades:
        assert saccade["start_time"] < saccade["end_time"]
        assert saccade["duration"] > 0
        assert saccade["amplitude"] > 0
        assert "direction" in saccade


def test_validate_saccades():
    """Test saccade validation functionality."""
    # Create sample saccades
    saccades = [
        {
            "start_time": 100,
            "end_time": 150,
            "duration": 50,
            "amplitude": 5.0,
            "direction": 0,
        },
        {
            "start_time": 300,
            "end_time": 350,
            "duration": 50,
            "amplitude": 4.0,
            "direction": 45,
        },
        {
            "start_time": 500,
            "end_time": 550,
            "duration": 50,
            "amplitude": 6.0,
            "direction": 90,
        },
        {
            "start_time": 700,
            "end_time": 750,
            "duration": 50,
            "amplitude": 3.0,
            "direction": 135,
        },
    ]

    # Test validation with different parameters
    # Test minimum duration
    validated = validate_saccades(saccades, min_duration=40)
    assert len(validated) == 4  # All saccades should pass

    validated = validate_saccades(saccades, min_duration=60)
    assert len(validated) == 0  # No saccades should pass

    # Test minimum amplitude
    validated = validate_saccades(saccades, min_amplitude=4.0)
    assert len(validated) == 3  # Three saccades should pass

    # Test maximum amplitude
    validated = validate_saccades(saccades, max_amplitude=5.0)
    assert len(validated) == 2  # Two saccades should pass


def test_detect_saccades_edge_cases():
    """Test saccade detection with edge cases."""
    # Test with empty data
    timestamps = np.array([])
    x_positions = np.array([])
    y_positions = np.array([])

    saccades = detect_saccades(x_positions, y_positions, timestamps)
    assert len(saccades) == 0

    # Test with constant data (no saccades)
    timestamps = np.arange(0, 1000, 1)
    x_positions = np.ones(1000)
    y_positions = np.ones(1000)

    saccades = detect_saccades(x_positions, y_positions, timestamps)
    assert len(saccades) == 0

    # Test with NaN values
    x_positions = np.full(1000, np.nan)
    y_positions = np.full(1000, np.nan)

    with pytest.raises(ValueError):
        detect_saccades(x_positions, y_positions, timestamps)


def test_validate_saccades_edge_cases():
    """Test saccade validation with edge cases."""
    # Test with empty saccades list
    validated = validate_saccades([])
    assert len(validated) == 0

    # Test with invalid saccade data
    invalid_saccades = [
        {"start_time": 100, "end_time": 50},  # Invalid duration
        {"start_time": 300, "end_time": 350, "duration": -10},  # Negative duration
        {"start_time": 500, "end_time": 550, "amplitude": -5.0},  # Negative amplitude
    ]

    with pytest.raises(ValueError):
        validate_saccades(invalid_saccades)
