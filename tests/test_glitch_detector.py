"""Tests for the glitch detector module."""

import numpy as np
import pytest

from src.glitch_detector import detect_glitches, validate_glitches


def test_detect_glitches():
    """Test glitch detection functionality."""
    # Create sample eye tracking data with glitches
    timestamps = np.arange(0, 1000, 1)  # 1 second of data at 1000 Hz
    x_positions = np.zeros(1000)
    y_positions = np.zeros(1000)

    # Add some glitches (sudden jumps in position)
    glitch_indices = [200, 400, 600, 800]
    for idx in glitch_indices:
        x_positions[idx] = 100  # Sudden jump
        y_positions[idx] = 100  # Sudden jump

    # Add some noise
    x_positions += np.random.normal(0, 0.1, 1000)
    y_positions += np.random.normal(0, 0.1, 1000)

    # Test glitch detection
    glitches = detect_glitches(x_positions, y_positions, timestamps)

    # Verify results
    assert len(glitches) == 4  # Should detect 4 glitches
    for glitch in glitches:
        assert glitch["time"] in glitch_indices
        assert glitch["magnitude"] > 0


def test_validate_glitches() -> None:
    """Test glitch validation functionality."""
    # Create sample glitches
    glitches = [
        {"time": 100, "magnitude": 5.0},
        {"time": 300, "magnitude": 4.0},
        {"time": 500, "magnitude": 6.0},
        {"time": 700, "magnitude": 3.0},
    ]

    # Test validation with different parameters
    # Test minimum magnitude
    validated = validate_glitches(glitches, min_magnitude=4.0)
    assert len(validated) == 3  # Three glitches should pass

    # Test maximum magnitude
    validated = validate_glitches(glitches, max_magnitude=5.0)
    assert len(validated) == 2  # Two glitches should pass


def test_detect_glitches_edge_cases():
    """Test glitch detection with edge cases."""
    # Test with empty data
    timestamps = np.array([])
    x_positions = np.array([])
    y_positions = np.array([])

    glitches = detect_glitches(x_positions, y_positions, timestamps)
    assert len(glitches) == 0

    # Test with constant data (no glitches)
    timestamps = np.arange(0, 1000, 1)
    x_positions = np.ones(1000)
    y_positions = np.ones(1000)

    glitches = detect_glitches(x_positions, y_positions, timestamps)
    assert len(glitches) == 0

    # Test with NaN values
    x_positions = np.full(1000, np.nan)
    y_positions = np.full(1000, np.nan)

    with pytest.raises(ValueError):
        detect_glitches(x_positions, y_positions, timestamps)


def test_validate_glitches_edge_cases():
    """Test glitch validation with edge cases."""
    # Test with empty glitches list
    validated = validate_glitches([])
    assert len(validated) == 0

    # Test with invalid glitch data
    invalid_glitches = [
        {"time": 100, "magnitude": -5.0},  # Negative magnitude
        {"time": -1, "magnitude": 5.0},  # Negative time
    ]

    with pytest.raises(ValueError):
        validate_glitches(invalid_glitches)


def test_glitch_detector_with_params() -> None:
    # Test data
    glitches = [(0, 2), (3, 4), (5, 8)]  # (start_index, end_index)
    x_positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Test validation
    valid_glitches = validate_glitches(glitches, x_positions, y_positions)
    assert len(valid_glitches) == 3


def test_glitch_detector_with_empty_data() -> None:
    # Test with empty data
    timestamps = np.array([])
    x_positions = np.array([])
    y_positions = np.array([])

    glitches = detect_glitches(x_positions, y_positions, timestamps)
    assert len(glitches) == 0
 