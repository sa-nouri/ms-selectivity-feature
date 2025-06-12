"""Tests for the blink detector module."""

import numpy as np
import pytest
from src.blink_detector import detect_blinks, validate_blinks

def test_detect_blinks():
    """Test blink detection functionality."""
    # Create sample eye tracking data with blinks
    timestamps = np.arange(0, 1000, 1)  # 1 second of data at 1000 Hz
    x_positions = np.zeros(1000)
    y_positions = np.zeros(1000)
    
    # Add some blinks (vertical eye movements)
    blink_indices = [200, 400, 600, 800]
    for idx in blink_indices:
        y_positions[idx:idx+50] = 10  # Simulate upward movement
        y_positions[idx+50:idx+100] = -10  # Simulate downward movement
    
    # Add some noise
    x_positions += np.random.normal(0, 0.1, 1000)
    y_positions += np.random.normal(0, 0.1, 1000)
    
    # Test blink detection
    blinks = detect_blinks(x_positions, y_positions, timestamps)
    
    # Verify results
    assert len(blinks) == 4  # Should detect 4 blinks
    for blink in blinks:
        assert blink['start_time'] < blink['end_time']
        assert blink['duration'] > 0
        assert blink['amplitude'] > 0

def test_validate_blinks():
    """Test blink validation functionality."""
    # Create sample blinks
    blinks = [
        {'start_time': 100, 'end_time': 150, 'duration': 50, 'amplitude': 5.0},
        {'start_time': 300, 'end_time': 350, 'duration': 50, 'amplitude': 4.0},
        {'start_time': 500, 'end_time': 550, 'duration': 50, 'amplitude': 6.0},
        {'start_time': 700, 'end_time': 750, 'duration': 50, 'amplitude': 3.0}
    ]
    
    # Test validation with different parameters
    # Test minimum duration
    validated = validate_blinks(blinks, min_duration=40)
    assert len(validated) == 4  # All blinks should pass
    
    validated = validate_blinks(blinks, min_duration=60)
    assert len(validated) == 0  # No blinks should pass
    
    # Test minimum amplitude
    validated = validate_blinks(blinks, min_amplitude=4.0)
    assert len(validated) == 3  # Three blinks should pass
    
    # Test maximum amplitude
    validated = validate_blinks(blinks, max_amplitude=5.0)
    assert len(validated) == 2  # Two blinks should pass

def test_detect_blinks_edge_cases():
    """Test blink detection with edge cases."""
    # Test with empty data
    timestamps = np.array([])
    x_positions = np.array([])
    y_positions = np.array([])
    
    blinks = detect_blinks(x_positions, y_positions, timestamps)
    assert len(blinks) == 0
    
    # Test with constant data (no blinks)
    timestamps = np.arange(0, 1000, 1)
    x_positions = np.ones(1000)
    y_positions = np.ones(1000)
    
    blinks = detect_blinks(x_positions, y_positions, timestamps)
    assert len(blinks) == 0
    
    # Test with NaN values
    x_positions = np.full(1000, np.nan)
    y_positions = np.full(1000, np.nan)
    
    with pytest.raises(ValueError):
        detect_blinks(x_positions, y_positions, timestamps)

def test_validate_blinks_edge_cases():
    """Test blink validation with edge cases."""
    # Test with empty blinks list
    validated = validate_blinks([])
    assert len(validated) == 0
    
    # Test with invalid blink data
    invalid_blinks = [
        {'start_time': 100, 'end_time': 50},  # Invalid duration
        {'start_time': 300, 'end_time': 350, 'duration': -10},  # Negative duration
        {'start_time': 500, 'end_time': 550, 'amplitude': -5.0}  # Negative amplitude
    ]
    
    with pytest.raises(ValueError):
        validate_blinks(invalid_blinks) 
