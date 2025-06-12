import numpy as np

from src.preprocess import (
    correct_baseline_drift,
    filter_data,
    interpolate_data,
    low_pass_filter_eye_positions,
    remove_blinks,
)


def test_filter_data():
    # Test data
    x_positions = np.array([0, 1, 2, 3, 4])
    y_positions = np.array([0, 1, 2, 3, 4])

    # Test filtering
    filtered_x, filtered_y = filter_data(x_positions, y_positions)

    # Check results
    assert len(filtered_x) == len(x_positions)
    assert len(filtered_y) == len(y_positions)
    assert not np.any(np.isnan(filtered_x))
    assert not np.any(np.isnan(filtered_y))


def test_remove_blinks():
    # Test data with a blink (gap in data)
    x_positions = np.array([0, 1, 2, 5, 6])
    y_positions = np.array([0, 1, 2, 5, 6])

    # Test blink removal
    cleaned_x, cleaned_y = remove_blinks(x_positions, y_positions, blink_threshold=2)

    # Check results
    assert len(cleaned_x) < len(x_positions)
    assert len(cleaned_y) < len(y_positions)
    assert not np.any(np.isnan(cleaned_x))
    assert not np.any(np.isnan(cleaned_y))


def test_correct_baseline_drift():
    # Test data with linear drift
    x_positions = np.array([0, 1, 2, 3, 4])
    y_positions = np.array([0, 2, 4, 6, 8])  # Linear drift with slope 2

    # Test drift correction
    x_corrected, y_corrected = correct_baseline_drift(x_positions, y_positions)

    # Check results
    assert len(x_corrected) == len(x_positions)
    assert len(y_corrected) == len(y_positions)
    assert np.allclose(x_corrected, x_positions)
    assert np.allclose(y_corrected, np.zeros_like(y_positions), atol=1e-10)


def test_interpolate_data():
    # Test data with non-uniform sampling
    x_positions = np.array([0, 1, 3, 6])
    y_positions = np.array([0, 1, 3, 6])
    timestamps = np.array([0, 1, 3, 6])

    # Test interpolation
    x_inter, y_inter, t_inter = interpolate_data(
        x_positions, y_positions, timestamps, sampling_freq=1
    )

    # Check results
    assert len(x_inter) == len(y_inter) == len(t_inter)
    assert np.all(np.diff(t_inter) == 1)  # Uniform sampling
    assert not np.any(np.isnan(x_inter))
    assert not np.any(np.isnan(y_inter))


def test_low_pass_filter_eye_positions():
    # Test data
    x_positions = np.array([0, 1, 2, 3, 4])
    y_positions = np.array([0, 1, 2, 3, 4])
    cutoff_frequency = 1
    sampling_rate = 10

    # Test filtering
    filtered_x, filtered_y = low_pass_filter_eye_positions(
        x_positions, y_positions, cutoff_frequency, sampling_rate
    )

    # Check results
    assert len(filtered_x) == len(x_positions)
    assert len(filtered_y) == len(y_positions)
    assert not np.any(np.isnan(filtered_x))
    assert not np.any(np.isnan(filtered_y))
