import numpy as np

from src.postprocess import (
    noise_threshold_extract,
    validate_saccades_min_duration,
)


def test_noise_threshold_extract() -> None:
    # Test data with two distinct distributions
    data = np.concatenate(
        [
            np.random.normal(0, 1, 1000),  # Noise distribution
            np.random.normal(10, 2, 1000),  # Signal distribution
        ]
    )

    # Test threshold extraction
    threshold = noise_threshold_extract(data)

    # Check results
    assert isinstance(threshold, float)
    assert threshold > 0
    assert threshold < 10  # Should be closer to noise distribution


def test_validate_saccades_min_duration() -> None:
    # Test data
    saccades = [(0, 2), (3, 4), (5, 8)]  # (start_index, end_index)
    timestamps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    min_duration = 2

    # Test validation
    valid_saccades = validate_saccades_min_duration(
        saccades, timestamps, min_duration, verbose=False
    )

    # Check results
    assert len(valid_saccades) < len(saccades)  # Some saccades should be filtered out
    for start, end in valid_saccades:
        duration = timestamps[end] - timestamps[start]
        assert (
            duration >= min_duration
        )  # All remaining saccades should meet duration requirement
