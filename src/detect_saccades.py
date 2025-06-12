from typing import Any, Optional, TypedDict

import numpy as np

from .utils import compute_velocity


class SaccadeParams(TypedDict):
    min_duration: float
    max_duration: float
    min_amplitude: float
    max_amplitude: float
    velocity_threshold: float


class SaccadeDetector:
    """Detector for saccades in eye tracking data."""

    def __init__(self, threshold_multiplier: float = 5.0) -> None:
        """Initialize the saccade detector.

        Args:
            threshold_multiplier: Multiplier for velocity threshold (default: 5.0)
        """
        self.threshold_multiplier = threshold_multiplier

    def detect(
        self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray
    ) -> list[dict[str, Any]]:
        """Detect saccades in eye tracking data."""
        if len(x) < 2 or len(y) < 2 or len(timestamps) < 2:
            return []

        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = self.threshold_multiplier * np.sqrt(
            sigma_vx**2 + sigma_vy**2
        )

        # Find points where velocity exceeds threshold
        saccade_indices = np.where(
            np.sqrt(velocities[0] ** 2 + velocities[1] ** 2) > velocity_threshold
        )[0]

        # Convert to list of dictionaries
        saccades = []
        for idx in saccade_indices:
            saccades.append(
                {
                    "time": timestamps[idx],
                    "magnitude": np.sqrt(
                        velocities[0][idx] ** 2 + velocities[1][idx] ** 2
                    ),
                }
            )

        return saccades


def detect_saccades(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    min_duration: float = 0.02,
    max_duration: float = 0.1,
    min_amplitude: float = 0.5,
    max_amplitude: float = 20.0,
    velocity_threshold: float = 30.0,
) -> list[dict]:
    """Detect saccades in eye movement data.

    Args:
        x_positions: Array of X positions of eye movements
        y_positions: Array of Y positions of eye movements
        timestamps: Array of timestamps corresponding to the eye movement data
        min_duration: Minimum duration (in seconds) required for a valid saccade
        max_duration: Maximum duration (in seconds) allowed for a valid saccade
        min_amplitude: Minimum amplitude (in degrees) required for a valid saccade
        max_amplitude: Maximum amplitude (in degrees) allowed for a valid saccade
        velocity_threshold: Velocity threshold for saccade detection

    Returns:
        List of dictionaries containing information about detected saccades, where each
        dictionary contains:
            - start_time: Starting time of the saccade
            - end_time: Ending time of the saccade
            - duration: Duration of the saccade in seconds
            - amplitude: Amplitude of the saccade in degrees
            - direction: Direction of the saccade in radians
    """
    params = {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "min_amplitude": min_amplitude,
        "max_amplitude": max_amplitude,
        "velocity_threshold": velocity_threshold,
    }
    detector = SaccadeDetector(params)
    return detector.detect(x_positions, y_positions, timestamps)


def validate_saccades(
    saccades: list[dict],
    min_duration: float = 0.0,
    min_amplitude: float = None,
    max_amplitude: float = None,
) -> list[dict]:
    """Validate detected saccades.

    Args:
        saccades: List of detected saccades (dicts)
        min_duration: Minimum duration threshold
        min_amplitude: Minimum amplitude threshold
        max_amplitude: Maximum amplitude threshold

    Returns:
        List of validated saccades
    """
    if not isinstance(saccades, list):
        raise ValueError("saccades must be a list of dicts")
    validated = []
    for s in saccades:
        if not isinstance(s, dict):
            raise ValueError("Each saccade must be a dict")
        duration = s.get("duration", s.get("end_time", 0) - s.get("start_time", 0))
        amplitude = s.get("amplitude", None)
        if duration is not None and duration < 0:
            raise ValueError("Negative duration in saccade")
        if amplitude is not None and amplitude < 0:
            raise ValueError("Negative amplitude in saccade")
        if s.get("start_time") is not None and s.get("end_time") is not None:
            if s["start_time"] > s["end_time"]:
                raise ValueError("start_time greater than end_time in saccade")
        if duration is not None and duration < min_duration:
            continue
        if (
            min_amplitude is not None
            and amplitude is not None
            and amplitude < min_amplitude
        ):
            continue
        if (
            max_amplitude is not None
            and amplitude is not None
            and amplitude > max_amplitude
        ):
            continue
        validated.append(s)
    return validated
