from typing import TypedDict, Any, Optional

import numpy as np

from .utils import compute_velocity


class SaccadeParams(TypedDict):
    min_duration: float
    max_duration: float
    min_amplitude: float
    max_amplitude: float
    velocity_threshold: float


class SaccadeDetector:
    """Detect saccades in eye movement data based on velocity thresholds."""

    def __init__(self, params: SaccadeParams) -> None:
        """Initialize the saccade detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid saccade
                - max_duration: Maximum duration (in seconds) allowed for a valid saccade
                - min_amplitude: Minimum amplitude (in degrees) required for a valid saccade
                - max_amplitude: Maximum amplitude (in degrees) allowed for a valid saccade
                - velocity_threshold: Velocity threshold for saccade detection
        """
        self.params = params

    def detect(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
        """Detect saccades in eye tracking data."""
        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = self.params["threshold_multiplier"] * np.sqrt(sigma_vx**2 + sigma_vy**2)

        saccades = []
        current_saccade = None

        for i in range(1, len(velocities) - 1):
            velocity_x, velocity_y = compute_velocity(x, y, timestamps, i - 1, i)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            if velocity_magnitude >= velocity_threshold:
                if current_saccade is None:
                    current_saccade = {
                        "start_idx": i,
                        "end_idx": i,
                        "duration": 0,
                        "amplitude": 0
                    }
                current_saccade["end_idx"] = i
            elif current_saccade is not None:
                current_saccade["duration"] = current_saccade["end_idx"] - current_saccade["start_idx"]
                current_saccade["amplitude"] = np.sqrt(
                    (x[current_saccade["end_idx"]] - x[current_saccade["start_idx"]])**2 +
                    (y[current_saccade["end_idx"]] - y[current_saccade["start_idx"]])**2
                )

                if (current_saccade["duration"] >= self.params["min_duration"] and
                    current_saccade["duration"] <= self.params["max_duration"] and
                    current_saccade["amplitude"] >= self.params["min_amplitude"] and
                    current_saccade["amplitude"] <= self.params["max_amplitude"]):
                    saccades.append(current_saccade)

                current_saccade = None

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
    saccades: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    max_duration: Optional[float] = None,
    max_amplitude: Optional[float] = None
) -> list[dict[str, Any]]:
    """Validate detected saccades."""
    if not saccades:
        return []

    valid_saccades = []
    for saccade in saccades:
        if max_duration is not None and saccade["duration"] > max_duration:
            continue
        if max_amplitude is not None and saccade["amplitude"] > max_amplitude:
            continue
        valid_saccades.append(saccade)

    return valid_saccades
