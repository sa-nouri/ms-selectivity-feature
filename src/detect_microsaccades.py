from typing import TypedDict, Any, Optional

import numpy as np

from .utils import compute_velocity


class MicrosaccadeParams(TypedDict):
    min_duration: float
    max_duration: float
    min_amplitude: float
    max_amplitude: float
    velocity_threshold: float


class MicrosaccadeDetector:
    """Detect microsaccades in eye movement data based on velocity thresholds."""

    def __init__(self, params: MicrosaccadeParams) -> None:
        """Initialize the microsaccade detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid
                    microsaccade
                - max_duration: Maximum duration (in seconds) allowed for a valid
                    microsaccade
                - min_amplitude: Minimum amplitude (in degrees) required for a valid
                    microsaccade
                - max_amplitude: Maximum amplitude (in degrees) allowed for a valid
                    microsaccade
                - velocity_threshold: Velocity threshold for microsaccade detection
        """
        self.params = params

    def detect(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
        """Detect microsaccades in eye tracking data."""
        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = self.params["threshold_multiplier"] * np.sqrt(sigma_vx**2 + sigma_vy**2)

        microsaccades = []
        current_microsaccade = None

        for i in range(1, len(velocities) - 1):
            velocity_x, velocity_y = compute_velocity(x, y, timestamps, i - 1, i)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            if velocity_magnitude >= velocity_threshold:
                if current_microsaccade is None:
                    current_microsaccade = {
                        "start_idx": i,
                        "end_idx": i,
                        "duration": 0,
                        "amplitude": 0
                    }
                current_microsaccade["end_idx"] = i
            elif current_microsaccade is not None:
                current_microsaccade["duration"] = current_microsaccade["end_idx"] - current_microsaccade["start_idx"]
                current_microsaccade["amplitude"] = np.sqrt(
                    (x[current_microsaccade["end_idx"]] - x[current_microsaccade["start_idx"]])**2 +
                    (y[current_microsaccade["end_idx"]] - y[current_microsaccade["start_idx"]])**2
                )

                if (current_microsaccade["duration"] >= self.params["min_duration"] and
                    current_microsaccade["duration"] <= self.params["max_duration"] and
                    current_microsaccade["amplitude"] >= self.params["min_amplitude"] and
                    current_microsaccade["amplitude"] <= self.params["max_amplitude"]):
                    microsaccades.append(current_microsaccade)

                current_microsaccade = None

        return microsaccades


def detect_microsaccades(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    min_duration: float = 0.006,
    max_duration: float = 0.025,
    min_amplitude: float = 0.1,
    max_amplitude: float = 1.0,
    velocity_threshold: float = 6.0,
) -> list[dict]:
    """Detect microsaccades in eye movement data.

    Args:
        x_positions: Array of X positions of eye movements
        y_positions: Array of Y positions of eye movements
        timestamps: Array of timestamps corresponding to the eye movement data
        min_duration: Minimum duration (in seconds) required for a valid microsaccade
        max_duration: Maximum duration (in seconds) allowed for a valid microsaccade
        min_amplitude: Minimum amplitude (in degrees) required for a valid microsaccade
        max_amplitude: Maximum amplitude (in degrees) allowed for a valid microsaccade
        velocity_threshold: Velocity threshold for microsaccade detection

    Returns:
        List of dictionaries containing information about detected microsaccades, where
        each dictionary contains:
            - start_time: Starting time of the microsaccade
            - end_time: Ending time of the microsaccade
            - duration: Duration of the microsaccade in seconds
            - amplitude: Amplitude of the microsaccade in degrees
            - direction: Direction of the microsaccade in radians
    """
    params = {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "min_amplitude": min_amplitude,
        "max_amplitude": max_amplitude,
        "velocity_threshold": velocity_threshold,
    }
    detector = MicrosaccadeDetector(params)
    return detector.detect(x_positions, y_positions, timestamps)


def validate_microsaccades(
    microsaccades: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    max_duration: Optional[float] = None,
    max_amplitude: Optional[float] = None
) -> list[dict[str, Any]]:
    """Validate detected microsaccades."""
    if not microsaccades:
        return []

    valid_microsaccades = []
    for microsaccade in microsaccades:
        if max_duration is not None and microsaccade["duration"] > max_duration:
            continue
        if max_amplitude is not None and microsaccade["amplitude"] > max_amplitude:
            continue
        valid_microsaccades.append(microsaccade)

    return valid_microsaccades
