from typing import TypedDict

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

    def __init__(self, params: MicrosaccadeParams):
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

    def detect_microsaccades(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[dict]:
        """Detect microsaccades in eye movement data.

        Args:
            x_positions: Array of X positions of eye movements
            y_positions: Array of Y positions of eye movements
            timestamps: Array of timestamps corresponding to the eye movement data

        Returns:
            List of dictionaries containing information about detected microsaccades,
            where each dictionary contains:
                - start_time: Starting time of the microsaccade
                - end_time: Ending time of the microsaccade
                - duration: Duration of the microsaccade in seconds
                - amplitude: Amplitude of the microsaccade in degrees
                - direction: Direction of the microsaccade in radians
        """
        if np.isnan(x_positions).any() or np.isnan(y_positions).any():
            raise ValueError("Input arrays must not contain NaN values")

        velocities, _, _ = compute_velocity(x_positions, y_positions, timestamps)
        velocity_magnitudes = np.sqrt(velocities[0] ** 2 + velocities[1] ** 2)

        microsaccades = []
        current_microsaccade = None

        for i in range(1, len(velocity_magnitudes) - 1):
            if velocity_magnitudes[i] >= self.params["velocity_threshold"]:
                if current_microsaccade is None:
                    current_microsaccade = {
                        "start_time": timestamps[i],
                        "end_time": timestamps[i],
                        "duration": 0,
                        "amplitude": 0,
                        "direction": 0,
                    }
                current_microsaccade["end_time"] = timestamps[i]
            elif current_microsaccade is not None:
                current_microsaccade["duration"] = (
                    current_microsaccade["end_time"]
                    - current_microsaccade["start_time"]
                )
                dx = x_positions[i] - x_positions[i - current_microsaccade["duration"]]
                dy = y_positions[i] - y_positions[i - current_microsaccade["duration"]]
                current_microsaccade["amplitude"] = np.sqrt(dx**2 + dy**2)
                current_microsaccade["direction"] = np.arctan2(dy, dx)

                if (
                    current_microsaccade["duration"] >= self.params["min_duration"]
                    and current_microsaccade["duration"] <= self.params["max_duration"]
                    and current_microsaccade["amplitude"]
                    >= self.params["min_amplitude"]
                    and current_microsaccade["amplitude"]
                    <= self.params["max_amplitude"]
                ):
                    microsaccades.append(current_microsaccade)

                current_microsaccade = None

        # Handle the last microsaccade if it exists
        if current_microsaccade is not None:
            current_microsaccade["duration"] = (
                current_microsaccade["end_time"] - current_microsaccade["start_time"]
            )
            dx = x_positions[-1] - x_positions[-1 - current_microsaccade["duration"]]
            dy = y_positions[-1] - y_positions[-1 - current_microsaccade["duration"]]
            current_microsaccade["amplitude"] = np.sqrt(dx**2 + dy**2)
            current_microsaccade["direction"] = np.arctan2(dy, dx)

            if (
                current_microsaccade["duration"] >= self.params["min_duration"]
                and current_microsaccade["duration"] <= self.params["max_duration"]
                and current_microsaccade["amplitude"] >= self.params["min_amplitude"]
                and current_microsaccade["amplitude"] <= self.params["max_amplitude"]
            ):
                microsaccades.append(current_microsaccade)

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
    return detector.detect_microsaccades(x_positions, y_positions, timestamps)


def validate_microsaccades(
    microsaccades: list[dict],
    min_duration: float = 0,
    max_duration: float = None,
    min_amplitude: float = 0,
    max_amplitude: float = None,
) -> list[dict]:
    """Validate microsaccades based on duration and amplitude.

    Args:
        microsaccades: List of dictionaries containing information about detected
            microsaccades
        min_duration: Minimum duration (in seconds) required for a valid microsaccade
        max_duration: Maximum duration (in seconds) allowed for a valid microsaccade
        min_amplitude: Minimum amplitude (in degrees) required for a valid microsaccade
        max_amplitude: Maximum amplitude (in degrees) allowed for a valid microsaccade

    Returns:
        List of dictionaries containing information about validated microsaccades
    """
    validated = []
    for m in microsaccades:
        if m.get("duration", 0) < 0 or m.get("amplitude", 0) < 0:
            raise ValueError("Invalid microsaccade data: negative values")
        if (
            m.get("duration", 0) >= min_duration
            and m.get("amplitude", 0) >= min_amplitude
        ):
            if (max_duration is None or m.get("duration", 0) < max_duration) and (
                max_amplitude is None or m.get("amplitude", 0) < max_amplitude
            ):
                validated.append(m)
    return validated
