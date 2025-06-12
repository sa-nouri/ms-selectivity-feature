from typing import TypedDict

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

    def __init__(self, params: SaccadeParams):
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

    def detect_saccades(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[dict]:
        """Detect saccades in eye movement data.

        Args:
            x_positions: Array of X positions of eye movements
            y_positions: Array of Y positions of eye movements
            timestamps: Array of timestamps corresponding to the eye movement data

        Returns:
            List of dictionaries containing information about detected saccades, where
            each dictionary contains:
                - start_time: Starting time of the saccade
                - end_time: Ending time of the saccade
                - duration: Duration of the saccade in seconds
                - amplitude: Amplitude of the saccade in degrees
                - direction: Direction of the saccade in radians
        """
        if np.isnan(x_positions).any() or np.isnan(y_positions).any():
            raise ValueError("Input arrays must not contain NaN values")

        velocities, _, _ = compute_velocity(x_positions, y_positions, timestamps)
        velocity_magnitudes = np.sqrt(velocities[0] ** 2 + velocities[1] ** 2)

        saccades = []
        current_saccade = None

        for i in range(1, len(velocity_magnitudes) - 1):
            if velocity_magnitudes[i] >= self.params["velocity_threshold"]:
                if current_saccade is None:
                    current_saccade = {
                        "start_time": timestamps[i],
                        "end_time": timestamps[i],
                        "duration": 0,
                        "amplitude": 0,
                        "direction": 0,
                    }
                current_saccade["end_time"] = timestamps[i]
            elif current_saccade is not None:
                current_saccade["duration"] = (
                    current_saccade["end_time"] - current_saccade["start_time"]
                )
                dx = x_positions[i] - x_positions[i - current_saccade["duration"]]
                dy = y_positions[i] - y_positions[i - current_saccade["duration"]]
                current_saccade["amplitude"] = np.sqrt(dx**2 + dy**2)
                current_saccade["direction"] = np.arctan2(dy, dx)

                if (
                    current_saccade["duration"] >= self.params["min_duration"]
                    and current_saccade["duration"] <= self.params["max_duration"]
                    and current_saccade["amplitude"] >= self.params["min_amplitude"]
                    and current_saccade["amplitude"] <= self.params["max_amplitude"]
                ):
                    saccades.append(current_saccade)

                current_saccade = None

        # Handle the last saccade if it exists
        if current_saccade is not None:
            current_saccade["duration"] = (
                current_saccade["end_time"] - current_saccade["start_time"]
            )
            dx = x_positions[-1] - x_positions[-1 - current_saccade["duration"]]
            dy = y_positions[-1] - y_positions[-1 - current_saccade["duration"]]
            current_saccade["amplitude"] = np.sqrt(dx**2 + dy**2)
            current_saccade["direction"] = np.arctan2(dy, dx)

            if (
                current_saccade["duration"] >= self.params["min_duration"]
                and current_saccade["duration"] <= self.params["max_duration"]
                and current_saccade["amplitude"] >= self.params["min_amplitude"]
                and current_saccade["amplitude"] <= self.params["max_amplitude"]
            ):
                saccades.append(current_saccade)

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
    return detector.detect_saccades(x_positions, y_positions, timestamps)


def validate_saccades(
    saccades: list[dict],
    min_duration: float = 0,
    max_duration: float = None,
    min_amplitude: float = 0,
    max_amplitude: float = None,
) -> list[dict]:
    """Validate saccades based on duration and amplitude.

    Args:
        saccades: List of dictionaries containing information about detected saccades
        min_duration: Minimum duration (in seconds) required for a valid saccade
        max_duration: Maximum duration (in seconds) allowed for a valid saccade
        min_amplitude: Minimum amplitude (in degrees) required for a valid saccade
        max_amplitude: Maximum amplitude (in degrees) allowed for a valid saccade

    Returns:
        List of dictionaries containing information about validated saccades
    """
    validated = []
    for s in saccades:
        if s.get("duration", 0) < 0 or s.get("amplitude", 0) < 0:
            raise ValueError("Invalid saccade data: negative values")
        if (
            s.get("duration", 0) >= min_duration
            and s.get("amplitude", 0) >= min_amplitude
        ):
            if (max_duration is None or s.get("duration", 0) < max_duration) and (
                max_amplitude is None or s.get("amplitude", 0) < max_amplitude
            ):
                validated.append(s)
    return validated
