import numpy as np
from typing import Dict, List, TypedDict, Optional

from .utils import compute_amplitude, compute_partial_velocity, compute_velocity


class SaccadeParams(TypedDict):
    """Parameters for saccade detection.

    Attributes:
        min_duration: Minimum duration (in seconds) required for a valid saccade.
        threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set velocity threshold.
    """

    min_duration: float
    threshold_multiplier: float


class Saccade(TypedDict):
    """Information about a detected saccade.

    Attributes:
        start: Starting index of the saccade.
        end: Ending index of the saccade.
        amplitude: Amplitude of the saccade in degrees.
        velocity: Velocity of the saccade in degrees per second.
        duration: Duration of the saccade in seconds.
    """

    start: int
    end: int
    amplitude: float
    velocity: float
    duration: float


def detect_saccades(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    params: SaccadeParams,
) -> List[Saccade]:
    """Detect saccades in eye movement data based on changes in gaze direction.

    This function implements a velocity-based algorithm to detect saccades in eye movement data.
    It uses a combination of velocity thresholds and duration criteria to identify valid saccades.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        params: Dictionary containing detection parameters:
            - min_duration: Minimum duration (in seconds) required for a valid saccade.
            - threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set velocity threshold.

    Returns:
        List of dictionaries containing information about detected saccades, where each
        dictionary contains:
            - start_time: Starting time of the saccade
            - end_time: Ending time of the saccade
            - duration: Duration of the saccade in milliseconds
            - amplitude: Amplitude of the saccade in degrees
            - direction: Direction of the saccade in degrees
    """
    # Check for NaN values
    if np.isnan(x_positions).any() or np.isnan(y_positions).any():
        raise ValueError("Input arrays must not contain NaN values")

    if len(x_positions) == 0 or np.all(x_positions == x_positions[0]):
        return []

    velocities, sigma_vx, sigma_vy = compute_velocity(
        x_positions, y_positions, timestamps
    )
    velocity_threshold = params["threshold_multiplier"] * np.sqrt(
        sigma_vx**2 + sigma_vy**2
    )

    saccades = []
    current_saccade = None
    saccade_start = None

    for i in range(1, len(velocities) - 1):
        velocity_x, velocity_y = compute_partial_velocity(
            x_positions, y_positions, timestamps, i - 1, i
        )
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

        if velocity_magnitude >= velocity_threshold:
            if current_saccade is None:
                saccade_start = i
                current_saccade = {
                    "start_time": timestamps[i],
                    "end_time": timestamps[i],
                    "duration": 0,
                    "amplitude": 0,
                    "direction": np.degrees(np.arctan2(velocity_y, velocity_x)),
                }
            current_saccade["end_time"] = timestamps[i]
        elif current_saccade is not None:
            current_saccade["duration"] = (
                current_saccade["end_time"] - current_saccade["start_time"]
            )

            if current_saccade["duration"] >= params["min_duration"]:
                current_saccade["amplitude"] = compute_amplitude(
                    x_positions, y_positions, saccade_start, i - 1
                )
                saccades.append(current_saccade)

            current_saccade = None
            saccade_start = None

    # Handle the last saccade if it exists
    if current_saccade is not None:
        current_saccade["duration"] = (
            current_saccade["end_time"] - current_saccade["start_time"]
        )
        if current_saccade["duration"] >= params["min_duration"]:
            current_saccade["amplitude"] = compute_amplitude(
                x_positions, y_positions, saccade_start, len(x_positions) - 1
            )
            saccades.append(current_saccade)

    return saccades


def validate_saccades(saccades, min_duration=0, min_amplitude=0, max_amplitude=None):
    """Validate saccades based on duration and amplitude for test compatibility."""
    validated = []
    for s in saccades:
        if s.get("duration", 0) < 0 or s.get("amplitude", 0) < 0:
            raise ValueError("Invalid saccade data: negative values")
        if (
            s.get("duration", 0) >= min_duration
            and s.get("amplitude", 0) >= min_amplitude
        ):
            if max_amplitude is None or s.get("amplitude", 0) < max_amplitude:
                validated.append(s)
    return validated


def detect_saccades(
    x_positions, y_positions, timestamps, min_duration=50, threshold_multiplier=6.0
):
    """Test-compatible wrapper for detect_saccades with default params."""
    params = {
        "min_duration": min_duration,
        "threshold_multiplier": threshold_multiplier,
    }
    return _detect_saccades_impl(x_positions, y_positions, timestamps, params)


# Save the original implementation for the wrapper
_detect_saccades_impl = detect_saccades
