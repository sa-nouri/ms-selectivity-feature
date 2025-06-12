import copy
from typing import Any, Optional, TypedDict

import numpy as np

from .utils import compute_velocity


class BlinkParams(TypedDict):
    min_duration: float
    max_duration: float
    threshold_multiplier: float


class BlinkDetector:
    """Detector for blinks in eye tracking data."""

    def __init__(self, threshold_multiplier: float = 5.0) -> None:
        """Initialize the blink detector.

        Args:
            threshold_multiplier: Multiplier for velocity threshold (default: 5.0)
        """
        self.threshold_multiplier = threshold_multiplier

    def detect(
        self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray
    ) -> list[dict[str, Any]]:
        """Detect blinks in eye tracking data."""
        if len(x) < 2 or len(y) < 2 or len(timestamps) < 2:
            return []

        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = self.threshold_multiplier * np.sqrt(
            sigma_vx**2 + sigma_vy**2
        )

        # Find points where velocity exceeds threshold
        blink_indices = np.where(
            np.sqrt(velocities[0] ** 2 + velocities[1] ** 2) > velocity_threshold
        )[0]

        # Convert to list of dictionaries
        blinks = []
        for idx in blink_indices:
            blinks.append(
                {
                    "time": timestamps[idx],
                    "magnitude": np.sqrt(
                        velocities[0][idx] ** 2 + velocities[1][idx] ** 2
                    ),
                }
            )

        return blinks


def detect_blinks(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    min_duration: float = 0.05,
    max_duration: float = 0.4,
    threshold_multiplier: float = 6.0,
) -> list[dict]:
    """Detect blinks in eye movement data.

    Args:
        x_positions: Array of X positions of eye movements
        y_positions: Array of Y positions of eye movements
        timestamps: Array of timestamps corresponding to the eye movement data
        min_duration: Minimum duration (in seconds) required for a valid blink
        max_duration: Maximum duration (in seconds) allowed for a valid blink
        threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set velocity
            threshold

    Returns:
        List of dictionaries containing information about detected blinks, where each
        dictionary contains:
            - start_time: Starting time of the blink
            - end_time: Ending time of the blink
            - duration: Duration of the blink in seconds
    """
    params = {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "threshold_multiplier": threshold_multiplier,
    }
    detector = BlinkDetector(params["threshold_multiplier"])
    return detector.detect(x_positions, y_positions, timestamps)


def validate_blinks(
    blinks: list[dict],
    min_duration: float = 0.0,
    min_amplitude: float = None,
    max_amplitude: float = None,
) -> list[dict]:
    """Validate detected blinks.

    Args:
        blinks: List of detected blinks (dicts)
        min_duration: Minimum duration threshold
        min_amplitude: Minimum amplitude threshold
        max_amplitude: Maximum amplitude threshold

    Returns:
        List of validated blinks
    """
    if not isinstance(blinks, list):
        raise ValueError("blinks must be a list of dicts")
    validated = []
    for b in blinks:
        if not isinstance(b, dict):
            raise ValueError("Each blink must be a dict")
        # Check for required fields
        duration = b.get("duration", b.get("end_time", 0) - b.get("start_time", 0))
        amplitude = b.get("amplitude", None)
        if duration is not None and duration < 0:
            raise ValueError("Negative duration in blink")
        if amplitude is not None and amplitude < 0:
            raise ValueError("Negative amplitude in blink")
        if b.get("start_time") is not None and b.get("end_time") is not None:
            if b["start_time"] > b["end_time"]:
                raise ValueError("start_time greater than end_time in blink")
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
        validated.append(b)
    return validated
