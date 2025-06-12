import copy
from typing import TypedDict

import numpy as np

from .utils import compute_velocity


class BlinkParams(TypedDict):
    min_duration: float
    max_duration: float
    threshold_multiplier: float


class BlinkDetectorByEyePositions:
    """Detect blinks in eye movement data based on changes in gaze direction."""

    def __init__(self, params: BlinkParams):
        """Initialize the blink detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid blink
                - max_duration: Maximum duration (in seconds) allowed for a valid blink
                - threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set
                    velocity threshold
        """
        self.params = params

    def detect_blinks(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[dict]:
        """Detect blinks in eye movement data.

        Args:
            x_positions: Array of X positions of eye movements
            y_positions: Array of Y positions of eye movements
            timestamps: Array of timestamps corresponding to the eye movement data

        Returns:
            List of dictionaries containing information about detected blinks, where
            each dictionary contains:
                - start_time: Starting time of the blink
                - end_time: Ending time of the blink
                - duration: Duration of the blink in seconds
        """
        velocities, sigma_vx, sigma_vy = compute_velocity(
            x_positions, y_positions, timestamps
        )
        velocity_threshold = self.params["threshold_multiplier"] * np.sqrt(
            sigma_vx**2 + sigma_vy**2
        )

        blinks = []
        current_blink = None
        blink_start = None

        for i in range(1, len(velocities) - 1):
            velocity_x, velocity_y = compute_velocity(
                x_positions, y_positions, timestamps, i - 1, i
            )
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            if velocity_magnitude >= velocity_threshold:
                if current_blink is None:
                    blink_start = i
                    current_blink = {
                        "start_time": timestamps[i],
                        "end_time": timestamps[i],
                        "duration": 0,
                    }
                current_blink["end_time"] = timestamps[i]
            elif current_blink is not None:
                current_blink["duration"] = (
                    current_blink["end_time"] - current_blink["start_time"]
                )

                if (
                    current_blink["duration"] >= self.params["min_duration"]
                    and current_blink["duration"] <= self.params["max_duration"]
                ):
                    blinks.append(current_blink)

                current_blink = None
                blink_start = None

        # Handle the last blink if it exists
        if current_blink is not None:
            current_blink["duration"] = (
                current_blink["end_time"] - current_blink["start_time"]
            )
            if (
                current_blink["duration"] >= self.params["min_duration"]
                and current_blink["duration"] <= self.params["max_duration"]
            ):
                blinks.append(current_blink)

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
    detector = BlinkDetectorByEyePositions(params)
    return detector.detect_blinks(x_positions, y_positions, timestamps)


def validate_blinks(
    blinks: list[dict],
    min_duration: float = 0,
    max_duration: float = None,
) -> list[dict]:
    """Validate blinks based on duration.

    Args:
        blinks: List of dictionaries containing information about detected blinks
        min_duration: Minimum duration (in seconds) required for a valid blink
        max_duration: Maximum duration (in seconds) allowed for a valid blink

    Returns:
        List of dictionaries containing information about validated blinks
    """
    validated = []
    for b in blinks:
        if b.get("duration", 0) < 0:
            raise ValueError("Invalid blink data: negative values")
        if b.get("duration", 0) >= min_duration:
            if max_duration is None or b.get("duration", 0) < max_duration:
                validated.append(b)
    return validated
