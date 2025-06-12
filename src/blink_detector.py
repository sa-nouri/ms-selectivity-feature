import copy
from typing import TypedDict, Any, Optional

import numpy as np

from .utils import compute_velocity


class BlinkParams(TypedDict):
    min_duration: float
    max_duration: float
    threshold_multiplier: float


class BlinkDetectorByEyePositions:
    """Detect blinks in eye movement data based on changes in gaze direction."""

    def __init__(self, params: BlinkParams) -> None:
        """Initialize the blink detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid blink
                - max_duration: Maximum duration (in seconds) allowed for a valid blink
                - threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set
                    velocity threshold
        """
        self.params = params

    def detect(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
        """Detect blinks in eye tracking data."""
        blinks = []
        for i in range(len(x) - 1):
            if np.isnan(x[i]) or np.isnan(y[i]):
                start_idx = i
                while i < len(x) and (np.isnan(x[i]) or np.isnan(y[i])):
                    i += 1
                end_idx = i
                if end_idx - start_idx >= self.params["min_duration"]:
                    blinks.append({
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration": end_idx - start_idx
                    })
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
    return detector.detect(x_positions, y_positions, timestamps)


def validate_blinks(
    blinks: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    max_duration: Optional[float] = None
) -> list[dict[str, Any]]:
    """Validate detected blinks."""
    if not blinks:
        return []

    valid_blinks = []
    for blink in blinks:
        if max_duration is not None and blink["duration"] > max_duration:
            continue
        valid_blinks.append(blink)

    return valid_blinks
