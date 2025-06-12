"""Module for detecting and validating glitches in eye tracking data.

This module provides functions for detecting and validating glitches in eye tracking data.
Glitches are sudden, large changes in eye position that are likely due to tracking errors
or other artifacts rather than actual eye movements.
"""

from typing import Dict, List, Optional, TypedDict, Any

import numpy as np

from .logger import logger
from .utils import compute_velocity


class GlitchParams(TypedDict):
    """Parameters for glitch detection.

    Attributes:
        min_duration: Minimum duration (in seconds) required for a valid glitch.
        max_duration: Maximum duration (in seconds) allowed for a valid glitch.
        velocity_threshold: Velocity threshold for glitch detection.
    """

    min_duration: float
    max_duration: float
    velocity_threshold: float


class GlitchDetector:
    """Detect glitches in eye movement data based on velocity thresholds."""

    def __init__(self, params: GlitchParams) -> None:
        """Initialize the glitch detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid glitch
                - max_duration: Maximum duration (in seconds) allowed for a valid glitch
                - velocity_threshold: Velocity threshold for glitch detection
        """
        self.params = params

    def detect(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
        """Detect glitches in eye tracking data."""
        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = self.params["threshold_multiplier"] * np.sqrt(sigma_vx**2 + sigma_vy**2)

        glitches = []
        current_glitch = None

        for i in range(1, len(velocities) - 1):
            velocity_x, velocity_y = compute_velocity(x, y, timestamps, i - 1, i)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            if velocity_magnitude >= velocity_threshold:
                if current_glitch is None:
                    current_glitch = {
                        "start_idx": i,
                        "end_idx": i,
                        "duration": 0,
                        "magnitude": 0
                    }
                current_glitch["end_idx"] = i
            elif current_glitch is not None:
                current_glitch["duration"] = current_glitch["end_idx"] - current_glitch["start_idx"]
                current_glitch["magnitude"] = velocity_magnitude

                if (current_glitch["duration"] >= self.params["min_duration"] and
                    current_glitch["duration"] <= self.params["max_duration"] and
                    current_glitch["magnitude"] >= self.params["min_magnitude"] and
                    current_glitch["magnitude"] <= self.params["max_magnitude"]):
                    glitches.append(current_glitch)

                current_glitch = None

        return glitches


def detect_glitches(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    min_duration: float = 0.001,
    max_duration: float = 0.01,
    velocity_threshold: float = 1000.0,
) -> list[dict]:
    """Detect glitches in eye movement data.

    Args:
        x_positions: Array of X positions of eye movements
        y_positions: Array of Y positions of eye movements
        timestamps: Array of timestamps corresponding to the eye movement data
        min_duration: Minimum duration (in seconds) required for a valid glitch
        max_duration: Maximum duration (in seconds) allowed for a valid glitch
        velocity_threshold: Velocity threshold for glitch detection

    Returns:
        List of dictionaries containing information about detected glitches, where each
        dictionary contains:
            - start_time: Starting time of the glitch
            - end_time: Ending time of the glitch
            - duration: Duration of the glitch in seconds
            - amplitude: Amplitude of the glitch in degrees
            - direction: Direction of the glitch in radians
    """
    params = {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "velocity_threshold": velocity_threshold,
    }
    detector = GlitchDetector(params)
    return detector.detect(x_positions, y_positions, timestamps)


def validate_glitches(
    glitches: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    max_duration: Optional[float] = None
) -> list[dict[str, Any]]:
    """Validate detected glitches."""
    if not glitches:
        return []

    valid_glitches = []
    for glitch in glitches:
        if max_duration is not None and glitch["duration"] > max_duration:
            continue
        valid_glitches.append(glitch)

    return valid_glitches
