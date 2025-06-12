"""Module for detecting and validating glitches in eye tracking data.

This module provides functions for detecting and validating glitches in eye tracking data.
Glitches are sudden, large changes in eye position that are likely due to tracking errors
or other artifacts rather than actual eye movements.
"""

from typing import Dict, List, Optional, TypedDict

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

    def __init__(self, params: GlitchParams):
        """Initialize the glitch detector with parameters.

        Args:
            params: Dictionary containing detection parameters:
                - min_duration: Minimum duration (in seconds) required for a valid glitch
                - max_duration: Maximum duration (in seconds) allowed for a valid glitch
                - velocity_threshold: Velocity threshold for glitch detection
        """
        self.params = params

    def detect_glitches(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[dict]:
        """Detect glitches in eye movement data.

        Args:
            x_positions: Array of X positions of eye movements
            y_positions: Array of Y positions of eye movements
            timestamps: Array of timestamps corresponding to the eye movement data

        Returns:
            List of dictionaries containing information about detected glitches, where
            each dictionary contains:
                - start_time: Starting time of the glitch
                - end_time: Ending time of the glitch
                - duration: Duration of the glitch in seconds
                - amplitude: Amplitude of the glitch in degrees
                - direction: Direction of the glitch in radians
        """
        if np.isnan(x_positions).any() or np.isnan(y_positions).any():
            raise ValueError("Input arrays must not contain NaN values")

        velocities, _, _ = compute_velocity(x_positions, y_positions, timestamps)
        velocity_magnitudes = np.sqrt(velocities[0] ** 2 + velocities[1] ** 2)

        glitches = []
        current_glitch = None

        for i in range(1, len(velocity_magnitudes) - 1):
            if velocity_magnitudes[i] >= self.params["velocity_threshold"]:
                if current_glitch is None:
                    current_glitch = {
                        "start_time": timestamps[i],
                        "end_time": timestamps[i],
                        "duration": 0,
                        "amplitude": 0,
                        "direction": 0,
                    }
                current_glitch["end_time"] = timestamps[i]
            elif current_glitch is not None:
                current_glitch["duration"] = (
                    current_glitch["end_time"] - current_glitch["start_time"]
                )
                dx = x_positions[i] - x_positions[i - current_glitch["duration"]]
                dy = y_positions[i] - y_positions[i - current_glitch["duration"]]
                current_glitch["amplitude"] = np.sqrt(dx**2 + dy**2)
                current_glitch["direction"] = np.arctan2(dy, dx)

                if (
                    current_glitch["duration"] >= self.params["min_duration"]
                    and current_glitch["duration"] <= self.params["max_duration"]
                ):
                    glitches.append(current_glitch)

                current_glitch = None

        # Handle the last glitch if it exists
        if current_glitch is not None:
            current_glitch["duration"] = (
                current_glitch["end_time"] - current_glitch["start_time"]
            )
            dx = x_positions[-1] - x_positions[-1 - current_glitch["duration"]]
            dy = y_positions[-1] - y_positions[-1 - current_glitch["duration"]]
            current_glitch["amplitude"] = np.sqrt(dx**2 + dy**2)
            current_glitch["direction"] = np.arctan2(dy, dx)

            if (
                current_glitch["duration"] >= self.params["min_duration"]
                and current_glitch["duration"] <= self.params["max_duration"]
            ):
                glitches.append(current_glitch)

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
    return detector.detect_glitches(x_positions, y_positions, timestamps)


def validate_glitches(
    glitches: list[dict],
    min_duration: float = 0,
    max_duration: float = None,
) -> list[dict]:
    """Validate glitches based on duration.

    Args:
        glitches: List of dictionaries containing information about detected glitches
        min_duration: Minimum duration (in seconds) required for a valid glitch
        max_duration: Maximum duration (in seconds) allowed for a valid glitch

    Returns:
        List of dictionaries containing information about validated glitches
    """
    validated = []
    for g in glitches:
        if g.get("duration", 0) < 0:
            raise ValueError("Invalid glitch data: negative values")
        if g.get("duration", 0) >= min_duration:
            if max_duration is None or g.get("duration", 0) < max_duration:
                validated.append(g)
    return validated
