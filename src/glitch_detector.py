"""Module for detecting and validating glitches in eye tracking data.

This module provides functions for detecting and validating glitches in eye tracking data.
Glitches are sudden, large changes in eye position that are likely due to tracking errors
or other artifacts rather than actual eye movements.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union

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
    """Detector for glitches in eye tracking data."""

    def __init__(self, threshold_multiplier: Union[float, dict] = 5.0) -> None:
        """Initialize the glitch detector.

        Args:
            threshold_multiplier: Multiplier for velocity threshold (float) or dict of parameters
        """
        if isinstance(threshold_multiplier, dict):
            self.threshold_multiplier = float(
                threshold_multiplier.get("velocity_threshold", 5.0)
            )
        else:
            self.threshold_multiplier = float(threshold_multiplier)

    def detect(
        self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray
    ) -> list[dict[str, Any]]:
        """Detect glitches in eye tracking data."""
        if len(x) < 2 or len(y) < 2 or len(timestamps) < 2:
            return []

        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = float(self.threshold_multiplier) * np.sqrt(
            sigma_vx**2 + sigma_vy**2
        )
        glitch_indices = np.where(
            np.sqrt(velocities[0] ** 2 + velocities[1] ** 2) > velocity_threshold
        )[0]
        glitches = []
        for idx in glitch_indices:
            glitches.append(
                {
                    "time": timestamps[idx],
                    "magnitude": np.sqrt(
                        velocities[0][idx] ** 2 + velocities[1][idx] ** 2
                    ),
                }
            )
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
    glitches: list[dict],
    min_magnitude: float = 0.0,
    max_magnitude: float = None,
) -> list[dict]:
    """Validate detected glitches.

    Args:
        glitches: List of detected glitches (dicts or tuples)
        min_magnitude: Minimum magnitude threshold
        max_magnitude: Maximum magnitude threshold

    Returns:
        List of validated glitches
    """
    if not isinstance(glitches, list):
        raise ValueError("glitches must be a list")
    validated = []
    for g in glitches:
        if isinstance(g, tuple) and len(g) == 2:
            # Convert tuple to dict
            g = {"time": g[0], "magnitude": g[1]}
        elif not isinstance(g, dict):
            raise ValueError(
                "Each glitch must be a dict or a tuple of (time, magnitude)"
            )

        # Check for required fields
        magnitude = float(g.get("magnitude", 0.0))
        time = g.get("time")

        if magnitude < 0:
            raise ValueError("Negative magnitude in glitch")
        if time is not None and time < 0:
            raise ValueError("Negative time in glitch")

        if magnitude < float(min_magnitude):
            continue
        if max_magnitude is not None and magnitude > float(max_magnitude):
            continue
        validated.append(g)
    return validated
