"""Module for detecting and validating glitches in eye tracking data.

This module provides functions for detecting and validating glitches in eye tracking data.
Glitches are sudden, large changes in eye position that are likely due to tracking errors
or other artifacts rather than actual eye movements.
"""

import numpy as np
from typing import Dict, List, Optional

from .logger import logger


def detect_glitches(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 5.0,
    window_size: int = 5,
) -> List[Dict[str, float]]:
    """Detect glitches in eye tracking data.

    A glitch is defined as a sudden, large change in eye position that exceeds the
    specified threshold. The function uses a sliding window approach to detect these
    sudden changes.

    Args:
        x_positions: Array of x-coordinates of eye positions
        y_positions: Array of y-coordinates of eye positions
        timestamps: Array of timestamps corresponding to the positions
        threshold: Threshold for glitch detection (default: 5.0)
        window_size: Size of the sliding window for detection (default: 5)

    Returns:
        List of dictionaries containing glitch information:
        - time: Timestamp of the glitch
        - magnitude: Magnitude of the position change
        - direction: Direction of the position change in degrees

    Raises:
        ValueError: If input arrays have different lengths or contain NaN values
    """
    logger.info("Starting glitch detection")
    logger.debug(f"Parameters: threshold={threshold}, window_size={window_size}")

    if len(x_positions) != len(y_positions) or len(x_positions) != len(timestamps):
        raise ValueError("Input arrays must have the same length")

    if np.isnan(x_positions).any() or np.isnan(y_positions).any():
        raise ValueError("Input arrays must not contain NaN values")

    # Compute position changes
    dx = np.diff(x_positions)
    dy = np.diff(y_positions)
    magnitudes = np.sqrt(dx**2 + dy**2)

    # Find glitches
    glitch_indices = np.where(magnitudes > threshold)[0]

    # Group consecutive glitches
    glitches = []
    if len(glitch_indices) > 0:
        current_group = [glitch_indices[0]]

        for i in range(1, len(glitch_indices)):
            if glitch_indices[i] - glitch_indices[i - 1] <= window_size:
                current_group.append(glitch_indices[i])
            else:
                # Process current group
                if len(current_group) > 0:
                    mid_idx = current_group[len(current_group) // 2]
                    glitches.append(
                        {
                            "time": timestamps[mid_idx],
                            "magnitude": magnitudes[mid_idx],
                            "direction": np.degrees(
                                np.arctan2(dy[mid_idx], dx[mid_idx])
                            ),
                        }
                    )
                current_group = [glitch_indices[i]]

        # Process last group
        if len(current_group) > 0:
            mid_idx = current_group[len(current_group) // 2]
            glitches.append(
                {
                    "time": timestamps[mid_idx],
                    "magnitude": magnitudes[mid_idx],
                    "direction": np.degrees(np.arctan2(dy[mid_idx], dx[mid_idx])),
                }
            )

    logger.info(f"Found {len(glitches)} glitches")
    return glitches


def validate_glitches(
    glitches: List[Dict[str, float]],
    min_magnitude: Optional[float] = None,
    max_magnitude: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Validate detected glitches based on magnitude criteria.

    Args:
        glitches: List of glitch dictionaries from detect_glitches
        min_magnitude: Minimum magnitude threshold (optional)
        max_magnitude: Maximum magnitude threshold (optional)

    Returns:
        List of validated glitch dictionaries

    Raises:
        ValueError: If glitch data is invalid
    """
    logger.info("Starting glitch validation")
    logger.debug(
        f"Parameters: min_magnitude={min_magnitude}, max_magnitude={max_magnitude}"
    )

    if not glitches:
        return []

    validated = []
    for glitch in glitches:
        if "magnitude" not in glitch or "time" not in glitch:
            raise ValueError("Invalid glitch data: missing required fields")

        if glitch["magnitude"] < 0 or glitch["time"] < 0:
            raise ValueError("Invalid glitch data: negative values")

        if min_magnitude is not None and glitch["magnitude"] < min_magnitude:
            continue

        if max_magnitude is not None and glitch["magnitude"] >= max_magnitude:
            continue

        validated.append(glitch)

    logger.info(f"Validated {len(validated)} glitches")
    return validated
