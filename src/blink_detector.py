import copy
from typing import Any, Optional, TypedDict, Union

import numpy as np

from .utils import compute_velocity


class BlinkParams(TypedDict):
    min_duration: float
    max_duration: float
    threshold_multiplier: float


class BlinkDetector:
    """Detector for blinks in eye tracking data."""

    def __init__(self, threshold_multiplier: Union[float, dict] = 5.0) -> None:
        """Initialize the blink detector.

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
        """Detect blinks in eye tracking data."""
        if len(x) < 2 or len(y) < 2 or len(timestamps) < 2:
            return []

        velocities, sigma_vx, sigma_vy = compute_velocity(x, y, timestamps)
        velocity_threshold = float(self.threshold_multiplier) * np.sqrt(
            sigma_vx**2 + sigma_vy**2
        )
        blink_indices = np.where(
            np.sqrt(velocities[0] ** 2 + velocities[1] ** 2) > velocity_threshold
        )[0]
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
    x_positions: np.ndarray, y_positions: np.ndarray, timestamps: np.ndarray
) -> list[dict]:
    """Detect blinks in eye tracking data.

    Args:
        x_positions: X positions
        y_positions: Y positions
        timestamps: Timestamps

    Returns:
        List of detected blinks
    """
    if len(x_positions) < 2 or len(y_positions) < 2 or len(timestamps) < 2:
        return []

    # Check for NaN values
    if (
        np.isnan(x_positions).any()
        or np.isnan(y_positions).any()
        or np.isnan(timestamps).any()
    ):
        raise ValueError("Input arrays contain NaN values")

    velocities, sigma_vx, sigma_vy = compute_velocity(
        x_positions, y_positions, timestamps
    )
    velocity_threshold = 5.0 * np.sqrt(sigma_vx**2 + sigma_vy**2)
    blink_indices = np.where(
        np.sqrt(velocities[0] ** 2 + velocities[1] ** 2) > velocity_threshold
    )[0]
    blinks = []
    for idx in blink_indices:
        blinks.append(
            {
                "start_time": timestamps[idx],
                "end_time": timestamps[idx + 1],
                "duration": timestamps[idx + 1] - timestamps[idx],
                "amplitude": np.sqrt(velocities[0][idx] ** 2 + velocities[1][idx] ** 2),
            }
        )
    return blinks


def validate_blinks(
    blinks: list[dict],
    min_duration: float = 0.0,
    min_amplitude: float = None,
    max_amplitude: float = None,
) -> list[dict]:
    """Validate detected blinks.

    Args:
        blinks: List of detected blinks (dicts or tuples)
        min_duration: Minimum duration threshold
        min_amplitude: Minimum amplitude threshold
        max_amplitude: Maximum amplitude threshold

    Returns:
        List of validated blinks
    """
    if not isinstance(blinks, list):
        raise ValueError("blinks must be a list")
    validated = []
    for b in blinks:
        if isinstance(b, tuple) and len(b) == 2:
            # Convert tuple to dict
            b = {
                "start_time": b[0],
                "end_time": b[1],
                "duration": b[1] - b[0],
                "amplitude": 0.0,
            }
        elif not isinstance(b, dict):
            raise ValueError(
                "Each blink must be a dict or a tuple of (start_time, end_time)"
            )

        # Check for required fields
        duration = float(b.get("duration", 0.0))
        amplitude = float(b.get("amplitude", 0.0))

        if duration < 0:
            raise ValueError("Negative duration in blink")
        if amplitude < 0:
            raise ValueError("Negative amplitude in blink")
        if b.get("start_time") is not None and b.get("end_time") is not None:
            if b["start_time"] > b["end_time"]:
                raise ValueError("start_time greater than end_time in blink")

        if duration < float(min_duration):
            continue
        if min_amplitude is not None and amplitude < float(min_amplitude):
            continue
        if max_amplitude is not None and amplitude > float(max_amplitude):
            continue
        validated.append(b)
    return validated
