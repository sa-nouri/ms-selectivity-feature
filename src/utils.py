import numpy as np
from typing import Tuple, Union, Optional


def compute_velocity(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    apply_smoothing: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Compute the velocity of eye movements based on the two-point method.

    This function computes eye movement velocities using the two-point method, applies interpolation
    for missing data, optionally applies a five-point running average, and calculates the median-based
    standard deviation of the horizontal and vertical velocity components.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to each eye movement position.
        apply_smoothing: Whether to apply a five-point running average to smooth velocities.
            Defaults to False.

    Returns:
        Tuple containing:
            - velocities: Array of smoothed velocity values for each time point.
            - sigma_vx: Median-based standard deviation for the horizontal (x) gaze positions.
            - sigma_vy: Median-based standard deviation for the vertical (y) gaze positions.
    """
    valid_indices = ~np.isnan(x_positions) & ~np.isnan(y_positions)
    x_positions = np.interp(
        timestamps, timestamps[valid_indices], x_positions[valid_indices]
    )
    y_positions = np.interp(
        timestamps, timestamps[valid_indices], y_positions[valid_indices]
    )

    velocities = []
    for i in range(1, len(timestamps)):
        dx = x_positions[i] - x_positions[i - 1]
        dy = y_positions[i] - y_positions[i - 1]
        dt = timestamps[i] - timestamps[i - 1]
        velocity = np.sqrt(dx**2 + dy**2) / dt
        velocities.append(velocity)

    velocities = np.array(velocities)
    # velocities *= 1e3  # degree per second

    if apply_smoothing:
        velocities = np.convolve(velocities, np.ones(5) / 5, mode="same")
        velocities[:2] = velocities[2]  # Handle edge cases for smoothing
        velocities[-2:] = velocities[-3]

    vx = np.diff(x_positions) / np.diff(timestamps)
    vy = np.diff(y_positions) / np.diff(timestamps)

    sigma_vx = np.sqrt(np.median((vx - np.median(vx)) ** 2))
    sigma_vy = np.sqrt(np.median((vy - np.median(vy)) ** 2))

    return velocities, sigma_vx, sigma_vy


def compute_amplitude(
    x_positions: np.ndarray, y_positions: np.ndarray, start_index: int, end_index: int
) -> float:
    """Compute the amplitude of an eye movement from start to end positions.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        start_index: Starting index of the eye movement.
        end_index: Ending index of the movement.

    Returns:
        The total amplitude of the eye movement (displacement) in degrees.
    """

    dx = x_positions[end_index] - x_positions[start_index]
    dy = y_positions[end_index] - y_positions[start_index]

    amplitude = np.sqrt(dx**2 + dy**2)

    return amplitude


def compute_partial_velocity(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Tuple[float, float]:
    """Compute the velocity between two specific points in the eye movement data.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to each eye movement position.
        start_idx: Starting index for velocity calculation.
        end_idx: Ending index for velocity calculation.

    Returns:
        Tuple containing:
            - velocity_x: Horizontal velocity component.
            - velocity_y: Vertical velocity component.
    """
    vx = x_positions[end_idx] - x_positions[start_idx]
    vy = y_positions[end_idx] - y_positions[start_idx]
    dt = timestamps[end_idx] - timestamps[start_idx]

    velocity_x = vx / dt
    velocity_y = vy / dt

    return velocity_x, velocity_y


def compute_velocity_consecutive(
    x_position: np.ndarray, y_position: np.ndarray, timestamp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute consecutive velocities in horizontal and vertical directions.

    This function calculates the consecutive velocities in both horizontal (x) and vertical (y)
    directions, as well as the combined Pythagorean velocity.

    Args:
        x_position: Array of X positions of eye movements.
        y_position: Array of Y positions of eye movements.
        timestamp: Array of timestamps corresponding to each eye movement position.

    Returns:
        Tuple containing:
            - x_velocity: Array of horizontal velocities.
            - y_velocity: Array of vertical velocities.
            - pythagorean_velocity: Array of combined velocities based on the Pythagorean theorem.
    """
    x_velocity = np.append(0, np.abs(np.diff(x_position) / np.diff(timestamp)))
    y_velocity = np.append(0, np.abs(np.diff(y_position) / np.diff(timestamp)))

    Pythagorean_velocity = np.sqrt(np.square(x_velocity) + np.square(y_velocity))
    return x_velocity, y_velocity, Pythagorean_velocity
