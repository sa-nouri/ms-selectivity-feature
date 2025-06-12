from typing import TypedDict

import numpy as np


class VelocityParams(TypedDict):
    """Parameters for velocity computation.

    Attributes:
        window_size: Size of the window for velocity computation in samples.
    """

    window_size: int


def compute_velocity(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    start_idx: int = None,
    end_idx: int = None,
    params: VelocityParams = None,
) -> tuple[tuple[np.ndarray, np.ndarray], float, float]:
    """Compute velocity of eye movements.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        start_idx: Starting index for velocity computation.
        end_idx: Ending index for velocity computation.
        params: Dictionary containing velocity computation parameters:
            - window_size: Size of the window for velocity computation in samples.

    Returns:
        Tuple containing:
            - Tuple of velocity arrays (vx, vy)
            - Standard deviation of x velocity
            - Standard deviation of y velocity
    """
    if params is None:
        params = {"window_size": 5}

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(x_positions)

    # Compute velocity using central difference
    dt = np.diff(timestamps[start_idx:end_idx])
    dx = np.diff(x_positions[start_idx:end_idx])
    dy = np.diff(y_positions[start_idx:end_idx])

    vx = dx / dt
    vy = dy / dt

    # Pad arrays to match input length
    vx = np.pad(vx, (1, 0), mode="edge")
    vy = np.pad(vy, (1, 0), mode="edge")

    # Compute standard deviations
    sigma_vx = np.std(vx)
    sigma_vy = np.std(vy)

    return (vx, vy), sigma_vx, sigma_vy


def compute_amplitude(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """Compute amplitude of eye movement.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        start_idx: Starting index for amplitude computation.
        end_idx: Ending index for amplitude computation.

    Returns:
        Amplitude of the eye movement in degrees.
    """
    dx = x_positions[end_idx] - x_positions[start_idx]
    dy = y_positions[end_idx] - y_positions[start_idx]
    return np.sqrt(dx**2 + dy**2)


def compute_direction(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """Compute direction of eye movement.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        start_idx: Starting index for direction computation.
        end_idx: Ending index for direction computation.

    Returns:
        Direction of the eye movement in radians.
    """
    dx = x_positions[end_idx] - x_positions[start_idx]
    dy = y_positions[end_idx] - y_positions[start_idx]
    return np.arctan2(dy, dx)


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
