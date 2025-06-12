import numpy as np
from typing import Tuple, Literal

from scripts.utils import compute_velocity


def detect_glitches(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    max_speed: float = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect glitches in eye-tracking data based on velocity thresholds.
    
    This function identifies glitches in eye-tracking data by detecting velocities
    that exceed a maximum speed threshold. Glitches are labeled as 'NOISE' while
    valid data points are labeled as 'VALID'.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to each eye movement position.
        max_speed: Maximum allowed speed (in degrees/second) before classifying
            as a glitch. Defaults to 1000 degrees/second.
    
    Returns:
        Tuple containing:
            - glitch_indices: Array of indices where glitches were detected
            - velocities: Array of computed velocities for each time step
            - labels: Array of labels where glitches are marked as 'NOISE' and
                others as 'VALID'
    """

    velocities = compute_velocity(x_positions, y_positions, timestamps)
    
    glitch_indices = np.where(velocities > max_speed)[0]
    
    labels = np.full(len(velocities), 'VALID')
    labels[glitch_indices] = 'NOISE'
    
    return glitch_indices, velocities, labels
