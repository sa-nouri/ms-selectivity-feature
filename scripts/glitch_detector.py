import numpy as np
from scripts.utils import compute_velocity

def detect_glitches(x_positions, y_positions, timestamps, max_speed=1000):
    """
    Detect glitches in eye-tracking data based on velocity exceeding the maxSpeed threshold.
    Glitches are defined as velocities that exceed the max_speed and are labeled as noise.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to each eye movement position.
    - max_speed (float): Maximum allowed speed (in degrees/second) before classifying as a glitch (default is 1000).

    Returns:
    - glitch_indices (array): Indices of the detected glitches.
    - velocities (array): The computed velocities for each time step.
    - labels (array): Array where glitches are labeled as 'NOISE' and others as 'VALID'.
    """

    velocities = compute_velocity(x_positions, y_positions, timestamps)
    
    glitch_indices = np.where(velocities > max_speed)[0]
    
    labels = np.full(len(velocities), 'VALID')
    labels[glitch_indices] = 'NOISE'
    
    return glitch_indices, velocities, labels
