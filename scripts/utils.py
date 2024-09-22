import numpy as np


def compute_velocity(x_positions, y_positions, timestamps, apply_smoothing=False):
    """
    Compute the velocity of eye movements based on the two-point method, apply interpolation for missing data,
    optionally apply a five-point running average, and calculate the median-based standard deviation of the horizontal and
    vertical velocity components (sigma_vx, sigma_vy).

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to each eye movement position.
    - sampling_rate (float, optional): Desired sampling rate, if specified for interpolation.
    - apply_smoothing (bool, optional): Whether to apply a five-point running average to smooth velocities.

    Returns:
    - velocities (array): Smoothed velocity values for each time point.
    - sigma_vx (float): Median-based standard deviation for the horizontal (x) gaze positions.
    - sigma_vy (float): Median-based standard deviation for the vertical (y) gaze positions.
    """
    valid_indices = ~np.isnan(x_positions) & ~np.isnan(y_positions)
    x_positions = np.interp(timestamps, timestamps[valid_indices], x_positions[valid_indices])
    y_positions = np.interp(timestamps, timestamps[valid_indices], y_positions[valid_indices])
    
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
        velocities = np.convolve(velocities, np.ones(5)/5, mode='same')
        velocities[:2] = velocities[2]  # Handle edge cases for smoothing
        velocities[-2:] = velocities[-3]
    
    vx = np.diff(x_positions) / np.diff(timestamps)
    vy = np.diff(y_positions) / np.diff(timestamps)
    
    sigma_vx = np.sqrt(np.median((vx - np.median(vx))**2))
    sigma_vy = np.sqrt(np.median((vy - np.median(vy))**2))

    return velocities, sigma_vx, sigma_vy

def compute_amplitude(x_positions, y_positions, start_index, end_index):
    """
    Compute the amplitude of a eye movement from start to end positions.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - start_index (int): Starting index of the eye movement.
    - end_index (int): Ending index of the movement.
    
    Returns:
    - amplitude (float): The total amplitude of the eye movement (displacement).
    """
    
    dx = x_positions[end_index] - x_positions[start_index]
    dy = y_positions[end_index] - y_positions[start_index]
    
    amplitude = np.sqrt(dx**2 + dy**2)
    
    return amplitude

def compute_partial_velocity(x_positions, y_positions, timestamps, start_idx, end_idx):
    vx = x_positions[end_idx] - x_positions[start_idx]
    vy = y_positions[end_idx] - y_positions[start_idx]
    dt = timestamps[end_idx] - timestamps[start_idx]
    
    velocity_x = vx / dt
    velocity_y = vy / dt
    
    return velocity_x, velocity_y

def compute_velocity_consecutive(x_position, y_position, timestamp):
    """
    Compute consecutive velocities in the horizontal (x) and vertical (y) directions, 
    and the combined Pythagorean velocity.

    Args:
    - x_position (array): X positions of eye movements.
    - y_position (array): Y positions of eye movements.
    - timestamp (array): Timestamps corresponding to each eye movement position.

    Returns:
    - x_velocity (array): Horizontal velocities.
    - y_velocity (array): Vertical velocities.
    - Pythagorean_velocity (array): Combined velocities based on the Pythagorean theorem.
    """
    x_velocity = np.append(0, np.abs(np.diff(x_position)/np.diff(timestamp)))
    y_velocity = np.append(0, np.abs(np.diff(y_position)/np.diff(timestamp)))
    
    Pythagorean_velocity = np.sqrt(np.square(x_velocity) + np.square(y_velocity))
    return x_velocity, y_velocity, Pythagorean_velocity
