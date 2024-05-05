import numpy as np


def detect_microsaccades(x_positions, y_positions, timestamps):
    """
    Detect microsaccades in eye movement data.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to the eye movement data.

    Returns:
    - dict: Dictionary containing information about detected microsaccades.
    """
    microsaccades = []
    window_size = 15

    for i in range(len(timestamps) - window_size):
        window_x = x_positions[i:i+window_size]
        window_y = y_positions[i:i+window_size]
        window_timestamps = timestamps[i:i+window_size]

        velocities = compute_velocity(window_x, window_y, window_timestamps)
        peak_indices = find_velocity_peaks(velocities)

        for peak_index in peak_indices:
            amplitude = compute_amplitude(window_x, window_y, peak_index)
            if is_microsaccade(velocities, peak_index, amplitude):
                start_time = timestamps[i + peak_index - 1]
                end_time = timestamps[i + peak_index + 1]
                duration = end_time - start_time
                peak_velocity = velocities[peak_index]

                microsaccade = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'amplitude': amplitude,
                    'duration': duration,
                    'peak_velocity': peak_velocity
                }
                microsaccades.append(microsaccade)

    return microsaccades

def compute_velocity(x_positions, y_positions, timestamps):
    """
    Compute velocity of eye movements based on change in position over time.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to the eye movement data.

    Returns:
    - array: Velocity values for each sample in the eye movement data.
    """
    velocities = []
    for i in range(1, len(timestamps)):
        dx = x_positions[i] - x_positions[i-1]
        dy = y_positions[i] - y_positions[i-1]
        dt = timestamps[i] - timestamps[i-1]
        velocity = np.sqrt(dx**2 + dy**2)/dt
        velocities.append(velocity)
    return velocities

def find_velocity_peaks(velocities):
    """
    Find peaks in velocity data.

    Args:
    - velocities (array): Velocity values.

    Returns:
    - list: Indices of peaks in the velocity data.
    """
    peak_indices = []
    for i in range(1, len(velocities) - 1):
        if velocities[i] > velocities[i-1] and velocities[i] > velocities[i+1]:
            peak_indices.append(i)
    return peak_indices

def is_microsaccade(velocities, peak_index, amplitude):
    """
    Determine if a peak in velocity corresponds to a microsaccade.

    Args:
    - velocities (array): Velocity values.
    - peak_index (int): Index of the peak in the velocity data.

    Returns:
    - bool: True if the peak is identified as a microsaccade, False otherwise.
    """
    min_velocity = 2

    if velocities[peak_index] > min_velocity:
        if amplitude < 1.0:
            return True
    else:
        return False

def compute_amplitude(x_positions, y_positions, peak_index):
    """
    Compute the amplitude of a microsaccade.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - peak_index (int): Index of the peak in the velocity data corresponding to the microsaccade.

    Returns:
    - float: Amplitude of the microsaccade.
    """
    dx = x_positions[peak_index] - x_positions[peak_index - 1]
    dy = y_positions[peak_index] - y_positions[peak_index - 1]
    amplitude = np.sqrt(dx**2 + dy**2)
    return amplitude

