import numpy as np


def detect_microsaccades(x_positions, y_positions, timestamps, velocity_multiplier=6, amplitude_threshold=1.0):
    """
    Detect microsaccades in eye movement data.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to the eye movement data.
    - velocity_multiplier (float): Multiplier for the median velocity threshold (default is 6).
    - amplitude_threshold (float): Maximum amplitude to classify as a microsaccade (default is 1.0).

    Returns:
    - microsaccades (list of dict): Detected microsaccades with start time, end time, amplitude, duration, and peak velocity.
    """
    
    velocities = compute_velocity(x_positions, y_positions, timestamps)
    median_velocity = np.median(velocities)
    velocity_threshold = velocity_multiplier * median_velocity
    
    microsaccades = []

    for i in range(1, len(velocities) - 1):
        if velocities[i] > velocity_threshold:
            
            amplitude = compute_amplitude(x_positions, y_positions, i)
            
            if amplitude < amplitude_threshold:
                start_time = timestamps[i - 1]
                end_time = timestamps[i + 1]
                duration = end_time - start_time
                peak_velocity = velocities[i]
                
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

def compute_amplitude(x_positions, y_positions, peak_index):
    """
    Compute the amplitude of a movement at a specific index.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - index (int): The index in the array corresponding to the peak movement.

    Returns:
    - float: Amplitude of the microsaccade.
    """
    dx = x_positions[peak_index] - x_positions[peak_index - 1]
    dy = y_positions[peak_index] - y_positions[peak_index - 1]
    amplitude = np.sqrt(dx**2 + dy**2)
    return amplitude

