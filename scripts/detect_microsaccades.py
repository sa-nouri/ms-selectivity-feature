import numpy as np

from scripts.utils import compute_amplitude, compute_velocity


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
            if current_microsaccade is None:
                current_microsaccade = {'start': i, 'end': i}
                start_index = i
            current_microsaccade['end'] = i
            
            amplitude = compute_amplitude(x_positions, y_positions, start_index, i)
            
            if amplitude < amplitude_threshold:
                end_time = timestamps[i]
                duration = end_time - timestamps[start_index]
                peak_velocity = velocities[i]

                microsaccade = {
                    'start_time': timestamps[start_index],
                    'end_time': end_time,
                    'amplitude': amplitude,
                    'duration': duration,
                    'peak_velocity': peak_velocity
                }
                microsaccades.append(microsaccade)
            
            current_microsaccade = None
        else:
            current_microsaccade = None
    
    return microsaccades

