import numpy as np

from scripts.utils import (compute_amplitude, compute_partial_velocity,
                           compute_velocity)


def detect_saccades(x_positions, y_positions, timestamps, params):
    """
    Detect saccades in eye movement data based on changes in gaze direction.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to the eye movement data.
    - params (dict): A dictionary containing:
        - 'min_duration': Minimum duration (in seconds) required for a valid saccade.
        - 'threshold_multiplier': Multiplier for sigma_vx and sigma_vy to set velocity threshold.
        - 'sampling_rate': The sampling rate for interpolation (optional).
    
    Returns:
    - list: List of dictionaries containing information about detected saccades.
    """
    velocities, sigma_vx, sigma_vy = compute_velocity(x_positions, y_positions, timestamps)
    # velocity_threshold = params['threshold_multiplier'] * np.sqrt(sigma_vx**2 + sigma_vy**2)
    
    velocity_threshold_x = sigma_vx * params['threshold_multiplier']
    velocity_threshold_y = sigma_vy * params['threshold_multiplier']
    
    saccades = []
    current_saccade = None
    saccade_start = None
    
    for i in range(1, len(velocities) - 1):
        
        velocity_x, velocity_y = compute_partial_velocity(x_positions, y_positions, timestamps, i-1, i)
        
        velocity_xy = (velocity_x**2 / velocity_threshold_x**2) + (velocity_y**2 / velocity_threshold_y**2)
        
        # if velocities[i] >= velocity_threshold:
        if velocity_xy > 1:
            if current_saccade is None:
                saccade_start = i
                current_saccade = {
                    'start': i, 
                    'end': i, 
                    'amplitude': 0, 
                    'velocity': 0, 
                    'duration': 0
                }
            current_saccade['end'] = i
        
        if current_saccade is not None:
            current_saccade['duration'] = timestamps[current_saccade['end']] - timestamps[saccade_start]
        
            if current_saccade['duration'] >= params['min_duration']:
                current_saccade['amplitude'] = compute_amplitude(x_positions, y_positions, saccade_start, current_saccade['end'])
                current_saccade['velocity'] = current_saccade['amplitude'] / current_saccade['duration']
                # current_saccade['velocity'] = velocities[i]

                saccades.append(current_saccade)

            current_saccade = None

    return saccades
