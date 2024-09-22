import numpy as np

from scripts.utils import compute_amplitude, compute_velocity


def detect_saccades(x_positions, y_positions, timestamps, velocity_multiplier=6, amplitude_threshold=1.0):
    """
    Detect saccades in eye movement data based on changes in gaze direction.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    
    Returns:
    - dict: Dictionary containing information about detected saccades.
    """
    
    velocities = compute_velocity(x_positions, y_positions, timestamps)
    median_velocity = np.median(velocities)
    velocity_threshold = velocity_multiplier * median_velocity
    
    
    saccades = []
    current_saccade = None
    saccade_start = None
    
    for i in range(1, len(velocities) - 1):
        dx = x_positions[i] - x_positions[i-1]
        dy = y_positions[i] - y_positions[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        
        
        if velocities[i] > velocity_threshold:
            if current_saccade is None:
                current_saccade = {'start': i, 'end': i, 'distance': distance, 'amplitude': 0, 'velocity': 0, 'duration': 0}
                saccade_start = i
            else:
                current_saccade['end'] = i
        
        if current_saccade is not None:
            current_saccade['duration'] = timestamps[i] - timestamps[saccade_start]
            
            amplitude = compute_amplitude(x_positions, y_positions, saccade_start, i)
            current_saccade['amplitude'] = amplitude
            
            if current_saccade['duration'] > 0:
                current_saccade['velocity'] = velocities[i]
            else:
                current_saccade['velocity'] = 0
        
            saccades.append(current_saccade)
            current_saccade = None
        
        if current_saccade is not None and velocities[i] <= velocity_threshold:
            current_saccade = None
            
    return saccades
