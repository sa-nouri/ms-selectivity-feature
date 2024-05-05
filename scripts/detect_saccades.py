import numpy as np


def detect_saccades(x_positions, y_positions):
    """
    Detect saccades in eye movement data based on changes in gaze direction.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    
    Returns:
    - dict: Dictionary containing information about detected saccades.
    """
    saccades = []
    current_saccade = None
    saccade_start = None
    
    for i in range(1, len(x_positions)):
        dx = x_positions[i] - x_positions[i-1]
        dy = y_positions[i] - y_positions[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        
        
        if current_saccade is None:
            current_saccade = {'start': i, 'end': i, 'distance': distance, 'amplitude': 0, 'velocity': 0, 'duration': 0}
            saccade_start = i
        else:
            current_saccade['end'] = i
        
        if current_saccade is not None:
            current_saccade['duration'] = i - saccade_start
            
            dx_amplitude = x_positions[current_saccade['end']] - x_positions[saccade_start]
            dy_amplitude = y_positions[current_saccade['end']] - y_positions[saccade_start]
            amplitude = np.sqrt(dx_amplitude**2 + dy_amplitude**2)
            
            current_saccade['amplitude'] = amplitude
            
            if current_saccade['duration'] > 0:
                current_saccade['velocity'] = amplitude / current_saccade['duration']
            else:
                current_saccade['velocity'] = 0
        
            saccades.append(current_saccade)
            current_saccade = None
            
    return saccades
