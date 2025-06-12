import numpy as np
from typing import Dict, List, TypedDict, Optional

from .utils import (compute_amplitude, compute_partial_velocity,
                   compute_velocity)


class MicrosaccadeParams(TypedDict):
    """Parameters for microsaccade detection.
    
    Attributes:
        min_duration: Minimum duration (in seconds) required for a valid microsaccade.
        threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set velocity threshold.
        amplitude_th: Maximum amplitude threshold for a valid microsaccade.
    """
    min_duration: float
    threshold_multiplier: float
    amplitude_th: float


class Microsaccade(TypedDict):
    """Information about a detected microsaccade.
    
    Attributes:
        start: Starting index of the microsaccade.
        end: Ending index of the microsaccade.
        amplitude: Amplitude of the microsaccade in degrees.
        velocity: Velocity of the microsaccade in degrees per second.
        duration: Duration of the microsaccade in seconds.
    """
    start: int
    end: int
    amplitude: float
    velocity: float
    duration: float


def detect_microsaccades(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    params: MicrosaccadeParams
) -> List[Microsaccade]:
    """Detect microsaccades in eye movement data based on changes in gaze direction.
    
    This function implements a velocity-based algorithm to detect microsaccades in eye movement data.
    It uses a combination of velocity thresholds and duration criteria to identify valid microsaccades.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        params: Dictionary containing detection parameters:
            - min_duration: Minimum duration (in seconds) required for a valid microsaccade.
            - threshold_multiplier: Multiplier for sigma_vx and sigma_vy to set velocity threshold.
            - amplitude_th: Maximum amplitude threshold for a valid microsaccade.
    
    Returns:
        List of dictionaries containing information about detected microsaccades, where each
        dictionary contains:
            - start: Starting index of the microsaccade
            - end: Ending index of the microsaccade
            - amplitude: Amplitude of the microsaccade in degrees
            - velocity: Velocity of the microsaccade in degrees per second
            - duration: Duration of the microsaccade in seconds
    """
    velocities, sigma_vx, sigma_vy = compute_velocity(x_positions, y_positions, timestamps)
    # velocity_threshold = params['threshold_multiplier'] * np.sqrt(sigma_vx**2 + sigma_vy**2)
    
    velocity_threshold_x = sigma_vx * params['threshold_multiplier']
    velocity_threshold_y = sigma_vy * params['threshold_multiplier']
    
    microsaccades = []
    current_saccade = None
    microsaccade_start = None
    
    for i in range(1, len(velocities) - 1):
        
        velocity_x, velocity_y = compute_partial_velocity(x_positions, y_positions, timestamps, i-1, i)
        velocity_xy = (velocity_x**2 / velocity_threshold_x**2) + (velocity_y**2 / velocity_threshold_y**2)
        
        # if velocities[i] >= velocity_threshold:
        if velocity_xy > 1:
            if current_saccade is None:
                microsaccade_start = i
                current_saccade = {
                    'start': i, 
                    'end': i, 
                    'amplitude': 0, 
                    'velocity': 0, 
                    'duration': 0
                }
            current_saccade['end'] = i
        
        if current_saccade is not None:
            current_saccade['duration'] = timestamps[current_saccade['end']] - timestamps[microsaccade_start]
        
            if current_saccade['duration'] >= params['min_duration']:
                current_saccade['amplitude'] = compute_amplitude(x_positions, y_positions, microsaccade_start, current_saccade['end'])
                current_saccade['velocity'] = current_saccade['amplitude'] / current_saccade['duration']
                # current_saccade['velocity'] = velocities[i]
                if current_saccade['amplitude'] <= params["amplitude_th"]:
                    microsaccades.append(current_saccade)

            current_saccade = None

    return microsaccades
