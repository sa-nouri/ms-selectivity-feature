import numpy as np
import pandas as pd


def filter_data(x_positions, y_positions, cutoff_frequency=20):
    """
    Apply a low-pass filter to eye movement data.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - cutoff_frequency (float): Cutoff frequency for the low-pass filter.
    
    Returns:
    - tuple: Filtered x and y positions.
    """
    min_periods = 1
    window_size = int(900/cutoff_frequency)
    
    filtered_x = pd.Series(x_positions).rolling(window=window_size, min_periods=min_periods).mean().values
    filtered_y = pd.Series(y_positions).rolling(window=window_size, min_periods=min_periods).mean().values
    
    return filtered_x, filtered_y

def remove_blinks(x_positions, y_positions, blink_threshold=10):
    """
    Remove blinks from eye movement data.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - blink_threshold (int): Maximum duration of a blink in milliseconds.
    
    Returns:
    - tuple: Eye movement data with blinks removed.
    """

    time_diffs = np.diff(np.arange(len(x_positions)))
    
    blink_indices = np.where(time_diffs > blink_threshold)[0]
    
    cleaned_x = np.delete(x_positions, blink_indices)
    cleaned_y = np.delete(y_positions, blink_indices)
    
    return cleaned_x, cleaned_y

def correct_baseline_drift(x_positions, y_positions):
    """
    Correct baseline drift in eye movement data.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    
    Returns:
    - tuple: Eye movement data with baseline drift corrected.
    """

    slope, intercept = np.polyfit(x_positions, y_positions, 1)
    corrected_y = y_positions - (slope * x_positions + intercept)
    
    return x_positions, corrected_y
