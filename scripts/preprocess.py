import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


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

def interpolate_data(x_positions, y_positions, timestamps, sampling_freq=None):
    """
    Interpolate x and y positions based on uniform timestamps if sampling is non-uniform.

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Corresponding timestamps of eye movements.
    - sampling_freq (float, optional): The desired sampling frequency in Hz. If None, no interpolation is done.

    Returns:
    - x_inter (array): Interpolated X positions.
    - y_inter (array): Interpolated Y positions.
    - t_inter (array): Interpolated timestamps.
    """
    
    if sampling_freq is None:
        return x_positions, y_positions, timestamps

    total_time = timestamps[-1] - timestamps[0]
    frame_duration = 1000 / sampling_freq  # Convert Hz to milliseconds
    num_samples = int(np.round(total_time / frame_duration)) + 1

    t_inter = np.linspace(timestamps[0], timestamps[-1], num_samples)

    x_inter = np.interp(t_inter, timestamps, x_positions)
    y_inter = np.interp(t_inter, timestamps, y_positions)

    return x_inter, y_inter, t_inter

def low_pass_filter_eye_positions(x_positions, y_positions, cutoff_frequency, sampling_rate, order=4):
    """
    Apply a low-pass Butterworth filter to eye position data (x and y positions).
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - cutoff_frequency (float): Cutoff frequency for the low-pass filter in Hz.
    - sampling_rate (float): Sampling rate of the eye-tracking data in Hz.
    - order (int): The order of the Butterworth filter (default is 4).
    
    Returns:
    - filtered_x (array): Filtered X positions.
    - filtered_y (array): Filtered Y positions.
    """
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    filtered_x = filtfilt(b, a, x_positions)
    filtered_y = filtfilt(b, a, y_positions)

    return filtered_x, filtered_y

def low_pass_filter_eye_positions(x_positions, y_positions, cutoff_frequency, sampling_rate, order=4):
    """
    Apply a low-pass Butterworth filter to eye position data (x and y positions).

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - cutoff_frequency (float): Cutoff frequency for the low-pass filter in Hz.
    - sampling_rate (float): Sampling rate of the eye-tracking data in Hz.
    - order (int): The order of the Butterworth filter (default is 4).

    Returns:
    - filtered_x (array): Filtered X positions.
    - filtered_y (array): Filtered Y positions.
    """
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    filtered_x = filtfilt(b, a, x_positions)
    filtered_y = filtfilt(b, a, y_positions)

    return filtered_x, filtered_y
