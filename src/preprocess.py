import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional


def filter_data(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    cutoff_frequency: float = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a low-pass filter to eye movement data using a rolling mean.
    
    This function applies a moving average filter to smooth eye movement data.
    The window size is calculated based on the cutoff frequency.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        cutoff_frequency: Cutoff frequency for the low-pass filter in Hz.
            Defaults to 20 Hz.
    
    Returns:
        Tuple containing:
            - filtered_x: Filtered X positions
            - filtered_y: Filtered Y positions
    """
    min_periods = 1
    window_size = int(900/cutoff_frequency)
    
    filtered_x = pd.Series(x_positions).rolling(window=window_size, min_periods=min_periods).mean().values
    filtered_y = pd.Series(y_positions).rolling(window=window_size, min_periods=min_periods).mean().values
    
    return filtered_x, filtered_y

def remove_blinks(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    blink_threshold: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove blinks from eye movement data.
    
    This function identifies and removes blink periods from eye movement data
    based on temporal gaps in the data.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        blink_threshold: Maximum duration of a blink in milliseconds.
            Defaults to 10 ms.
    
    Returns:
        Tuple containing:
            - cleaned_x: X positions with blinks removed
            - cleaned_y: Y positions with blinks removed
    """

    time_diffs = np.diff(np.arange(len(x_positions)))
    
    blink_indices = np.where(time_diffs > blink_threshold)[0]
    
    cleaned_x = np.delete(x_positions, blink_indices)
    cleaned_y = np.delete(y_positions, blink_indices)
    
    return cleaned_x, cleaned_y

def correct_baseline_drift(
    x_positions: np.ndarray,
    y_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct baseline drift in eye movement data.
    
    This function corrects for baseline drift in eye movement data by fitting
    a linear regression and subtracting the trend.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
    
    Returns:
        Tuple containing:
            - x_positions: Original X positions
            - corrected_y: Y positions with baseline drift corrected
    """

    slope, intercept = np.polyfit(x_positions, y_positions, 1)
    corrected_y = y_positions - (slope * x_positions + intercept)
    
    return x_positions, corrected_y

def interpolate_data(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    sampling_freq: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate eye movement data to a uniform sampling rate.
    
    This function interpolates eye movement data to achieve a uniform sampling rate
    if the original data has non-uniform sampling.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        sampling_freq: The desired sampling frequency in Hz. If None, no interpolation
            is performed. Defaults to None.
    
    Returns:
        Tuple containing:
            - x_inter: Interpolated X positions
            - y_inter: Interpolated Y positions
            - t_inter: Interpolated timestamps
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

def low_pass_filter_eye_positions(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    cutoff_frequency: float,
    sampling_rate: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a low-pass Butterworth filter to eye position data.
    
    This function applies a Butterworth low-pass filter to smooth eye movement data.
    The filter is applied to both X and Y positions.
    
    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        cutoff_frequency: Cutoff frequency for the low-pass filter in Hz.
        sampling_rate: Sampling rate of the eye-tracking data in Hz.
        order: The order of the Butterworth filter. Defaults to 4.
    
    Returns:
        Tuple containing:
            - filtered_x: Filtered X positions
            - filtered_y: Filtered Y positions
    """
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    filtered_x = filtfilt(b, a, x_positions)
    filtered_y = filtfilt(b, a, y_positions)

    return filtered_x, filtered_y
