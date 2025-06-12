from typing import Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.utils import compute_velocity_magnitude


class PreprocessParams(TypedDict):
    """Parameters for preprocessing eye movement data.

    Attributes:
        sampling_rate: Sampling rate of the eye movement data in Hz.
        filter_order: Order of the Butterworth filter.
        cutoff_freq: Cutoff frequency for the Butterworth filter in Hz.
    """

    sampling_rate: float
    filter_order: int
    cutoff_freq: float


def filter_data(
    x_positions: np.ndarray, y_positions: np.ndarray, cutoff_frequency: float = 20
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
    window_size = int(900 / cutoff_frequency)

    filtered_x = (
        pd.Series(x_positions)
        .rolling(window=window_size, min_periods=min_periods)
        .mean()
        .values
    )
    filtered_y = (
        pd.Series(y_positions)
        .rolling(window=window_size, min_periods=min_periods)
        .mean()
        .values
    )

    return filtered_x, filtered_y


def remove_blinks(
    x_positions: np.ndarray, y_positions: np.ndarray, blink_threshold: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Remove blinks from eye tracking data.

    Args:
        x_positions: X positions
        y_positions: Y positions
        blink_threshold: Threshold for blink detection (default: 2.0)

    Returns:
        Tuple of (cleaned_x, cleaned_y)
    """
    # Create mask for non-blink points
    mask = np.ones(len(x_positions), dtype=bool)

    # Find points where both x and y are NaN
    nan_mask = np.isnan(x_positions) | np.isnan(y_positions)

    # Find points where velocity exceeds threshold
    velocity_magnitude = compute_velocity_magnitude(
        x_positions, y_positions, np.arange(len(x_positions))
    )
    velocity_mask = np.zeros(len(x_positions), dtype=bool)
    velocity_mask[1:] = velocity_magnitude > blink_threshold

    # Combine masks
    mask = ~(nan_mask | velocity_mask)

    return x_positions[mask], y_positions[mask]


def correct_baseline_drift(
    x_positions: np.ndarray, y_positions: np.ndarray
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
    sampling_freq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate eye movement data to a uniform sampling rate.

    This function interpolates eye movement data to achieve a uniform sampling rate
    if the original data has non-uniform sampling.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        sampling_freq: The desired sampling frequency in Hz.

    Returns:
        Tuple containing:
            - x_inter: Interpolated X positions
            - y_inter: Interpolated Y positions
            - t_inter: Interpolated timestamps
    """
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
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a low-pass filter to eye position data.

    Args:
        x_positions: X positions
        y_positions: Y positions
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz

    Returns:
        Filtered x and y positions
    """
    if len(x_positions) < 4 or len(y_positions) < 4:
        return x_positions, y_positions

    # Design filter
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = butter(2, normalized_cutoff, btype="low")

    # Apply filter
    filtered_x = filtfilt(b, a, x_positions)
    filtered_y = filtfilt(b, a, y_positions)

    return filtered_x, filtered_y


def preprocess_data(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    params: PreprocessParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess eye movement data by filtering and interpolating.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        params: Dictionary containing preprocessing parameters:
            - sampling_rate: Sampling rate of the eye movement data in Hz.
            - filter_order: Order of the Butterworth filter.
            - cutoff_freq: Cutoff frequency for the Butterworth filter in Hz.

    Returns:
        Tuple containing:
            - Filtered and interpolated X positions
            - Filtered and interpolated Y positions
            - Interpolated timestamps
    """
    if np.isnan(x_positions).any() or np.isnan(y_positions).any():
        raise ValueError("Input arrays must not contain NaN values")

    # Interpolate missing values
    x_interp = np.interp(
        timestamps,
        timestamps[~np.isnan(x_positions)],
        x_positions[~np.isnan(x_positions)],
    )
    y_interp = np.interp(
        timestamps,
        timestamps[~np.isnan(y_positions)],
        y_positions[~np.isnan(y_positions)],
    )

    # Apply Butterworth filter
    nyquist = params["sampling_rate"] / 2
    normal_cutoff = params["cutoff_freq"] / nyquist
    b, a = butter(params["filter_order"], normal_cutoff, btype="low")
    x_filtered = filtfilt(b, a, x_interp)
    y_filtered = filtfilt(b, a, y_interp)

    return x_filtered, y_filtered, timestamps


def preprocess(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float = 1000.0,
    filter_order: int = 4,
    cutoff_freq: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess eye movement data with default parameters.

    Args:
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        timestamps: Array of timestamps corresponding to the eye movement data.
        sampling_rate: Sampling rate of the eye movement data in Hz.
        filter_order: Order of the Butterworth filter.
        cutoff_freq: Cutoff frequency for the Butterworth filter in Hz.

    Returns:
        Tuple containing:
            - Filtered and interpolated X positions
            - Filtered and interpolated Y positions
            - Interpolated timestamps
    """
    params = {
        "sampling_rate": sampling_rate,
        "filter_order": filter_order,
        "cutoff_freq": cutoff_freq,
    }
    return preprocess_data(x_positions, y_positions, timestamps, params)
