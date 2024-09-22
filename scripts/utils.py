import numpy as np
from scipy.signal import butter, filtfilt
from sklearn import mixture


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

    # velocities *= 1e3  # degree per second
    return velocities

def compute_amplitude(x_positions, y_positions, start_index, end_index):
    """
    Compute the amplitude of a eye movement from start to end positions.
    
    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - start_index (int): Starting index of the eye movement.
    - end_index (int): Ending index of the movement.
    
    Returns:
    - amplitude (float): The total amplitude of the eye movement (displacement).
    """
    
    dx = x_positions[end_index] - x_positions[start_index]
    dy = y_positions[end_index] - y_positions[start_index]
    
    amplitude = np.sqrt(dx**2 + dy**2)
    
    return amplitude

def compute_velocity_consecutive(x_position, y_position, timestamp):
    x_velocity = np.append(0, np.abs(np.diff(x_position)/np.diff(timestamp)))
    y_velocity = np.append(0, np.abs(np.diff(y_position)/np.diff(timestamp)))
    
    Pythagorean_velocity = np.sqrt((np.square(x_velocity)) + (np.square(y_velocity)))
    # Pythagorean_velocity *= 1e3  # degree per second
    return x_velocity, y_velocity, Pythagorean_velocity


def noise_threshold_extract(data_input):
    # Prefer to lower the values to about 20 deg/sec if the idea is to detect microsaccades
    threshold = 10
    idx = (data_input < threshold)
    gmm = mixture.GaussianMixture(n_components=2, 
                                  covariance_type='full',
                                  max_iter=100).fit(data_input[idx].reshape(-1, 1))    
    means = gmm.means_
    covariances = np.sqrt(gmm.covariances_)
    idd = np.argsort(means.reshape(2))
    value = float((means[idd[0]]+3*covariances[idd[0]])[0])
    return value

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


def validate_saccades_min_duration(saccades, timestamps, min_duration):
    """
    Validate the detected saccades based on their duration.
    If a saccade is shorter than min_duration, it is considered noise and discarded.

    Args:
    - saccades (list of tuples): List of detected saccades, where each saccade is represented as a tuple (start_index, end_index).
    - timestamps (array): Array of timestamps corresponding to the eye movement data.
    - min_duration (float): The minimal duration (in seconds or milliseconds) required for a valid saccade.

    Returns:
    - valid_saccades (list of tuples): List of valid saccades (those that pass the duration filter).
    """
    valid_saccades = []
    
    for start_index, end_index in saccades:
        duration = timestamps[end_index] - timestamps[start_index]
        
        if duration >= min_duration:
            valid_saccades.append((start_index, end_index))
        else:
            print(f"Saccade from {timestamps[start_index]} to {timestamps[end_index]} discarded (duration: {duration} < {min_duration})")
    
    return valid_saccades
