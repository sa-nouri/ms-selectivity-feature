import numpy as np
from scipy.signal import butter, filtfilt
from sklearn import mixture


def compute_velocity(x_positions, y_positions, timestamps, sampling_rate=None):
    """
    Compute the velocity of eye movements based on the two-point method, apply interpolation for missing data,
    apply a five-point running average, and calculate the median-based standard deviation of the horizontal and
    vertical velocity components (σvx, σvy).

    Args:
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - timestamps (array): Timestamps corresponding to each eye movement position.
    - sampling_rate (float, optional): Desired sampling rate, if specified for interpolation.

    Returns:
    - velocities (array): Smoothed velocity values for each time point.
    - sigma_vx (float): Median-based standard deviation for the horizontal (x) gaze positions.
    - sigma_vy (float): Median-based standard deviation for the vertical (y) gaze positions.
    """
    
    valid_indices = ~np.isnan(x_positions) & ~np.isnan(y_positions)
    x_positions = np.interp(timestamps, timestamps[valid_indices], x_positions[valid_indices])
    y_positions = np.interp(timestamps, timestamps[valid_indices], y_positions[valid_indices])
    
    velocities = []
    for i in range(1, len(timestamps)):
        dx = x_positions[i] - x_positions[i - 1]
        dy = y_positions[i] - y_positions[i - 1]
        dt = timestamps[i] - timestamps[i - 1]
        
        # Velocity is calculated as the Euclidean distance over time
        velocity = np.sqrt(dx**2 + dy**2) / dt
        velocities.append(velocity)
    
    # velocities *= 1e3  # degree per second
    velocities = np.array(velocities)
    
    # velocities = np.convolve(velocities, np.ones(5)/5, mode='same')
    # velocities[:2] = velocities[2]
    # velocities[-2:] = velocities[-3]
    
    vx = np.diff(x_positions) / np.diff(timestamps)
    vy = np.diff(y_positions) / np.diff(timestamps)
    
    # Median-based standard deviation for x and y
    sigma_vx = np.sqrt(np.median((vx - np.median(vx))**2))
    sigma_vy = np.sqrt(np.median((vy - np.median(vy))**2))

    return velocities, sigma_vx, sigma_vy

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
