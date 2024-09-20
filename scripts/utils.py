import numpy as np
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

