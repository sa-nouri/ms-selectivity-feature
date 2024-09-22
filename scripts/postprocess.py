import numpy as np
from sklearn import mixture


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

def validate_saccades_min_duration(saccades, timestamps, min_duration, verbose=False):
    """
    Validate the detected saccades based on their duration.
    If a saccade is shorter than min_duration, it is considered noise and discarded.

    Args:
    - saccades (list of tuples): List of detected saccades, where each saccade is represented as a tuple (start_index, end_index).
    - timestamps (array): Array of timestamps corresponding to the eye movement data.
    - min_duration (float): The minimal duration (in seconds or milliseconds) required for a valid saccade.
    - verbose (bool): Whether to print details of discarded saccades (default is False).

    Returns:
    - valid_saccades (list of tuples): List of valid saccades (those that pass the duration filter).
    """
    valid_saccades = []
    
    for start_index, end_index in saccades:
        duration = timestamps[end_index] - timestamps[start_index]
        
        if duration >= min_duration:
            valid_saccades.append((start_index, end_index))
        elif verbose:
            print(f"Saccade from {timestamps[start_index]} to {timestamps[end_index]} discarded (duration: {duration} < {min_duration})")
    
    return valid_saccades
