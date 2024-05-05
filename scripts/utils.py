import numpy as np
from sklearn import mixture


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
