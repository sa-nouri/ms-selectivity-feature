import numpy as np
from sklearn import mixture


def compute_velocity_consecutive(x_position, y_position, timestamp):
    x_velocity = np.append(0, np.abs(np.diff(x_position)/np.diff(timestamp)))
    y_velocity = np.append(0, np.abs(np.diff(y_position)/np.diff(timestamp)))
    
    Pythagorean_velocity = np.sqrt((np.square(x_velocity)) + (np.square(y_velocity)))
    return x_velocity, y_velocity, Pythagorean_velocity


