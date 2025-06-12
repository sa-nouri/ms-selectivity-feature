from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
from sklearn import mixture


class PostprocessParams(TypedDict):
    """Parameters for postprocessing eye movement data.

    Attributes:
        min_interval: Minimum interval between events in seconds.
        max_interval: Maximum interval between events in seconds.
    """

    min_interval: float
    max_interval: float


def noise_threshold_extract(
    data_input: np.ndarray,
    preliminary_threshold: float = 10.0,
    n_components: int = 2,
    std_multiplier: float = 3.0,
) -> float:
    """Extract noise threshold using Gaussian Mixture Model.

    This function uses a Gaussian Mixture Model to separate signal from noise
    in eye movement data. It fits a GMM to the data below a preliminary threshold
    and returns a refined threshold value.

    Args:
        data_input: Array of eye movement data (e.g., velocities).
        preliminary_threshold: Initial threshold to filter data for GMM fitting.
            Defaults to 10.0.
        n_components: Number of components in the GMM. Defaults to 2.
        std_multiplier: Multiplier for standard deviation to set final threshold.
            Defaults to 3.0.

    Returns:
        A float value representing the noise threshold, calculated as the mean
        of the lower component plus std_multiplier times its standard deviation.
    """
    # Filter data below preliminary threshold
    idx = data_input < preliminary_threshold
    if not np.any(idx):
        return preliminary_threshold

    # Fit GMM to filtered data
    gmm = mixture.GaussianMixture(
        n_components=n_components, covariance_type="full", max_iter=100
    ).fit(data_input[idx].reshape(-1, 1))

    # Get means and standard deviations
    means = gmm.means_
    covariances = np.sqrt(gmm.covariances_)

    # Sort components by mean
    idd = np.argsort(means.reshape(n_components))

    # Calculate threshold
    value = float((means[idd[0]] + std_multiplier * covariances[idd[0]])[0])
    return value


def validate_saccades_min_duration(
    saccades: List[Tuple[int, int]],
    timestamps: np.ndarray,
    min_duration: float,
    verbose: bool = False,
) -> List[Tuple[int, int]]:
    """Validate the detected saccades based on their duration.

    This function filters out saccades that are shorter than the minimum duration
    threshold, as they are likely to be noise or artifacts.

    Args:
        saccades: List of detected saccades, where each saccade is represented
            as a tuple (start_index, end_index).
        timestamps: Array of timestamps corresponding to the eye movement data.
        min_duration: The minimal duration (in seconds or milliseconds) required
            for a valid saccade.
        verbose: Whether to print details of discarded saccades.
            Defaults to False.

    Returns:
        List of valid saccades (those that pass the duration filter), where each
        saccade is represented as a tuple (start_index, end_index).
    """
    valid_saccades = []

    for start_index, end_index in saccades:
        duration = timestamps[end_index] - timestamps[start_index]

        if duration >= min_duration:
            valid_saccades.append((start_index, end_index))
        elif verbose:
            print(
                f"Saccade from {timestamps[start_index]} to {timestamps[end_index]} discarded (duration: {duration} < {min_duration})"
            )

    return valid_saccades


def postprocess_data(
    events: list[dict],
    timestamps: np.ndarray,
    params: PostprocessParams,
) -> list[dict]:
    """Postprocess detected eye movement events by merging and filtering.

    Args:
        events: List of dictionaries containing information about detected events.
        timestamps: Array of timestamps corresponding to the eye movement data.
        params: Dictionary containing postprocessing parameters:
            - min_interval: Minimum interval between events in seconds.
            - max_interval: Maximum interval between events in seconds.

    Returns:
        List of dictionaries containing information about postprocessed events.
    """
    if not events:
        return []

    # Sort events by start time
    sorted_events = sorted(events, key=lambda x: x["start_time"])

    # Merge events that are too close together
    merged_events = []
    current_event = sorted_events[0]

    for next_event in sorted_events[1:]:
        interval = next_event["start_time"] - current_event["end_time"]
        if interval < params["min_interval"]:
            # Merge events
            current_event["end_time"] = next_event["end_time"]
            current_event["duration"] = (
                current_event["end_time"] - current_event["start_time"]
            )
            if "amplitude" in current_event and "amplitude" in next_event:
                current_event["amplitude"] = max(
                    current_event["amplitude"], next_event["amplitude"]
                )
        else:
            # Check if the current event is valid
            if (
                current_event["duration"] >= params["min_interval"]
                and current_event["duration"] <= params["max_interval"]
            ):
                merged_events.append(current_event)
            current_event = next_event

    # Add the last event if it's valid
    if (
        current_event["duration"] >= params["min_interval"]
        and current_event["duration"] <= params["max_interval"]
    ):
        merged_events.append(current_event)

    return merged_events


def postprocess(
    events: list[dict],
    timestamps: np.ndarray,
    min_interval: float = 0.02,
    max_interval: float = 0.4,
) -> list[dict]:
    """Postprocess detected eye movement events with default parameters.

    Args:
        events: List of dictionaries containing information about detected events.
        timestamps: Array of timestamps corresponding to the eye movement data.
        min_interval: Minimum interval between events in seconds.
        max_interval: Maximum interval between events in seconds.

    Returns:
        List of dictionaries containing information about postprocessed events.
    """
    params = {
        "min_interval": min_interval,
        "max_interval": max_interval,
    }
    return postprocess_data(events, timestamps, params)
