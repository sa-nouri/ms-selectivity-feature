import numpy as np
import copy
from typing import Dict, Any, Optional, TypedDict

from .utils import compute_velocity


class BlinkDetectorParams(TypedDict):
    """Parameters for blink detection.
    
    Attributes:
        VERBOSE: Whether to print debug information.
        MAXIMAL_DISTANCE_TO_SACCADE_MILLISEC: Maximum time window to look for
            nearby saccades in milliseconds.
        MINIMAL_BLINK_DURATION_MILLISEC: Minimum duration for a valid blink
            in milliseconds.
    """
    VERBOSE: bool
    MAXIMAL_DISTANCE_TO_SACCADE_MILLISEC: float
    MINIMAL_BLINK_DURATION_MILLISEC: float


class GazePoint(TypedDict):
    """Structure for a single gaze point.
    
    Attributes:
        time: Timestamp of the gaze point.
        status: Confidence status of the gaze point.
        EYE_MOVEMENT_TYPE: Type of eye movement at this point.
        SACC_INTERVAL_INDEX: Index of the saccade interval.
        INTERSACC_INTERVAL_INDEX: Index of the intersaccade interval.
    """
    time: float
    status: int
    EYE_MOVEMENT_TYPE: str
    SACC_INTERVAL_INDEX: int
    INTERSACC_INTERVAL_INDEX: int


class GazePoints(TypedDict):
    """Structure for gaze points data.
    
    Attributes:
        data: Array of GazePoint objects.
    """
    data: np.ndarray


def BlinkDetectorByEyePositions(
    param: BlinkDetectorParams,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    gaze_points: GazePoints,
    inplace: bool = False
) -> GazePoints:
    """Detect blinks in eye movement data based on eye position and gaze points.
    
    This function detects blinks in eye movement data by analyzing eye positions
    and gaze points with confidence intervals. It extends the 0-confidence
    intervals by adding nearby saccades if they are within a defined distance.
    
    Args:
        param: Dictionary containing detection parameters:
            - VERBOSE: Whether to print debug information
            - MAXIMAL_DISTANCE_TO_SACCADE_MILLISEC: Maximum time window to look
                for nearby saccades
            - MINIMAL_BLINK_DURATION_MILLISEC: Minimum duration for a valid blink
        x_positions: Array of X positions of eye movements.
        y_positions: Array of Y positions of eye movements.
        gaze_points: Dictionary containing gaze recording data with fields:
            - data: Array of gaze points with fields:
                - time: Timestamps
                - status: Confidence status
                - EYE_MOVEMENT_TYPE: Type of eye movement
                - SACC_INTERVAL_INDEX: Saccade interval index
                - INTERSACC_INTERVAL_INDEX: Intersaccade interval index
        inplace: Whether to modify gaze_points in place or create a copy.
            Defaults to False.
    
    Returns:
        Updated gaze_points dictionary with 'BLINK' labels added to the
        appropriate intervals.
    """
    if not inplace:
        gaze_points = copy.deepcopy(gaze_points)

    if 'status' not in gaze_points['data'].dtype.names:
        return gaze_points

    is_blink = (gaze_points['data']['status'] == 0).astype(int)
    
    blink_diff = np.diff(np.hstack([[0], is_blink]))
    blink_onsets = np.nonzero(blink_diff == 1)[0]
    blink_offsets = np.nonzero(np.diff(np.hstack([is_blink, [0]])) == -1)[0]

    times = gaze_points['data']['time']
    
    assert len(blink_onsets) == len(blink_offsets)
    
    for onset, offset in zip(blink_onsets, blink_offsets):
        if param["VERBOSE"]:
            print(f"Found blink from {times[onset]} to {times[offset]}")

        onset_candidate = onset
        while onset_candidate >= 0 and times[onset] - times[onset_candidate] < param["MAXIMAL_DISTANCE_TO_SACCADE_MILLISEC"]:
            if gaze_points['data'][onset_candidate]['EYE_MOVEMENT_TYPE'] == 'SACCADE':
                sacc_index = gaze_points['data'][onset_candidate]['SACC_INTERVAL_INDEX']
                first_saccade_index = np.nonzero(gaze_points['data']['SACC_INTERVAL_INDEX'] == sacc_index)[0][0]
                onset = first_saccade_index
                break
            onset_candidate -= 1

        offset_candidate = offset
        while offset_candidate < len(times) and times[offset_candidate] - times[offset] < param["MAXIMAL_DISTANCE_TO_SACCADE_MILLISEC"]:
            if gaze_points['data'][offset_candidate]['EYE_MOVEMENT_TYPE'] == 'SACCADE':
                sacc_index = gaze_points['data'][offset_candidate]['SACC_INTERVAL_INDEX']
                last_saccade_index = np.nonzero(gaze_points['data']['SACC_INTERVAL_INDEX'] == sacc_index)[0][-1]
                offset = last_saccade_index
                break
            offset_candidate += 1

        if param["VERBOSE"]:
            print(f"Extended it to {times[onset]} {times[offset]}")

        if times[offset] - times[onset] < param['MINIMAL_BLINK_DURATION_MILLISEC']:
            gaze_points['data'][onset:offset + 1]['EYE_MOVEMENT_TYPE'] = 'NOISE'
        else:
            gaze_points['data'][onset:offset + 1]['EYE_MOVEMENT_TYPE'] = 'BLINK'
            gaze_points['data'][onset:offset + 1]['SACC_INTERVAL_INDEX'] = -1
            gaze_points['data'][onset:offset + 1]['INTERSACC_INTERVAL_INDEX'] = -1

    return gaze_points
