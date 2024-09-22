import numpy as np
import copy

def BlinkDetectorByEyePositions(param, x_positions, y_positions, gaze_points, inplace=False):
    """
    Detect blinks in eye movement data based on eye position data (x, y) and gaze points with confidence intervals.
    We extend the 0-confidence intervals by adding the nearest saccade if within a defined distance.

    Args:
    - param (dict): Parameters such as maximal distance to saccade, minimal blink duration, etc.
    - x_positions (array): X positions of eye movements.
    - y_positions (array): Y positions of eye movements.
    - gaze_points (dict): Gaze recording data (includes 'data', 'time', 'status', etc.).
    - inplace (bool): Whether to modify gaze_points in place or create a copy.

    Returns:
    - gaze_points (dict): Updated gaze points with 'BLINK' labels added.
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
