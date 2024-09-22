# Eye Movement Analysis Scripts

This folder contains Python scripts for analyzing eye movement data, specifically for detecting saccades/microsaccades and computing their characteristics.

## Contents

- [`preprocess.py`](./preprocess.py): Python functions for applying preprocessing to eye movement data.
- [`blink_detector.py`](./blink_detector.py): Python function for detecting blinks in eye movement data based on eye position data
- [`glitch_detector.py`](./glitch_detector.py): Python function for detecting glitches in eye-tracking.
- [`uitls.py`](./utils.py): Python scripts for applying and computing velocity and amplitude.
- [`detect_saccades.py`](./detect_saccades.py): Python script for detecting saccades in eye movement data.
- [`detect_microsaccades.py`](./detect_microsaccades.py): Python script for detecting microsaccades in eye movement data.

## Requirements

- Ensure that you have Python installed on your system (Python +3.10).
- Make sure that you have installed the required python packages (Described in [Requirements](./../requirements.txt))
        - `pip install -r requirements.txt`
- Run the scripts using Python. For example:
        - `python detect_microsaccades.py`

## Notes

- Modify the scripts as needed to suit your specific requirements.
- Ensure that your eye movement data is properly formatted before running the scripts.
- Refer to the documentation within each script for detailed usage instructions and function descriptions.
