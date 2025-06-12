"""Microsaccade Selectivity as Discriminative Feature for Object Decoding.

This package provides tools for analyzing microsaccade selectivity in eye movement data
and using it as a discriminative feature for object decoding.
"""

from .blink_detector import BlinkDetectorByEyePositions
from .detect_microsaccades import detect_microsaccades
from .detect_saccades import detect_saccades
from .glitch_detector import detect_glitches
from .postprocess import noise_threshold_extract, validate_saccades_min_duration
from .preprocess import (
    filter_data,
    remove_blinks,
    correct_baseline_drift,
    interpolate_data,
    low_pass_filter_eye_positions,
)
from .utils import (
    compute_velocity,
    compute_amplitude,
    compute_partial_velocity,
    compute_velocity_consecutive,
)
from .logger import logger, setup_logger

__version__ = "0.1.0"
__author__ = "Salar Nouri"
__email__ = "salar.nouri@epfl.ch"

__all__ = [
    "BlinkDetectorByEyePositions",
    "detect_microsaccades",
    "detect_saccades",
    "detect_glitches",
    "noise_threshold_extract",
    "validate_saccades_min_duration",
    "filter_data",
    "remove_blinks",
    "correct_baseline_drift",
    "interpolate_data",
    "low_pass_filter_eye_positions",
    "compute_velocity",
    "compute_amplitude",
    "compute_partial_velocity",
    "compute_velocity_consecutive",
    "logger",
    "setup_logger",
]
