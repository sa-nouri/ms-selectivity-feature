"""Eye movement detection package.

This package provides tools for detecting and analyzing various types of eye movements,
including saccades, microsaccades, blinks, and glitches.
"""

from .blink_detector import detect_blinks, validate_blinks
from .detect_microsaccades import detect_microsaccades, validate_microsaccades
from .detect_saccades import detect_saccades, validate_saccades
from .glitch_detector import detect_glitches, validate_glitches
from .postprocess import postprocess
from .preprocess import preprocess

__all__ = [
    "detect_blinks",
    "validate_blinks",
    "detect_microsaccades",
    "validate_microsaccades",
    "detect_saccades",
    "validate_saccades",
    "detect_glitches",
    "validate_glitches",
    "postprocess",
    "preprocess",
]
