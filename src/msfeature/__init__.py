"""msfeature — microsaccade selectivity feature extraction and decoding.

Implements the Engbert & Kliegl (2003) microsaccade detection algorithm
with the monocular / 1° amplitude cap / 30 ms refractory merge / 2.5 SD
amplitude-outlier rejection variants used in Nouri et al. 2025 (iScience).

References:
    Engbert R, Kliegl R. Microsaccades uncover the orientation of covert
    attention. Vision Research, 43(9):1035-1045, 2003.

    Nouri S, et al. Microsaccade selectivity as discriminative feature for
    object decoding. iScience, 28(1):111584, 2025.
"""

from .config import DatasetConfig, MONKEY_CONFIG, HUMAN_CONFIG
from .detect import detect_microsaccades
from .events import Microsaccade

__all__ = [
    "DatasetConfig",
    "MONKEY_CONFIG",
    "HUMAN_CONFIG",
    "Microsaccade",
    "detect_microsaccades",
]
