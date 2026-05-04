"""Loading the .mat files shipped with the repo.

The monkey sample is a Matlab v5 file containing a single struct `sess`
with fields:

    EyeX                   (n_trials, n_samples)  float64, degrees of arc
    EyeY                   (n_trials, n_samples)  float64, degrees of arc
    trial_isface           (n_trials, 1)          uint8,  0/1 face label
    trial_stimulus_code    (n_trials, 1)          uint8,  1..155 stimulus id

There is no timestamp channel and no stimulus-onset marker; the sampling
rate must be supplied by the caller (see `msfeature.config`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class EyeDataset:
    """In-memory view of one .mat session."""

    eye_x: np.ndarray  # (n_trials, n_samples)
    eye_y: np.ndarray  # (n_trials, n_samples)
    is_face: np.ndarray  # (n_trials,) bool
    stimulus_code: np.ndarray  # (n_trials,) int

    @property
    def n_trials(self) -> int:
        return self.eye_x.shape[0]

    @property
    def n_samples(self) -> int:
        return self.eye_x.shape[1]


_REQUIRED_FIELDS = ("EyeX", "EyeY", "trial_isface", "trial_stimulus_code")


def load_session(path: str | Path) -> EyeDataset:
    """Load one monkey-format .mat session into an `EyeDataset`.

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if the file does not contain a `sess` struct.
        AttributeError: if `sess` is missing one of the required fields.
        ValueError: if the loaded shapes are inconsistent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    mat = loadmat(str(p), squeeze_me=False, struct_as_record=False)
    if "sess" not in mat:
        raise KeyError(f"{p}: no 'sess' top-level variable in .mat file")
    sess = mat["sess"][0, 0]
    missing = [f for f in _REQUIRED_FIELDS if not hasattr(sess, f)]
    if missing:
        raise AttributeError(
            f"{p}: 'sess' is missing required fields: {missing}; "
            f"have {sess._fieldnames}"
        )

    eye_x = np.asarray(sess.EyeX, dtype=float)
    eye_y = np.asarray(sess.EyeY, dtype=float)
    is_face = np.asarray(sess.trial_isface, dtype=int).ravel().astype(bool)
    stim = np.asarray(sess.trial_stimulus_code, dtype=int).ravel()

    if eye_x.ndim != 2 or eye_y.ndim != 2:
        raise ValueError(
            f"{p}: EyeX/EyeY must be 2-D (trials x samples); "
            f"got shapes {eye_x.shape}, {eye_y.shape}"
        )
    if eye_x.shape != eye_y.shape:
        raise ValueError(
            f"{p}: EyeX shape {eye_x.shape} != EyeY shape {eye_y.shape}"
        )
    n_trials = eye_x.shape[0]
    if is_face.size != n_trials or stim.size != n_trials:
        raise ValueError(
            f"{p}: trial-count mismatch — EyeX has {n_trials} trials, "
            f"trial_isface has {is_face.size}, trial_stimulus_code has {stim.size}"
        )

    return EyeDataset(
        eye_x=eye_x, eye_y=eye_y, is_face=is_face, stimulus_code=stim
    )
