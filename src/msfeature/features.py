"""Per-trial features from microsaccade event lists.

The primary feature used in Nouri et al. 2025 is the microsaccade rate
binned at 15 ms within a 100-600 ms post-stimulus window. Secondary
descriptors (peak velocity, amplitude, duration, direction) are reported
per-event but can be aggregated per-trial for descriptive statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .events import Microsaccade


def shift_to_onset(
    events: list[Microsaccade], onset_sample: int, dt: float
) -> list[Microsaccade]:
    """Re-index events so that t=0 is the stimulus-onset sample.

    This is a pure re-coordinate operation — `start_idx` and `end_idx`
    become relative to onset (negative for pre-stim) and `start_time` /
    `end_time` are recomputed in seconds.
    """
    onset_s = onset_sample * dt
    out: list[Microsaccade] = []
    for ev in events:
        out.append(
            replace(
                ev,
                start_idx=ev.start_idx - onset_sample,
                end_idx=ev.end_idx - onset_sample,
                start_time=ev.start_time - onset_s,
                end_time=ev.end_time - onset_s,
            )
        )
    return out


@dataclass(frozen=True)
class RateBinSpec:
    """Defines the binned-rate feature window.

    Times are in seconds, relative to the trial-start sample (index 0).
    For analyses that want stimulus-locked rates, shift the trial array
    upstream so index 0 is stimulus onset.

    If `(window_end_s - window_start_s)` is not a whole multiple of
    `bin_width_s`, the spec is rounded down to the largest whole number
    of bins; the effective window therefore ends at
    `window_start_s + n_bins * bin_width_s`, possibly < window_end_s.
    """

    window_start_s: float
    window_end_s: float
    bin_width_s: float

    def __post_init__(self) -> None:
        if self.bin_width_s <= 0:
            raise ValueError("bin_width_s must be positive")
        if self.window_end_s <= self.window_start_s:
            raise ValueError("window_end_s must be > window_start_s")

    def bin_edges(self) -> np.ndarray:
        # Add a small tolerance before flooring so that exact-integer
        # ratios (e.g. 30.0 represented as 29.999999...) don't get
        # truncated to 29 by floating-point error.
        ratio = (self.window_end_s - self.window_start_s) / self.bin_width_s
        n_bins = max(1, int(np.floor(ratio + 1e-9)))
        return self.window_start_s + np.arange(n_bins + 1) * self.bin_width_s

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges()) - 1

    @classmethod
    def from_trial_length(
        cls,
        n_samples: int,
        sampling_rate_hz: float,
        bin_width_s: float = 0.015,
        window_start_s: float = 0.0,
    ) -> "RateBinSpec":
        """Construct a window covering the full trial after `window_start_s`.

        Useful when the data is already pre-trimmed to a post-stimulus
        window and the paper's 100-600 ms range doesn't fit. The end is
        clamped down to a whole number of `bin_width_s` bins.
        """
        trial_end_s = n_samples / sampling_rate_hz
        usable_s = trial_end_s - window_start_s
        n_bins = max(1, int(np.floor(usable_s / bin_width_s)))
        window_end_s = window_start_s + n_bins * bin_width_s
        return cls(window_start_s, window_end_s, bin_width_s)


# Per-paper window: 100-600 ms post-stimulus, 15 ms bins. Only valid when
# the input trial has at least 600 ms of post-onset data.
NOURI_RATE_BINS = RateBinSpec(
    window_start_s=0.100,
    window_end_s=0.600,
    bin_width_s=0.015,
)


def binned_rate(
    events: list[Microsaccade], spec: RateBinSpec = NOURI_RATE_BINS
) -> np.ndarray:
    """Microsaccade rate (events / s) per bin.

    Each event contributes 1 count to the bin its `start_time` falls in.
    The count in each bin is divided by the bin width to give a rate in
    Hz, matching how the paper reports "microsaccade rate".
    """
    edges = spec.bin_edges()
    times = np.array([ev.start_time for ev in events], dtype=float)
    counts, _ = np.histogram(times, bins=edges)
    return counts / spec.bin_width_s


def trial_descriptors(events: list[Microsaccade]) -> dict[str, float]:
    """Summary statistics for one trial. NaN if no events."""
    if not events:
        return {
            "n_events": 0,
            "mean_amplitude": float("nan"),
            "mean_duration": float("nan"),
            "mean_peak_velocity": float("nan"),
            "mean_direction_x": float("nan"),
            "mean_direction_y": float("nan"),
        }
    amps = np.array([ev.amplitude for ev in events])
    durs = np.array([ev.duration for ev in events])
    pkvs = np.array([ev.peak_velocity for ev in events])
    dirs = np.array([ev.direction for ev in events])
    return {
        "n_events": len(events),
        "mean_amplitude": float(np.mean(amps)),
        "mean_duration": float(np.mean(durs)),
        "mean_peak_velocity": float(np.mean(pkvs)),
        # Mean direction via unit-vector averaging (the only well-defined
        # circular mean — arithmetic mean of angles is meaningless).
        "mean_direction_x": float(np.mean(np.cos(dirs))),
        "mean_direction_y": float(np.mean(np.sin(dirs))),
    }


def stack_trial_features(
    per_trial_events: list[list[Microsaccade]],
    spec: RateBinSpec = NOURI_RATE_BINS,
) -> np.ndarray:
    """Stack binned-rate vectors across trials into an (n_trials, n_bins) matrix.

    This is the primary feature matrix passed to the SVM in Nouri et al.
    Returns an empty (0, n_bins) array when given no trials.
    """
    if not per_trial_events:
        return np.zeros((0, spec.n_bins), dtype=float)
    return np.vstack([binned_rate(evs, spec) for evs in per_trial_events])
