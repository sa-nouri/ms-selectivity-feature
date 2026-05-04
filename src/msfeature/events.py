"""Microsaccade event representation and post-detection rules.

The post-detection rules implement the variants Nouri et al. 2025 apply on
top of canonical Engbert-Kliegl detection:

1. Refractory merge: events whose onset falls within `refractory_merge_s`
   of the previous event's offset are merged into the previous event.
   This collapses the "corrective" follow-up movements that frequently
   accompany a microsaccade.
2. Amplitude cap: events with amplitude > `max_amplitude_deg` are
   classified as macro-saccades and dropped.
3. Outlier rejection: events whose amplitude is more than
   `amplitude_outlier_sd` standard deviations above the mean amplitude
   of the surviving events are dropped (population-level cleanup).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Microsaccade:
    """One detected microsaccade event.

    All times are in seconds, amplitudes in the same unit as the input
    positions (degrees of visual angle for our datasets), and direction
    in radians in [-pi, pi] from arctan2(dy, dx).
    """

    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    amplitude: float
    peak_velocity: float
    direction: float

    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx + 1


def merge_refractory(
    events: list[Microsaccade],
    refractory_samples: int,
    dt: float,
    max_merged_duration_s: float = 0.050,
) -> list[Microsaccade]:
    """Merge events separated by < `refractory_samples` of intervening fixation.

    Implements the "merge corrective movement" rule from Nouri et al. 2025.
    The merge is only applied when the resulting event stays under
    `max_merged_duration_s` total — without this cap, a chain of many
    sub-threshold events spaced < refractory apart cascade-merges into
    a single multi-hundred-ms super-event, which is not what the rule is
    meant to model. 50 ms is a generous upper bound for one microsaccade
    plus one or two post-saccadic oscillations.

    The merged event's amplitude/direction are placeholders here (set to
    NaN) and are recomputed by `recompute_amplitude_direction` against
    the underlying position trace. Peak velocity is the max of the
    component peaks.
    """
    if not events:
        return []
    max_merged_samples = int(round(max_merged_duration_s / dt))
    merged: list[Microsaccade] = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        gap = ev.start_idx - prev.end_idx - 1
        merged_span_samples = ev.end_idx - prev.start_idx
        if gap < refractory_samples and merged_span_samples <= max_merged_samples:
            merged[-1] = Microsaccade(
                start_idx=prev.start_idx,
                end_idx=ev.end_idx,
                start_time=prev.start_time,
                end_time=ev.end_time,
                duration=merged_span_samples * dt,
                amplitude=float("nan"),
                peak_velocity=max(prev.peak_velocity, ev.peak_velocity),
                direction=float("nan"),
            )
        else:
            merged.append(ev)
    return merged


def recompute_amplitude_direction(
    events: list[Microsaccade], x: np.ndarray, y: np.ndarray
) -> list[Microsaccade]:
    """Recompute `amplitude` and `direction` from the underlying position trace.

    Used after refractory merging, when the start/end indices have changed.
    """
    out: list[Microsaccade] = []
    for ev in events:
        dx = float(x[ev.end_idx] - x[ev.start_idx])
        dy = float(y[ev.end_idx] - y[ev.start_idx])
        out.append(
            Microsaccade(
                start_idx=ev.start_idx,
                end_idx=ev.end_idx,
                start_time=ev.start_time,
                end_time=ev.end_time,
                duration=ev.duration,
                amplitude=float(np.hypot(dx, dy)),
                peak_velocity=ev.peak_velocity,
                direction=float(np.arctan2(dy, dx)),
            )
        )
    return out


def cap_amplitude(
    events: list[Microsaccade], max_amplitude: float
) -> list[Microsaccade]:
    """Drop events with amplitude > `max_amplitude`."""
    return [ev for ev in events if ev.amplitude <= max_amplitude]


def cap_duration(
    events: list[Microsaccade], max_duration_s: float
) -> list[Microsaccade]:
    """Drop events with duration > `max_duration_s`.

    Used as a sanity filter against EK threshold-collapse on too-clean
    fixation data — without this, a sub-threshold-noise stretch can
    light up the entire trial as one "event" several hundred ms long.
    """
    return [ev for ev in events if ev.duration <= max_duration_s]


def reject_amplitude_outliers(
    events: list[Microsaccade], n_sd: float
) -> list[Microsaccade]:
    """Drop events with amplitude > mean + n_sd * SD over the surviving set.

    Two-sided trimming was not applied in the paper (small microsaccades
    are still real microsaccades), so we only trim from above.
    """
    if len(events) < 2 or not np.isfinite(n_sd):
        return events
    amps = np.array([ev.amplitude for ev in events], dtype=float)
    mu = float(np.mean(amps))
    sd = float(np.std(amps, ddof=0))
    if sd == 0.0:
        return events
    cutoff = mu + n_sd * sd
    return [ev for ev in events if ev.amplitude <= cutoff]
