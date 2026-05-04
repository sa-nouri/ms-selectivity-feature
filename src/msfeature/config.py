"""Per-dataset configuration.

The .mat files in this repo carry no timestamps, no stimulus-onset markers,
and no sampling-rate metadata. All of that has to be supplied externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SigmaMethod = Literal["engbert2003", "engbert2015"]


@dataclass(frozen=True)
class DatasetConfig:
    """Sampling and detection parameters for one dataset.

    Attributes:
        name: human-readable label (used in plots / logs).
        sampling_rate_hz: sampling rate of the eye trace in Hz.
        velocity_threshold_lambda: multiplier on per-axis sigma in the EK
            elliptic threshold. Engbert 2003 default = 6.
        min_duration_ms: minimum saccade duration in milliseconds. The
            paper text recommends 12 ms; the Engbert R toolbox defaults
            to ~6 ms; pymovements is sample-count rather than time. We
            default to 6 ms to match the toolbox.
        max_duration_ms: events longer than this are discarded as
            artifacts. Real microsaccades are 6-30 ms; macro-saccades
            up to ~100 ms; anything longer is almost certainly a
            threshold-collapse artifact (very clean data + low sigma
            lets a continuous drift stretch cross threshold). Default
            100 ms.
        max_amplitude_deg: events with amplitude above this are dropped
            (classified as macro-saccades). Nouri 2025 = 1.0 deg.
        refractory_merge_ms: merge events whose onset falls within this
            many ms of the previous event's offset. Nouri 2025 = 30 ms.
        amplitude_outlier_sd: drop surviving events whose amplitude is
            more than this many SDs above the mean amplitude (one-sided).
            Nouri 2025 = 2.5.
        sigma_method: which form of the EK sigma estimate to use.
            'engbert2003' = sqrt(median(v^2) - median(v)^2) -- verbatim
            EK 2003 paper text. 'engbert2015' = sqrt(median((v-median(v))^2))
            -- the form used by the R Microsaccade Toolbox 0.9 and by
            pymovements (default).
        sigma_fallback_to_std: if True, fall back to mean-based SD when
            the median-based sigma underflows. Disabled by default to
            avoid silent threshold changes; the canonical R toolbox
            raises in this case.
        stim_onset_sample: index of the stimulus-onset sample within
            each trial. None = no stimulus alignment (use sample 0 as
            time zero). For the monkey sample shipped here, the layout
            is 300 ms baseline + post-stim, sampled at 1 kHz, so
            stimulus onset is at sample 300.
        lowpass_cutoff_hz: optional pre-detection low-pass cutoff. None
            (the default) skips filtering -- the EK 5-point velocity
            kernel is itself a smoother and double-smoothing biases
            the threshold upward. Set this only if you have specific
            high-frequency tracker noise to remove.
    """

    name: str
    sampling_rate_hz: float
    velocity_threshold_lambda: float = 6.0
    min_duration_ms: float = 6.0
    max_duration_ms: float = 100.0
    max_amplitude_deg: float = 1.0
    refractory_merge_ms: float = 30.0
    amplitude_outlier_sd: float = 2.5
    sigma_method: SigmaMethod = "engbert2015"
    sigma_fallback_to_std: bool = False
    stim_onset_sample: int | None = None
    lowpass_cutoff_hz: float | None = None

    @property
    def dt(self) -> float:
        return 1.0 / self.sampling_rate_hz

    @property
    def min_duration_samples(self) -> int:
        return max(1, int(round(self.min_duration_ms * 1e-3 * self.sampling_rate_hz)))

    @property
    def refractory_merge_samples(self) -> int:
        return max(0, int(round(self.refractory_merge_ms * 1e-3 * self.sampling_rate_hz)))

MONKEY_CONFIG = DatasetConfig(
    name="monkey",
    sampling_rate_hz=1000.0,
    velocity_threshold_lambda=6.0,
    min_duration_ms=6.0,
    max_amplitude_deg=1.0,
    refractory_merge_ms=30.0,
    amplitude_outlier_sd=2.5,
    sigma_method="engbert2015",
    stim_onset_sample=250,
    lowpass_cutoff_hz=None,
)


HUMAN_CONFIG = DatasetConfig(
    name="human",
    sampling_rate_hz=500.0,
    velocity_threshold_lambda=6.0,
    min_duration_ms=12.0,
    max_amplitude_deg=1.0,
    refractory_merge_ms=30.0,
    amplitude_outlier_sd=2.5,
    sigma_method="engbert2015",
    stim_onset_sample=None,
    lowpass_cutoff_hz=None,
)
