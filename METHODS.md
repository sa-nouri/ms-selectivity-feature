# Methods

This document describes the algorithm implemented in `msfeature` and the deviations from / faithful reproductions of Engbert & Kliegl 2003 and Nouri et al. 2025.

## 1. Pipeline overview

```
raw eye position (x, y)
   │
   ▼
preprocess_trial()
   ├─ mark_blinks         flag NaN / pupil-loss samples
   ├─ remove_glitches     interpolate over short tracker spikes
   ├─ correct_drift       subtract linear-in-time trend (per axis)
   └─ lowpass             Butterworth zero-phase filter
   │
   ▼
detect_microsaccades()
   ├─ ek_velocity_2d           5-point EK velocity (Eq. 1, EK 2003)
   ├─ compute_sigma            median-based per-axis scale (Eq. 2, EK 2003)
   ├─ elliptic_threshold_mask  (vx/λσx)² + (vy/λσy)² > 1
   ├─ extract_runs             group consecutive supra-threshold samples
   └─ post-rules               refractory merge → amp cap → outlier reject
   │
   ▼
list[Microsaccade]
   │
   ▼
features.binned_rate()    15 ms-binned MS rate over a configurable window
   │
   ▼
decode.cross_validate()   Stratified k-fold linear SVM
```

---

## 2. Detector details

### 2.1 Velocity (EK 2003 Eq. 1)

For each axis independently:

```
v[n] = (x[n+2] + x[n+1] - x[n-1] - x[n-2]) / (6 · Δt)
```

This is a 5-sample symmetric kernel that suppresses single-sample noise while preserving the time resolution needed to detect 12 ms events. Edge samples (the first two and last two) are returned as `NaN` so that downstream operations never confuse "no defined velocity" with "zero velocity".

### 2.2 Sigma (EK 2003 Eq. 2)

Per axis:

```
σ = sqrt( median(v²) − median(v)² )
```

This is a robust scale estimate: outliers (the saccades themselves) do not inflate σ the way a sample standard deviation does. When the expression underflows (pristine fixation ⇒ med(v²) ≈ 0 ≈ med(v)²) we fall back to mean-based SD, matching Engbert's R toolbox and `saccadr`.

### 2.3 Threshold (elliptic)

```
( vx / (λ σx) )² + ( vy / (λ σy) )² > 1
```

Because the σ values are estimated independently per axis, this is an **ellipse** centred at the origin in velocity space, not a circle. Anisotropic tracker noise (common in head-fixed monkey work, where horizontal noise tends to differ from vertical) is therefore handled correctly. The previous library used `λ · sqrt(σx² + σy²)` against `sqrt(vx² + vy²)`, which is a circle of average radius — a known mis-application of the EK rule.

### 2.4 Run-length grouping

Consecutive samples where the elliptic test fires are grouped into one event. Runs shorter than `min_duration_samples` are dropped. The default of 6 ms (12 samples at 2 kHz, 6 samples at 500 Hz) matches the EK 2003 minimum-duration criterion.

### 2.5 Post-detection rules (Nouri-specific)

After the EK detector emits raw events, three additional rules from Nouri et al. 2025 are applied:

1. **Refractory merge** — events whose onset falls within `refractory_merge_s` (30 ms) of the previous event's offset are merged. This collapses the corrective follow-up movements that commonly accompany a microsaccade.
2. **Amplitude cap** — events with start-to-end amplitude > 1.0° are classified as macro-saccades and dropped. Only sub-1° events are retained as "microsaccades".
3. **Amplitude outlier rejection** — within the surviving event set, events with amplitude > mean + 2.5 SD are dropped. One-sided only; small microsaccades remain real microsaccades.

The post-rules can be skipped with `apply_post_rules=False` for analyses that want raw EK output.

---

## 3. Event field definitions

A `Microsaccade` is a frozen dataclass with:

| field           | unit          | definition                                   |
|-----------------|---------------|----------------------------------------------|
| `start_idx`     | sample        | first supra-threshold sample                  |
| `end_idx`       | sample        | last supra-threshold sample                   |
| `start_time`    | s             | `start_idx / fs`                              |
| `end_time`      | s             | `end_idx / fs`                                |
| `duration`      | s             | `(end_idx - start_idx) / fs`                  |
| `amplitude`     | deg           | `√((x_end − x_start)² + (y_end − y_start)²)`  |
| `peak_velocity` | deg/s         | `max √(vx² + vy²)` over `[start_idx, end_idx]`|
| `direction`     | rad ∈ [-π, π] | `arctan2(y_end − y_start, x_end − x_start)`   |

`amplitude` is **start-to-end displacement**, not peak-to-peak position. This matches EK 2003 and the convention used by `pymovements`, `saccadr`, and the Engbert toolbox. Per-trial circular-mean direction
(in `features.trial_descriptors`) is computed via unit-vector averaging because arithmetic mean of angles is meaningless near the ±π wrap.

---