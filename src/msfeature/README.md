# `msfeature` — package layout

End-to-end Engbert & Kliegl 2003 microsaccade detection plus the variants
specific to Nouri et al. 2025 (iScience). Each module owns one stage of
the pipeline and can be used independently.

| module          | purpose                                                          |
|-----------------|------------------------------------------------------------------|
| `config.py`     | `DatasetConfig`, `MONKEY_CONFIG`, `HUMAN_CONFIG`. All numeric defaults live here so callers do not hard-code parameters. |
| `io.py`         | `load_session()` for the monkey `.mat` layout (`sess.EyeX/EyeY/trial_isface/trial_stimulus_code`). Validates shapes and raises explicit errors on missing fields. |
| `velocity.py`   | `ek_velocity` / `ek_velocity_2d` — EK 2003 Eq. 1, the 5-sample symmetric kernel. |
| `detect.py`     | `compute_sigma`, `elliptic_threshold_mask`, `extract_runs`, and the orchestrating `detect_microsaccades()` that runs the whole pipeline. Sigma supports `engbert2003` and `engbert2015` forms. |
| `events.py`     | `Microsaccade` dataclass plus the Nouri-specific post-rules (refractory merge, amplitude cap, duration cap, amplitude-outlier rejection). |
| `preprocess.py` | `mark_blinks`, `remove_glitches` (MAD-based), `correct_drift` (regress on time), `lowpass`, and `preprocess_trial` to compose them. |
| `features.py`   | `RateBinSpec`, `binned_rate`, `trial_descriptors`, `shift_to_onset`, `stack_trial_features`. The 15-ms binned-rate matrix is what feeds the SVM. |
| `decode.py`     | Standard-scaler + SVM pipeline; `cross_validate()` for stratified k-fold. RBF kernel by default (matches paper). |

## Pipeline composition

```
load_session ─► preprocess_trial ─► detect_microsaccades ─► shift_to_onset
                                                                 │
                                                                 ▼
                                              stack_trial_features ─► cross_validate
```

For a worked end-to-end example see
[`../../notebooks/reproduce_monkey.ipynb`](../../notebooks/reproduce_monkey.ipynb).

## Algorithm details

See [`../../METHODS.md`](../../METHODS.md) for the equations, parameter
table, deviations from the paper, and reference citations.
