# `tests/`

Pytest suite for `msfeature`. Run from the repo root:

```bash
.venv/bin/python -m pytest        # all tests
.venv/bin/python -m pytest -q     # quiet
.venv/bin/python -m pytest --cov=src tests/   # with coverage
```

## Test modules

| file                              | what it covers                                           |
|-----------------------------------|----------------------------------------------------------|
| `conftest.py`                     | shared fixtures: `monkey_cfg`, `human_cfg`, and `build_trace` for synthetic input. |
| `test_velocity.py`                | EK 5-point kernel correctness, NaN propagation rules, edge / shape validation. |
| `test_detect.py`                  | `compute_sigma` (both methods), `elliptic_threshold_mask`, `extract_runs`, and the end-to-end detector on synthetic data with injected microsaccades. |
| `test_events.py`                  | post-detection rules: refractory merge (incl. cascade-cap), amplitude cap, duration cap, amplitude-outlier rejection. |
| `test_preprocess.py`              | blink masking, MAD-based glitch removal, time-based drift correction, low-pass filter NaN handling. |
| `test_features.py`                | `RateBinSpec` validation, binned-rate counting, `shift_to_onset`, circular-mean direction. |
| `test_decode.py`                  | SVM pipeline construction, stratified-CV plumbing, input validation. |
| `test_io.py`                      | `.mat` round-trip via `scipy.io.savemat`; exercises every error path in the loader. |
| `test_pymovements_crosscheck.py`  | sample-perfect parity with `pymovements.events.microsaccades` on synthetic input. Skipped if `pymovements` is unavailable. |
| `test_monkey_acceptance.py`       | end-to-end test on the shipped monkey sample: rate-trace timing must match the published nadir / rebound pattern, and event-level stats must stay in the typical primate microsaccade range. |

## Test philosophy

- Synthetic-data tests inject events with known timing / amplitude /
  direction and verify the detector recovers them. This is how we know
  the detector is correct *in principle*.
- The `pymovements` cross-check tests it is correct *relative to the
  reference implementation* — if both detectors disagree on the same
  input, one of us has a bug.
- The monkey-data acceptance test guards against silent regressions on
  real eye-tracking data: if a future change moves the suppression
  nadir or rebound peak outside the published window, that test fails.

## Adding tests

Per-module tests live in `test_<module>.py`. Acceptance tests on real
data go in `test_<dataset>_acceptance.py` and gate themselves on
`DATA_PATH.exists()` so the suite stays runnable without the data.
