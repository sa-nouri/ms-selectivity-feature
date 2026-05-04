# Microsaccade Selectivity as Discriminative Feature

Microsaccade detection, feature extraction, and decoding code for Nouri et al. 2025, *Microsaccade selectivity as discriminative feature for object decoding*, iScience 28(1):111584.


## Layout

```
ms-selectivity-feature/
├── src/                      see src/README.md
│   └── msfeature/            see src/msfeature/README.md
├── tests/                    see tests/README.md          (82 passing)
├── notebooks/                see notebooks/README.md
│   └── reproduce_monkey.ipynb
├── data/                     see data/README.md
│   ├── monkey_sample/
│   └── human_sample/
├── .github/workflows/ci.yml  GitHub Actions: install + pytest
├── METHODS.md                algorithm spec, parameters, deviations
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml            single source of truth for build config
└── requirements.txt
```

Every top-level subdirectory carries its own `README.md` describing what
lives there and how to use it.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # tests + linting + pymovements crosscheck
```

## Quick start

```python
from msfeature.io import load_session
from msfeature.config import MONKEY_CONFIG
from msfeature.preprocess import preprocess_trial
from msfeature.detect import detect_microsaccades

ds = load_session("data/monkey_sample/sample_1_EyeData.mat")
x, y, _ = preprocess_trial(
    ds.eye_x[0], ds.eye_y[0],
    sampling_rate_hz=MONKEY_CONFIG.sampling_rate_hz,
    lowpass_cutoff_hz=MONKEY_CONFIG.lowpass_cutoff_hz,
)
events = detect_microsaccades(x, y, MONKEY_CONFIG)
for ev in events:
    print(ev)
```

The full pipeline (preprocess → detect → bin rate → SVM decode) is in
[`notebooks/reproduce_monkey.ipynb`](notebooks/reproduce_monkey.ipynb).

## Tests

```bash
.venv/bin/python -m pytest
# 82 passed
```

Tests are organized into per-module units, a reference cross-check
against `pymovements`, and an end-to-end acceptance test on the shipped
monkey sample that verifies the suppression-rebound rate-trace timing
matches the published values. See `tests/README.md`.

## Algorithm + parameter details

`METHODS.md` documents:
- the EK 2003 equations and the two canonical sigma forms (`engbert2003`
  vs `engbert2015`)
- per-dataset parameter values (sampling rate, λ, min/max duration,
  amplitude cap, refractory window, ...) and their literature sources
- known deviations from the published methods and why
- the bug list inherited from the previous library

## Citation

```bibtex
@article{nouri2025microsaccade,
  title  = {Microsaccade selectivity as discriminative feature for object decoding},
  author = {Nouri, Salar and Tehrani, Amirali Soltani and Faridani, Niloufar
            and Toosi, Ramin and Noroozi, Jalaledin and Dehaqani, Mohammad-Reza A},
  journal= {iScience},
  volume = {28},
  number = {1},
  pages  = {111584},
  year   = {2025},
}
```

## License

MIT — see `LICENSE`.
