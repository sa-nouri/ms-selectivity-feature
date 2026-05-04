# `data/`

Sample eye-tracking recordings for development and testing. Both
subdirectories are kept under version control so the test suite and
notebooks run out of the box.

## `monkey_sample/`

Single Matlab v5 `.mat` session (two files with identical content; same
MD5 — they are renamed copies of one another):

- `sample_1_EyeData.mat`

Top-level struct `sess` with fields:

| field                  | shape         | meaning                                              |
|------------------------|---------------|------------------------------------------------------|
| `sess.EyeX`            | (774, 900)    | horizontal gaze position, degrees of visual angle    |
| `sess.EyeY`            | (774, 900)    | vertical gaze position, degrees                      |
| `sess.trial_isface`    | (774,)        | uint8, 1 if face stimulus, 0 if non-face             |
| `sess.trial_stimulus_code` | (774,)    | uint8, stimulus identity (1..155)                    |

There is **no timestamp channel and no stimulus-onset marker** in the
file. Working assumptions for this sample (see `MONKEY_CONFIG` in
`src/msfeature/config.py`):

- Sampling rate: **1 kHz** (so 900 samples = 900 ms / trial). The paper
  text says 2 kHz monkey, but the trial layout fits 1 kHz cleanly:
  60 data points per trial = 20 baseline + 40 post-stim at 15 ms bins.
- Stimulus onset: **sample 250** (= 250 ms baseline + 650 ms post-stim).
  The fit comes from matching the published microsaccade-rate
  suppression nadir (~210-230 ms) and rebound peak (~380-395 ms).

Class balance: 225 face / 549 non-face trials.

Loader: `msfeature.io.load_session(path)`. The loader validates shapes
and raises clear errors if the file does not have the expected layout.

## `human_sample/`

Three `.mat` files in a different layout from monkey (one each for eye
positions, message stream, and task data):

- `sample_1_EyePositionList.mat`
- `sample_1_EyeMessageList.mat`
- `sample_1_TaskData.mat`

A loader for this layout is **not yet implemented** — focus is on the
monkey pipeline first. When adding a human loader, keep the public API
parallel to `load_session()` (return an `EyeDataset` or a similar
dataclass) so downstream code does not branch on dataset type.
