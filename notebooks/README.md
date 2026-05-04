# `notebooks/`

End-to-end demonstrations of the `msfeature` pipeline on the data
shipped with the repo.

| notebook                       | dataset                                  |
|--------------------------------|------------------------------------------|
| `reproduce_monkey.ipynb`       | `data/monkey_sample/sample_1_EyeData.mat` — load → preprocess → EK detection → 15 ms binned rate → RBF-SVM decoding |

## Running

The notebooks use a relative path `../data/...` so launch the kernel
with the repo root on the import path:

```bash
.venv/bin/jupyter notebook notebooks/reproduce_monkey.ipynb
```

To re-execute headlessly (used by CI):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/reproduce_monkey.ipynb --output reproduce_monkey.ipynb
```

## Style

- All plot styling lives in the first cell (seaborn theme + custom rc +
  fixed face / non-face palette). Subsequent cells should not change
  rcParams — keep the look uniform across panels.
- Use `sns.despine()` after each figure, and `plt.tight_layout()` before
  `plt.show()`.
- Annotate figures with quantitative info (slope of fits, peak times,
  baseline rate) in plain text rather than relying on the reader to
  infer it from the panel.

## Adding a new analysis notebook

Create `notebooks/<topic>.ipynb`, follow the styling cell convention,
and end with a "Notes" markdown cell that lists assumptions, parameter
choices, and where the analysis differs from the published reference.
