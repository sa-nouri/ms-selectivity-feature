# `src/`

Source layout follows the **src layout** convention (PEP 517 / pyproject):
the package lives one level down from this directory so that `pytest`,
linters, and editors don't accidentally pick up the working tree before
the package is installed.

The single package shipped here is `msfeature` — see
[`msfeature/README.md`](msfeature/README.md) for the module map and how
the pipeline composes.
