"""SVM decoding from microsaccade-rate features.

The paper reports ~85% accuracy on a face-vs-non-face binary task using
linear classifiers on the binned-rate feature vector. This module wraps
that recipe in a thin scikit-learn pipeline so callers don't have to
assemble it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class DecodeResult:
    fold_scores: np.ndarray
    mean_score: float
    sd_score: float
    n_train: int
    n_test_per_fold: list[int]


def make_classifier(
    C: float = 1.0,
    kernel: str = "rbf",
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Standard-scaler + SVM pipeline.

    Defaults match Nouri et al. 2025's reported decoder: RBF kernel on
    standardized features, no class balancing. Pass
    `class_weight='balanced'` for honest performance on imbalanced
    data (the monkey sample is 225 face / 549 non-face).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel=kernel, class_weight=class_weight)),
        ]
    )


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    C: float = 1.0,
    kernel: str = "rbf",
    class_weight: str | dict | None = None,
    scoring: str = "accuracy",
    random_state: int = 0,
) -> DecodeResult:
    """Stratified k-fold cross-validation on a feature matrix."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of trials")
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D feature matrix")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(
            f"need at least 2 classes for stratified CV, got {classes.size}"
        )
    if counts.min() < n_splits:
        raise ValueError(
            f"smallest class has {counts.min()} samples but n_splits={n_splits}"
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = make_classifier(C=C, kernel=kernel, class_weight=class_weight)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)
    test_sizes = [len(test) for _, test in cv.split(X, y)]
    return DecodeResult(
        fold_scores=scores,
        mean_score=float(np.mean(scores)),
        sd_score=float(np.std(scores, ddof=1) if scores.size > 1 else 0.0),
        n_train=int(X.shape[0]),
        n_test_per_fold=test_sizes,
    )
