"""The label probe's estimation: a stratified k-fold logistic regression scored by the test-fold ROC AUC."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class UnscorableProbe(ValueError):
    """The label holds one class, the period has fewer firms than folds, or no fold could be scored."""


@dataclass(frozen=True)
class FoldScore:
    """One scored fold.

    Attributes:
        fold: Fold index in the split's order, from 0.
        auc: ROC AUC on the fold's test firms.
        n_train: Training firms.
        n_test: Test firms.
        n_test_positive: Test firms with label 1.
    """

    fold: int
    auc: float
    n_train: int
    n_test: int
    n_test_positive: int


@dataclass(frozen=True)
class ProbeFit:
    """The probe of one period.

    Attributes:
        value: Mean test-fold AUC over the scored folds.
        folds: The scored folds.
    """

    value: float
    folds: List[FoldScore] = field(default_factory=list)


def cross_validated_auc(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> ProbeFit:
    """Mean test-fold ROC AUC of a logistic regression of ``y`` on ``X`` over a stratified k-fold split.

    The split is ``StratifiedKFold(n_folds, shuffle=True, random_state=seed)``; each fold fits
    ``LogisticRegression(max_iter=1000)`` (L2, C=1) on its training part. A fold whose test or training part holds a
    single class is skipped: its AUC is undefined, or the logit cannot be fitted (a class with one firm).

    Args:
        X: Features, one row per firm (already scaled).
        y: Labels 0/1.
        n_folds: Number of folds.
        seed: Seed of the split.

    Returns:
        The mean AUC and the scored folds.

    Raises:
        UnscorableProbe: When ``y`` holds one class, has fewer firms than folds, or no fold can be scored.
    """
    if len(np.unique(y)) < 2:
        raise UnscorableProbe("the label holds one class")
    if len(y) < n_folds:
        # StratifiedKFold refuses more splits than samples.
        raise UnscorableProbe(f"fewer firms ({len(y)}) than folds ({n_folds})")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(X, y)):
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            continue
        model = LogisticRegression(max_iter=1000).fit(X[train], y[train])
        auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
        folds.append(
            FoldScore(
                fold=fold,
                auc=float(auc),
                n_train=len(train),
                n_test=len(test),
                n_test_positive=int(y[test].sum()),
            )
        )
    if not folds:
        raise UnscorableProbe("no fold holds both classes in its training and test parts")
    return ProbeFit(value=float(np.mean([f.auc for f in folds])), folds=folds)
