"""Ridge estimation of the benchmark evaluators: penalty selection, cross-section and holdout scoring.

Two estimators share one interface. `CrossSectionRidge` fits one ridge on every firm of the period and scores it on
the same cross section. `HoldoutRidge` repeats a firm holdout (Koijen-style hash folds): per split the ridge is
fitted on the training firms and scored on the held-out ones, and the value is the median (or the mean) over the
splits. Each
estimator selects its penalty (``select_penalty``) and scores with a given penalty (``score``) separately, so a
caller can select once and score many resamples; ``fit`` does both.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Penalty grid of the cross-validated ridge. The first 25 points are the original 1e-5..1e5 grid, kept exactly so
# full-sample CV selections do not move; the six points up to 1e8 were appended because small high-dimensional
# subsamples (the coverage groups at d=128) saturated at the old 1e5 ceiling. Full-period samples never select the
# appended alphas.
CV_ALPHA_GRID = np.concatenate([np.logspace(-5, 5, 25, base=10), np.logspace(5.5, 8, 6, base=10)])

MIN_CV_ROWS = 10
N_CV_FOLDS = 10

SplitAggregate = Literal["median", "mean"]
_AGGREGATORS = {"median": np.median, "mean": np.mean}


def holdout_fold(stock_id: str, seed: int, n_folds: int) -> int:
    """Fold index of one firm in a firm split.

    Deterministic sha1 of ``"{seed}:{stock_id}"`` modulo ``n_folds``: independent of row order and of which other
    firms are present, so every model and universe puts the same firm in the same fold. Fold 0 is the test fold.

    Args:
        stock_id: The firm's token.
        seed: Split seed (the holdout seed plus the split index).
        n_folds: Number of hash folds.

    Returns:
        An integer in ``[0, n_folds)``.
    """
    return int(hashlib.sha1(f"{seed}:{stock_id}".encode()).hexdigest(), 16) % n_folds


def holdout_test_mask(stock_ids: Sequence[str], seed: int, n_folds: int) -> np.ndarray:
    """Boolean mask over ``stock_ids`` that is True where ``holdout_fold(id, seed, n_folds) == 0`` (the test rows)."""
    return np.fromiter((holdout_fold(str(s), seed, n_folds) == 0 for s in stock_ids), dtype=bool, count=len(stock_ids))


class UnscorableSplit(ValueError):
    """A holdout split that cannot be fitted or scored.

    Raised for too few training firms, no test firm, or a constant target on the test firms. A caller that scores
    many resamples (the grouped bootstrap) catches it to leave one out.
    """


@dataclass(frozen=True)
class SplitScore:
    """The score of one holdout split.

    Attributes:
        split: Split index; split 0 is the single Koijen split.
        value: Koijen test R²: ``1 - Var(resid) / Var(y_test)`` with population variances, residual demeaned.
        value_sklearn: The sklearn test R²: ``1 - SS_res / SS_tot``.
        alpha: The penalty of the split.
        n_train: Training firms.
        n_test: Held-out firms.
    """

    split: int
    value: float
    value_sklearn: float
    alpha: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class RidgeFit:
    """The score of one fit, as a value (no model, no arrays).

    Attributes:
        metric: ``"r2"`` (cross section) or ``"r2_holdout"``.
        value: The cross-sectional R², or the Koijen R² aggregated over the splits by ``aggregate``.
        alpha: The penalty used, or the median over the splits.
        n_scored: Rows the fit was given.
        splits: One entry per holdout split; empty for the cross section.
        aggregate: ``"median"`` or ``"mean"`` over the splits; None for the cross section.
    """

    metric: str
    value: float
    alpha: float
    n_scored: int
    splits: Tuple[SplitScore, ...] = ()
    aggregate: Optional[SplitAggregate] = None

    @property
    def holdout(self) -> bool:
        """Whether the value comes from holdout splits."""
        return bool(self.splits)

    @property
    def value_std(self) -> Optional[float]:
        """Standard deviation (``ddof=0``) of the Koijen split values; None for the cross section."""
        return float(np.std([s.value for s in self.splits])) if self.splits else None

    @property
    def value_sklearn(self) -> Optional[float]:
        """The sklearn split values aggregated like ``value``; None for the cross section."""
        if not self.splits:
            return None
        return float(_AGGREGATORS[self.aggregate or "mean"]([s.value_sklearn for s in self.splits]))


@dataclass(frozen=True)
class Penalty:
    """The selected penalty: one alpha for the cross section, one per split for the holdout."""

    alphas: Tuple[float, ...]


class PenaltySelector:
    """The fixed penalty, or the 10-fold cross-validated one over a grid.

    Args:
        fixed: The configured ``ridge_alpha``; None selects by CV.
        grid: Candidate penalties of the CV.
    """

    def __init__(self, fixed: Optional[float], grid: np.ndarray = CV_ALPHA_GRID):
        self.fixed = fixed
        self.grid = grid

    def select(self, X, y, seed: int) -> float:
        """The fixed penalty as configured, or the R²-maximizing grid point of a shuffled 10-fold CV.

        Raises:
            ValueError: When CV is requested on fewer than 10 rows. sklearn would otherwise fail deep inside the fit
                with "Cannot have number of splits n_splits=10 ..."; refusing here ties the failure to the penalty
                selection, where the cause (a sample shrunk by filtering) is visible.
        """
        if self.fixed is not None:
            return self.fixed
        n = len(y)
        if n < MIN_CV_ROWS:
            raise ValueError(
                f"CV alpha resolution requires >=10 samples for the default "
                f"10-fold KFold (got n={n}). Set config.ridge_alpha to a "
                f"fixed value, increase the dataset, or relax variant "
                f"filtering that may have shrunk the sample."
            )
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
        cv = RidgeCV(alphas=self.grid, cv=kf, scoring="r2").fit(X, y)
        logger.info(f"CV-resolved Ridge alpha: {cv.alpha_:.6f}")
        return float(cv.alpha_)


class _CenteredRidge:
    """sklearn ``Ridge``'s centered training rows and their Gram matrix, formed once and solved for many penalties.

    On a dense design without sample weights ``Ridge`` centers X and y, then solves the normal equations by Cholesky
    (``_solve_cholesky``, or ``_solve_cholesky_kernel`` on the kernel X X' when there are more features than rows).
    ``coef`` repeats those steps with the same arithmetic, so a coefficient equals ``Ridge``'s to the last bit while
    the per-call validation that dominates many small fits is skipped.
    """

    def __init__(self, X, y):
        Xc = np.array(X, dtype=float, order="K", copy=True)
        self.X_offset = np.average(Xc, axis=0)
        Xc -= self.X_offset
        yc = np.array(y, dtype=float, copy=True)
        self.y_offset = np.average(yc, axis=0)
        yc -= self.y_offset
        self.Xc, self.yc = Xc, yc.reshape(-1, 1)
        n, p = Xc.shape
        self.kernel = p > n
        self.gram = Xc @ Xc.T if self.kernel else Xc.T @ Xc
        self.Xy = None if self.kernel else Xc.T @ self.yc

    def coef(self, alpha: float) -> Tuple[np.ndarray, float]:
        """Coefficients and intercept at ``alpha``.

        Raises:
            numpy.linalg.LinAlgError: When the Cholesky solve rejects the system (sklearn then falls back to another
                solver, so the caller hands the fit to sklearn).
        """
        A = self.gram.copy()
        A.flat[:: A.shape[0] + 1] += alpha
        if self.kernel:
            coef = (self.Xc.T @ linalg.solve(A, self.yc, assume_a="pos", overwrite_a=False)).T
        else:
            coef = linalg.solve(A, self.Xy, assume_a="pos", overwrite_a=True).T
        coef = np.divide(coef.ravel(), np.ones(self.Xc.shape[1]))
        return coef, self.y_offset - np.dot(self.X_offset, coef)


def ridge_predict(X_train, y_train, X_test, alpha: float, seed: int) -> np.ndarray:
    """Predictions on ``X_test`` of ``Ridge(alpha, random_state=seed)`` fitted on the training rows, bitwise.

    A system the Cholesky solve rejects goes to ``Ridge`` itself, which falls back the way sklearn does.
    """
    try:
        coef, intercept = _CenteredRidge(X_train, y_train).coef(alpha)
    except linalg.LinAlgError:
        return Ridge(alpha=alpha, random_state=seed).fit(X_train, y_train).predict(X_test)
    return np.asarray(X_test, dtype=float) @ coef + intercept


def _r2(y_true: np.ndarray, denominator: np.ndarray, y_pred: np.ndarray) -> float:
    """``sklearn.metrics.r2_score`` of one output, with its arithmetic and its conventions.

    A single held-out row gives NaN (the score is undefined), a zero-variance target 1.0 for an exact prediction and
    0.0 otherwise.
    """
    if len(y_true) < 2:
        return float("nan")
    numerator = np.sum(1.0 * (y_true - y_pred.reshape(-1, 1)) ** 2, axis=0)
    if denominator[0] != 0 and numerator[0] != 0:
        return float(np.average(1 - (numerator / denominator)))
    return 1.0 if numerator[0] == 0 else 0.0


def cv_penalty(X, y, seed: int, grid: np.ndarray) -> float:
    """The penalty ``RidgeCV(alphas=grid, cv=KFold(10, shuffle=True, random_state=seed), scoring="r2")`` selects.

    With an explicit ``cv`` RidgeCV runs a grid search that fits ``Ridge`` once per fold and penalty (310 fits on the
    default grid). Here each fold's centered Gram matrix is formed once and every penalty solved from it, the fold
    R² follows ``r2_score``, the fold scores are averaged as the grid search does and the first maximum wins, so the
    selected penalty is bitwise RidgeCV's. A fold whose solve is rejected hands the whole selection to RidgeCV.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    scores = np.empty((len(grid), N_CV_FOLDS))
    try:
        for k, (train, test) in enumerate(kf.split(X)):
            fold = _CenteredRidge(X[train], y[train])
            y_test = y[test].reshape(-1, 1)
            denominator = np.sum(1.0 * (y_test - np.average(y_test, axis=0)) ** 2, axis=0)
            for a, alpha in enumerate(grid):
                coef, intercept = fold.coef(float(alpha))
                scores[a, k] = _r2(y_test, denominator, X[test] @ coef + intercept)
    except linalg.LinAlgError:
        return float(RidgeCV(alphas=grid, cv=kf, scoring="r2").fit(X, y).alpha_)
    return float(grid[_best_index(np.average(scores, axis=1))])


def _best_index(means: np.ndarray) -> int:
    """The grid search's choice: the first maximum, NaN ranked below every score, the first point when all are NaN.

    A training sample of 10 to 19 rows has a one-row fold in a 10-fold split, whose R² is NaN for every penalty.
    """
    if np.isnan(means).all():
        return 0
    return int(np.argmax(np.nan_to_num(means, nan=np.nanmin(means) - 1)))


def ridge_value(X, y, alpha: float, seed: int) -> float:
    """In-sample R² of one ridge fit; the inner call of a bootstrap."""
    return Ridge(alpha=alpha, random_state=seed).fit(X, y).score(X, y)


class RidgeEstimator(ABC):
    """Penalty selection and scoring of one period's ridge.

    Args:
        selector: How the penalty is chosen.
        seed: ``config.seed``: the ``Ridge`` random state (and the CV fold seed of the cross section).
    """

    def __init__(self, selector: PenaltySelector, seed: int):
        self.selector = selector
        self.seed = seed

    @abstractmethod
    def select_penalty(self, X, y, ids: Sequence[str]) -> Penalty:
        """Choose the penalty for ``(X, y)``."""
        ...

    @abstractmethod
    def score(self, X, y, ids: Sequence[str], penalty: Penalty) -> RidgeFit:
        """Fit and score with a given penalty."""
        ...

    def fit(self, X, y, ids: Sequence[str]) -> RidgeFit:
        """Select the penalty, then score with it."""
        return self.score(X, y, ids, self.select_penalty(X, y, ids))


class CrossSectionRidge(RidgeEstimator):
    """One ridge on every firm of the period, scored on the same firms (the default setting)."""

    def select_penalty(self, X, y, ids: Sequence[str]) -> Penalty:
        """The fixed penalty, or CV over all rows with folds seeded by ``seed``."""
        return Penalty((self.selector.select(X, y, self.seed),))

    def score(self, X, y, ids: Sequence[str], penalty: Penalty) -> RidgeFit:
        """In-sample R² of the ridge with ``penalty``."""
        (alpha,) = penalty.alphas
        return RidgeFit(metric="r2", value=ridge_value(X, y, alpha, self.seed), alpha=alpha, n_scored=len(y))


class HoldoutRidge(RidgeEstimator):
    """Repeated firm holdout: fit on the training firms, score on the held-out firms, aggregate over the splits.

    Split ``s`` uses seed ``split_seed + s``; a firm is held out when its hash fold is 0 among
    ``round(1 / test_size)`` folds. The penalty is the fixed one, or a 10-fold CV on the split's training rows with
    folds seeded by the split seed (as in Koijen), computed by ``cv_penalty`` with RidgeCV's result. Each firm's folds
    over all splits are hashed once and kept for the estimator's life, so penalty selection, scoring and every
    bootstrap resample reuse them.

    Args:
        selector: How the penalty is chosen.
        seed: ``config.seed``, the ``Ridge`` random state.
        test_size: Held-out share per split.
        n_splits: Number of splits.
        split_seed: Seed of split 0.
        aggregate: ``"median"`` or ``"mean"`` of the split values.

    Raises:
        ValueError: On an unknown ``aggregate``.
    """

    def __init__(
        self,
        selector: PenaltySelector,
        seed: int,
        test_size: float,
        n_splits: int,
        split_seed: int,
        aggregate: SplitAggregate = "median",
    ):
        super().__init__(selector, seed)
        if aggregate not in _AGGREGATORS:
            raise ValueError(f"aggregate must be one of {sorted(_AGGREGATORS)}, got {aggregate!r}")
        self.n_folds = int(round(1.0 / test_size))
        self.n_splits = n_splits
        self.split_seed = split_seed
        self.aggregate = aggregate
        self._folds: Dict[str, np.ndarray] = {}

    def _fold_matrix(self, ids: List[str]) -> np.ndarray:
        """``(len(ids), n_splits)`` hash folds: ``holdout_fold(id, split_seed + s, n_folds)``, cached per firm."""
        for stock_id in dict.fromkeys(ids):
            if stock_id not in self._folds:
                self._folds[stock_id] = np.fromiter(
                    (holdout_fold(stock_id, self.split_seed + s, self.n_folds) for s in range(self.n_splits)),
                    dtype=np.int64,
                    count=self.n_splits,
                )
        return np.stack([self._folds[stock_id] for stock_id in ids]) if ids else np.empty((0, self.n_splits), int)

    def _masks(self, y, ids: Sequence[str]):
        """Per split ``(s, seed, train, test)``, after checking every split can be fitted and scored.

        Raises:
            ValueError: When ``ids`` and ``y`` differ in length.
            UnscorableSplit: When a split has fewer than 10 training rows (the KFold minimum, enforced under both
                penalty modes) or no test row. A bad split is never skipped: the mean would then be over a different
                set of firms than recorded.
        """
        ids = [str(s) for s in ids]
        if len(ids) != len(y):
            raise ValueError(f"ids ({len(ids)}) and y ({len(y)}) differ in length")
        folds = self._fold_matrix(ids)
        for s in range(self.n_splits):
            seed = self.split_seed + s
            test = folds[:, s] == 0
            train = ~test
            n_train, n_test = int(train.sum()), int(test.sum())
            if n_train < MIN_CV_ROWS or n_test == 0:
                raise UnscorableSplit(
                    f"holdout split {s} (seed {seed}): {n_train} training and {n_test} test firms; "
                    f"need >= 10 training rows for the 10-fold RidgeCV and a non-empty test set"
                )
            yield s, seed, train, test

    def select_penalty(self, X, y, ids: Sequence[str]) -> Penalty:
        """One penalty per split: the fixed one, or CV on the split's training rows."""
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        alphas = []
        for _, seed, train, _ in self._masks(y_arr, ids):
            if self.selector.fixed is None:
                alphas.append(cv_penalty(X_arr[train], y_arr[train], seed, self.selector.grid))
            else:
                alphas.append(float(self.selector.fixed))
        return Penalty(tuple(alphas))

    def score(self, X, y, ids: Sequence[str], penalty: Penalty) -> RidgeFit:
        """Koijen and sklearn test R² per split with that split's penalty; aggregated value, median penalty.

        Raises:
            ValueError: When ``penalty`` does not hold one alpha per split (the value would silently cover fewer
                splits).
            UnscorableSplit: On a split that cannot be fitted or scored (see ``_masks``), or zero target variance on
                a split's test firms.
        """
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if len(penalty.alphas) != self.n_splits:
            raise ValueError(f"penalty has {len(penalty.alphas)} alpha(s) for {self.n_splits} holdout splits")
        splits = []
        for (s, seed, train, test), alpha in zip(self._masks(y_arr, ids), penalty.alphas):
            y_test = y_arr[test]
            resid = y_test - ridge_predict(X_arr[train], y_arr[train], X_arr[test], alpha, self.seed)
            var_y = float(np.var(y_test))
            if var_y == 0.0:
                raise UnscorableSplit(
                    f"holdout split {s} (seed {seed}): zero target variance on the {int(test.sum())} test firms"
                )
            value = 1.0 - float(np.var(resid)) / var_y
            value_sklearn = 1.0 - float(np.sum(resid**2)) / float(np.sum((y_test - y_test.mean()) ** 2))
            splits.append(SplitScore(s, value, value_sklearn, alpha, int(train.sum()), int(test.sum())))
            logger.debug(
                f"holdout split {s}: alpha={alpha:.6g} r2={value:.4f} r2_sklearn={value_sklearn:.4f} "
                f"n_train={int(train.sum())} n_test={int(test.sum())}"
            )
        return RidgeFit(
            metric="r2_holdout",
            value=float(_AGGREGATORS[self.aggregate]([sp.value for sp in splits])),
            alpha=float(np.median([sp.alpha for sp in splits])),
            n_scored=len(y_arr),
            splits=tuple(splits),
            aggregate=self.aggregate,
        )


def make_estimator(config, grid: np.ndarray = CV_ALPHA_GRID) -> RidgeEstimator:
    """The estimator ``config.scoring`` asks for: `HoldoutRidge` for ``holdout``, `CrossSectionRidge` otherwise."""
    selector = PenaltySelector(config.ridge_alpha, grid)
    if config.scoring == "holdout":
        return HoldoutRidge(
            selector,
            config.seed,
            config.holdout_test_size,
            config.holdout_n_splits,
            config.holdout_seed,
            config.holdout_split_aggregate,
        )
    return CrossSectionRidge(selector, config.seed)
