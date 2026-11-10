"""The ridge benchmarks (RV, RC): a period's target regressed on its features by ridge.

[`RidgeBenchmarkEvaluator`][asset_embeddings.evaluators.benchmark.RidgeBenchmarkEvaluator] owns the period: it asks
the subclass for the target (``target``), builds the design matrix (``design``), fits it with the estimator the
config selects (cross section or holdout) and builds the period's rows: the result, one row per holdout split and
the period's sample funnel.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from asset_embeddings.records import Record, Result, RidgeSplit, SampleStep

from .base import PeriodOutcome
from .data import PeriodFrame
from .estimation import RidgeEstimator, RidgeFit, make_estimator
from .feature import FeatureEvaluator


class RidgeBenchmarkEvaluator(FeatureEvaluator):
    """A benchmark whose period is one ridge fit of a target on the feature matrix.

    Hooks: ``target`` (abstract), ``design`` (the normalized features by default), ``log_diagnostics``,
    ``make_estimator``.
    """

    def __init__(self):
        super().__init__()
        self.estimator: Optional[RidgeEstimator] = None

    @property
    @abstractmethod
    def result_class(self) -> Type[Result]:
        """The `Result` subclass of the score rows."""
        ...

    @abstractmethod
    def target(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
        """Build the period's target.

        Args:
            frame: The period's assembled frame.

        Returns:
            ``(frame, y, diagnostics)``: the frame the design is built from, holding exactly the rows the target is
            fitted on (a target that drops rows returns the reduced frame, so dropped rows do not enter the feature
            scaling), the target values in the frame's row order, and what ``log_diagnostics`` reports.
            ``diagnostics["steps"]``, when present, lists the ``(step, n_rows)`` funnel steps the target added; the
            period's funnel gets them once (the grouped evaluation calls ``target`` again on every resample).
        """
        ...

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        """The feature matrix of ``frame``: the selected columns, normalized."""
        return self.features.matrix(frame)

    def log_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Log what ``target`` found once per fitted frame (not per resample); nothing by default."""

    def make_estimator(self) -> RidgeEstimator:
        """The estimator of this evaluation, from the config's penalty and scoring fields."""
        return make_estimator(self.config)

    def prepare(self) -> None:
        """Feature preparation plus the estimator."""
        super().prepare()
        self.estimator = self.make_estimator()

    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """Fit the period's target; the value is its score.

        Records: the result, one split row per holdout split, and the period's sample funnel.
        """
        frame, y, diagnostics = self.target(pf.frame)
        for step, n_rows in diagnostics.get("steps", ()):
            pf.sample.add(step, n_rows)
        self.log_diagnostics(diagnostics)
        n_features = len(self.features.columns(frame))
        X = self.design(frame)
        self.logger.debug(f"Feature matrix shape: {X.shape}")

        self._warn_if_constant(pf.period, y)
        fit = self.estimator.fit(X, y, frame["Token"].to_numpy())
        self._log_fit(fit)
        records: List[Record] = [self.build_result(pf.period, fit, n_features)]
        records.extend(self.build_splits(pf.period, fit))
        records.extend(self.build_sample(pf))
        return PeriodOutcome(value=fit.value, records=records)

    def _warn_if_constant(self, period: str, y) -> None:
        """Warn when the target does not vary across firms; its row is still written as computed.

        A constant target is predicted exactly by the intercept, and sklearn scores that as an R² of 1.0.
        """
        y = np.asarray(y, dtype=float)
        if len(y) and np.ptp(y) <= 1e-12 * max(1.0, float(np.max(np.abs(y)))):
            self.logger.warning(
                f"{period}: the target is constant across {len(y)} firms; its R² carries no information "
                f"(recorded as computed)"
            )

    def _log_fit(self, fit: RidgeFit) -> None:
        self.logger.debug(f"Ridge R² = {fit.value:.4f} (alpha={fit.alpha:.6g}, n={fit.n_scored})")
        if fit.holdout:
            self.logger.info(
                f"holdout R²: {fit.aggregate} {fit.value:.4f} (sd {fit.value_std:.4f}) over {len(fit.splits)} splits, "
                f"median alpha {fit.alpha:.4g}"
            )

    def build_result(self, period: str, fit: RidgeFit, n_features: int) -> Result:
        """The period's result row; ``run_id`` is filled by ``Run.save``."""
        return self.result_class(
            period=period,
            metric=fit.metric,
            value=float(fit.value),
            # float() because a fixed ridge_alpha read from JSON may be an integer (1000).
            ridge_alpha=float(fit.alpha),
            n_scored=int(fit.n_scored),
            n_features=n_features,
        )

    @staticmethod
    def build_splits(period: str, fit: RidgeFit) -> List[RidgeSplit]:
        """One ``eval_ridge_splits`` row per holdout split; none for the cross section."""
        return [
            RidgeSplit(
                period=period,
                split=s.split,
                value=float(s.value),
                value_sklearn=float(s.value_sklearn),
                alpha=float(s.alpha),
                n_train=int(s.n_train),
                n_test=int(s.n_test),
            )
            for s in fit.splits
        ]
