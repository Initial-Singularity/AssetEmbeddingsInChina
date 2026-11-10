"""Temporal RC evaluator: RC slopes fitted on earlier periods and applied, unchanged, to a later one.

For each period T after the first, a ridge of each earlier period s's target (the period-(s+1) return) on s's
features gives a slope. Under ``slope_window: "previous"`` only the slope of T-1 is applied to period T's features;
under ``"expanding"`` the predictions of the slopes of every earlier period are averaged, which in a fixed frame is
the prediction of the mean slope. The prediction is scored against period T's target (the period-(T+1) return) by
the Koijen R². Only the slope is carried over: the residual is demeaned, so the intercept is free in every period and
absorbs that period's common return, which a cross-sectional R² does not credit anyway. With ``align:
"procrustes"`` period T's raw embedding is first rotated onto each earlier period's by the orthogonal Procrustes
rotation fitted on the firms present in both, so that a frame that rotates between periods does not count against
the slope. Run from ``python -m asset_embeddings.scripts.eval.rc_temporal``.
"""

from dataclasses import dataclass
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from asset_embeddings.configs import Config, TemporalRCEvalConfig
from asset_embeddings.records import Record

from .base import PeriodOutcome
from .data import PeriodFrame
from .estimation import procrustes_rotation
from .rc import RCBenchmarkEvaluator


@dataclass(frozen=True)
class _Period:
    """What a period leaves for the next one: its firms, raw (unnormalized, unrotated) features and target."""

    period: str
    tokens: np.ndarray
    cols: List[str]
    raw: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class _Slope:
    """A ridge fitted on one earlier period, kept for the periods scored after it."""

    period: _Period
    model: Ridge
    alpha: float


class TemporalRCEvaluator(RCBenchmarkEvaluator):
    """RC with the slope estimated on earlier periods: fit on T-1 (or on every period before T), score on T.

    The first period has no predecessor: its funnel is written but no result row, its outcome value is NaN, and the
    summary is the mean over the scored periods.
    """

    def __init__(self):
        super().__init__()
        self._last: Optional[_Period] = None
        self._slopes: List[_Slope] = []

    @property
    def config_class(self) -> Type[Config]:
        """`TemporalRCEvalConfig`: the RC fields plus ``align`` and ``slope_window``."""
        return TemporalRCEvalConfig

    @property
    def task_name(self) -> str:
        """``"RC_temporal"``, the task label written to the ``runs`` row and every result row."""
        return "RC_temporal"

    @property
    def metric(self) -> str:
        """``r2_temporal_aligned`` under Procrustes alignment, ``r2_temporal`` otherwise."""
        return "r2_temporal_aligned" if self.config.align == "procrustes" else "r2_temporal"

    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """Score period T with the slope(s) fitted on earlier periods; the first period is stored, not scored.

        Records: the result (none for the first period) and the period's sample funnel. The result's
        ``ridge_alpha`` is the mean penalty of the slopes applied.

        Raises:
            ValueError: When the feature columns differ from the previous period's, or (under alignment) when the
                common firms do not determine a rotation.
        """
        frame, y, diagnostics = self.target(pf.frame)
        for step, n_rows in diagnostics.get("steps", ()):
            pf.sample.add(step, n_rows)
        self.log_diagnostics(diagnostics)
        cols = self.features.columns(frame)
        current = _Period(
            period=pf.period,
            tokens=frame["Token"].to_numpy(),
            cols=cols,
            raw=frame[cols].to_numpy(dtype=float),
            y=np.asarray(y, dtype=float),
        )
        last, self._last = self._last, current
        if last is None:
            self.logger.info(f"{pf.period}: first period, stored for the next one and not scored")
            return PeriodOutcome(value=float("nan"), records=self.build_sample(pf))
        self._check_columns(last, current)
        self._slopes.append(self._fit(last))
        if self.config.slope_window == "previous":
            # Earlier slopes are never applied again.
            self._slopes = self._slopes[-1:]

        predictions = [s.model.predict(self._features_in_frame(s.period, current)) for s in self._slopes]
        prediction = np.mean(predictions, axis=0)
        alpha = float(np.mean([s.alpha for s in self._slopes]))
        first, latest = self._slopes[0].period.period, self._slopes[-1].period.period
        span = latest if first == latest else f"{first}-{latest}"
        self.logger.info(f"{pf.period}: scored with {len(self._slopes)} slope(s) fitted on {span}")

        self._warn_if_constant(pf.period, current.y)
        # np.var demeans the residual, so the intercept is free: only the slope is carried over.
        value = 1.0 - float(np.var(current.y - prediction)) / float(np.var(current.y))
        row = self.result_class(
            period=pf.period,
            metric=self.metric,
            value=value,
            ridge_alpha=alpha,
            n_scored=len(current.y),
            n_features=len(cols),
        )
        records: List[Record] = [row]
        records.extend(self.build_sample(pf))
        return PeriodOutcome(value=value, records=records)

    def _fit(self, period: _Period) -> _Slope:
        """The ridge of ``period``'s target on its own normalized features, with the configured penalty choice."""
        X = self.features.normalize(pd.DataFrame(period.raw, columns=period.cols))
        alpha = float(self.estimator.selector.select(X, period.y, self.config.seed))
        model = Ridge(alpha=alpha, random_state=self.config.seed).fit(X, period.y)
        self.logger.info(f"{period.period}: slope fitted over {len(period.y)} firms, alpha {alpha:.6g}")
        return _Slope(period=period, model=model, alpha=alpha)

    def _features_in_frame(self, earlier: _Period, current: _Period) -> pd.DataFrame:
        """Period T's normalized features in the frame of ``earlier``, the period whose slope is applied."""
        raw = self._align(earlier, current) if self.config.align == "procrustes" else current.raw
        return self.features.normalize(pd.DataFrame(raw, columns=current.cols))

    @staticmethod
    def _check_columns(prev: _Period, current: _Period) -> None:
        if current.cols == prev.cols:
            return
        if len(current.cols) != len(prev.cols):
            detail = f"{len(prev.cols)} vs {len(current.cols)} columns"
        else:
            i = next(i for i, (a, b) in enumerate(zip(prev.cols, current.cols)) if a != b)
            detail = f"column {i} is {prev.cols[i]!r} vs {current.cols[i]!r}"
        raise ValueError(
            f"feature columns of {prev.period} and {current.period} differ ({detail}); the slope fitted on one "
            f"cannot be applied to the other"
        )

    def _align(self, prev: _Period, current: _Period) -> np.ndarray:
        """Period T's raw features rotated onto an earlier period's frame, centered on the common firms.

        The earlier matrix stays in its own coordinates and uncentered; the per-period normalization recenters both.
        """
        _, i_prev, i_cur = np.intersect1d(prev.tokens, current.tokens, return_indices=True)
        center = current.raw[i_cur].mean(axis=0)
        A = current.raw[i_cur] - center
        B = prev.raw[i_prev] - prev.raw[i_prev].mean(axis=0)
        n_common, dim = len(i_cur), len(current.cols)
        try:
            R = procrustes_rotation(A, B)
        except ValueError as e:
            raise ValueError(
                f"{current.period}: cannot align onto {prev.period} over {n_common} common firms at dimension "
                f"{dim}: {e}"
            ) from e
        s_min = float(np.linalg.svd(A.T @ B, compute_uv=False)[-1])
        self.logger.info(
            f"{current.period}: Procrustes rotation onto {prev.period} over {n_common} common firms, "
            f"smallest singular value {s_min:.6g}"
        )
        return (current.raw - center) @ R

    def summarize(self, outcomes: List[PeriodOutcome]) -> PeriodOutcome:
        """Log the mean over the scored periods (the first is not scored) and return it.

        Raises:
            ValueError: When no period was scored (a config with one period).
        """
        values = [o.value for o in outcomes if np.isfinite(o.value)]
        if not values:
            raise ValueError(f"{self.task_name}: no period was scored; temporal RC needs at least two periods")
        mean = float(np.mean(values))
        self.logger.info(f"Mean {self.task_name} over {len(values)} scored period(s): {mean}")
        self.logger.info(f"{self.task_name} Benchmark Evaluation Completed.")
        return PeriodOutcome(value=mean)
