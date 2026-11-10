"""RV (relative valuation) benchmark evaluator.

Ridge fit of the period's valuation residual on the same-period embedding, scored by the cross-sectional
explanatory R². Run from ``python -m asset_embeddings.scripts.eval.rv``.
"""

from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import statsmodels.api as sm

from asset_embeddings.configs import RVEvalConfig, Config
from asset_embeddings.records import RVResult, Result
from .benchmark import RidgeBenchmarkEvaluator

#: Firm characteristics of the RV benchmark (beta, asset growth, profitability, dividend yield); the label probe's
#: characteristics baseline uses the same set.
RV_CHAR_COLS = ["beta_60m", "at_gr1", "gp_at", "div_at"]


def log_equity_target(merged_df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Build the RV target: the OLS residual of log book equity on log market equity over the period.

    Args:
        merged_df: One period's merged frame with ``be`` and ``me`` columns.
        mode: ``"drop"`` removes firms with non-positive ``be`` or ``me`` before the log and returns the kept
            frame with a fresh positional index, so residuals, embeddings and ``Token`` ids stay aligned.
            ``"keep"`` is the earlier behavior, retained so results recorded under it can be reproduced: such
            a firm's raw value (order 1e9) enters the OLS next to the logs of the other firms (about 22), and
            its residual dominates every fit and every R² that touches it.

    Returns:
        ``(frame, residuals, diagnostics)``: ``diagnostics`` holds ``ols_r_squared`` (logged, not recorded),
        ``n_dropped_nonpositive_equity`` and ``steps``, the ``nonpositive_equity_dropped`` funnel step.

    Raises:
        ValueError: On an unknown ``mode``, or when ``"drop"`` leaves no firm to fit.
    """
    if mode == "drop":
        ok = (merged_df["be"] > 0) & (merged_df["me"] > 0)
        n_drop = int((~ok).sum())
        if n_drop:
            merged_df = merged_df.loc[ok].reset_index(drop=True)
        if len(merged_df) == 0:
            raise ValueError("RV target: every firm has non-positive book or market equity; nothing left to fit")
        x, y = np.log(merged_df["me"]), np.log(merged_df["be"])
    elif mode == "keep":
        n_drop = 0
        x = merged_df["me"].apply(lambda v: v if v <= 0 else np.log(v))
        y = merged_df["be"].apply(lambda v: v if v <= 0 else np.log(v))
    else:
        raise ValueError(f"nonpositive_equity must be 'drop' or 'keep', got {mode!r}")
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    diagnostics = {
        "ols_r_squared": float(ols.rsquared),
        "n_dropped_nonpositive_equity": n_drop,
        "steps": [("nonpositive_equity_dropped", len(merged_df))],
    }
    return merged_df, ols.resid, diagnostics


class RVBenchmarkEvaluator(RidgeBenchmarkEvaluator):
    """Relative valuation benchmark: the ``log(be) ~ log(me)`` OLS residual regressed on the features by ridge.

    The parents supply the period loop, the data and the ridge; this class supplies the target
    ([`log_equity_target`][asset_embeddings.evaluators.rv.log_equity_target]) and the RV result class.
    """

    @property
    def config_class(self) -> Type[Config]:
        """`RVEvalConfig`: the shared benchmark fields plus ``nonpositive_equity``."""
        return RVEvalConfig

    @property
    def char_cols(self) -> List[str]:
        """Firm characteristics: beta, asset growth, profitability, dividend yield (``RV_CHAR_COLS``)."""
        return list(RV_CHAR_COLS)

    @property
    def benchmark_columns(self) -> List[str]:
        """Benchmark columns of the RV target: market equity ``me`` and book equity ``be``."""
        return ["me", "be"]

    @property
    def task_name(self) -> str:
        """``"RV"``, the task label written to the ``runs`` row and every result row."""
        return "RV"

    @property
    def result_class(self) -> Type[Result]:
        """`RVResult`, the per-period score row."""
        return RVResult

    def target(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
        """The valuation residual (firms with non-positive equity dropped first under ``"drop"``)."""
        return log_equity_target(frame, self.config.nonpositive_equity)

    def log_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Log the firms dropped for non-positive equity and the OLS R² of the valuation regression."""
        if diagnostics["n_dropped_nonpositive_equity"]:
            self.logger.info(
                f"Dropped {diagnostics['n_dropped_nonpositive_equity']} firm(s) with non-positive book or market "
                f"equity before the RV residual"
            )
        self.logger.debug(f"OLS regression completed, R²: {diagnostics['ols_r_squared']:.4f}")
