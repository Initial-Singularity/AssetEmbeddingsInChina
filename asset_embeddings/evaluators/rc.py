"""RC (return comovement) benchmark evaluator.

Ridge fit of the one-quarter-ahead return on the same-period embedding, scored by the cross-sectional
predictive R². Run from ``python -m asset_embeddings.scripts.eval.rc``.
"""

from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from asset_embeddings.configs import RCEvalConfig, Config
from asset_embeddings.records import RCResult, Result
from .benchmark import RidgeBenchmarkEvaluator


class RCBenchmarkEvaluator(RidgeBenchmarkEvaluator):
    """Return comovement benchmark: the next-quarter return ``qret_next`` regressed on the features by ridge.

    Differs from RV in the target (the raw next-quarter return, no OLS step) and in the RC result class.
    """

    @property
    def config_class(self) -> Type[Config]:
        """`RCEvalConfig`: the shared benchmark fields."""
        return RCEvalConfig

    @property
    def char_cols(self) -> List[str]:
        """Firm characteristics: beta, size, book-to-market, asset growth, profitability, momentum."""
        return ["beta_60m", "me", "be_me", "at_gr1", "gp_at", "ret_12_1"]

    @property
    def benchmark_columns(self) -> List[str]:
        """Benchmark column of the RC target: the next-quarter return ``qret_next``."""
        return ["qret_next"]

    @property
    def task_name(self) -> str:
        """``"RC"``, the task label written to the ``runs`` row and every result row."""
        return "RC"

    @property
    def result_class(self) -> Type[Result]:
        """`RCResult`, the per-period score row."""
        return RCResult

    def target(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
        """The next-quarter return ``qret_next``."""
        y = frame["qret_next"]
        self.logger.debug(f"Target stats - mean: {y.mean():.4f}, std: {y.std():.4f}")
        return frame, y, {}
