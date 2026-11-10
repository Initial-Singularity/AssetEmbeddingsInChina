"""The feature layer: evaluations whose period is a firm cross section described by feature columns.

[`FeatureEvaluator`][asset_embeddings.evaluators.feature.FeatureEvaluator] sits between the lifecycle
(`BaseEvaluator`) and the evaluations configured by the feature fields (RV, RC and their grouped variants, the
label probe). It declares the benchmark's characteristics, assembles each period with the feature columns
(`FeatureAssembler`), builds the feature selection once (`FeatureSpec`) and, after the period loop, checks that
the industry dummies agree across periods.
"""

import logging
from abc import abstractmethod
from typing import FrozenSet, List, Optional, Sequence

from .base import BaseEvaluator
from .data import FeatureAssembler, FeatureSpec, PeriodAssembler


def check_industry_drift(periods: Sequence[str], col_sets: Sequence[FrozenSet[str]], logger: logging.Logger) -> None:
    """Warn when the industry dummy columns differ between periods.

    The ``Ind_*`` columns should be the same every period (they are under the Shenwan level-1 classification). A
    drift does not distort a period's R²: an industry with no firm in a period would be an all-zero column whose
    coefficient the penalty drives to zero. The only second-order effect is the ``drop_first`` reference category,
    whose change shifts the geometry of the L2 penalty slightly. Cross-period panel or column-identity analyses do
    need the warning, since their column names and dimensions then differ between periods.

    Args:
        periods: The period labels, aligned with ``col_sets``.
        col_sets: Each period's dummy columns.
        logger: Where the warning goes.
    """
    if len(col_sets) <= 1:
        return
    union = frozenset().union(*col_sets)
    if all(s == union for s in col_sets):
        return
    missing = {p: sorted(union - s) for p, s in zip(periods, col_sets) if s != union}
    logger.warning(
        f"Cross-period industry dummy columns are not consistent; "
        f"per-period missing vs union: {missing}. Per-period R2 is essentially "
        f"unaffected (absent-industry dummies are inert under intercept Ridge; only "
        f"the drop_first reference, if it drifts, perturbs R2 second-order), but "
        f"cross-period panel/column-identity analysis must account for the differing "
        f"column sets. Consider a fixed category universe if this "
        f"fires regularly."
    )


class FeatureEvaluator(BaseEvaluator):
    """An evaluation over the feature columns of ``config.features`` (its config carries the feature fields).

    A subclass declares its characteristics (``char_cols``); ``self.features`` selects and scales the feature
    columns of a period's frame.
    """

    def __init__(self):
        super().__init__()
        self.features: Optional[FeatureSpec] = None

    @property
    @abstractmethod
    def char_cols(self) -> List[str]:
        """Firm characteristic columns of this benchmark (the ``"chars"`` block, and what ``align_chars`` reads)."""
        ...

    def make_assembler(self) -> PeriodAssembler:
        """The feature assembler: the firm universe plus the feature columns."""
        return FeatureAssembler(self.config, self.benchmark_columns, self.char_cols)

    def prepare(self) -> None:
        """Base preparation plus the feature spec."""
        super().prepare()
        self.features = FeatureSpec(self.config, self.char_cols)
        self.logger.info(f"Features {self.config.features}, align_chars {self.config.align_chars}.")

    def after_periods(self) -> None:
        """With the ``"industry"`` feature, warn when the periods' dummy columns differ."""
        if "industry" in self.config.features:
            periods = self.config.periods
            check_industry_drift(periods, [self.assembler.industry_cols[p] for p in periods], self.logger)
