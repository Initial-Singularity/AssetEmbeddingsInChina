"""Label probe: how well a binary firm label (such as state ownership) can be recovered from the embedding.

Per period the label file is joined onto the period's frame (unlabeled firms leave), the features are scaled on the
whole cross section as in RV and RC, and a logistic regression of the label on them is scored by the mean test-fold
ROC AUC of a stratified k-fold split. The features follow ``config.features``, so ``["chars"]`` is the
characteristics baseline. Run from ``python -m asset_embeddings.scripts.eval.label_probe``.
"""

from typing import List, Type

import numpy as np
import pandas as pd

from asset_embeddings.configs import Config, LabelProbeEvalConfig
from asset_embeddings.records import ProbeFold, ProbeResult, Record

from .base import PeriodOutcome
from .data import LabelFileSource, PeriodFrame, PeriodSource
from .estimation import UnscorableProbe, cross_validated_auc
from .feature import FeatureEvaluator
from .rv import RV_CHAR_COLS


class LabelProbeEvaluator(FeatureEvaluator):
    """Cross-validated logit AUC of a binary firm label on the features (the embedding, or a baseline).

    A period whose label holds one class, that has fewer labeled firms than folds, or none of whose folds can be
    scored writes no result row; its sample funnel is still recorded and a warning names the reason.
    """

    @property
    def config_class(self) -> Type[Config]:
        """`LabelProbeEvalConfig`: the data, feature and label fields plus ``n_folds``."""
        return LabelProbeEvalConfig

    @property
    def task_name(self) -> str:
        """``"PROBE"``."""
        return "PROBE"

    @property
    def char_cols(self) -> List[str]:
        """The RV characteristics (``RV_CHAR_COLS``), the features of the characteristics baseline."""
        return list(RV_CHAR_COLS)

    def sources(self) -> List[PeriodSource]:
        """The label file."""
        cfg = self.config
        return [LabelFileSource(cfg.label_path, cfg.label_stock_col, cfg.label_col)]

    def labels(self, pf: PeriodFrame) -> np.ndarray:
        """The period's labels as integers.

        Raises:
            ValueError: When the label takes values other than 0 and 1.
        """
        label_col = self.config.label_col
        values = set(pd.unique(pf.frame[label_col]))
        if not values <= {0, 1}:
            raise ValueError(
                f"label_col {label_col!r} must be binary 0/1; found values {sorted(values)} in period {pf.period}"
            )
        return pf.frame[label_col].astype(int).to_numpy()

    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """Probe one period: the result row, one row per scored fold and the funnel; no result row if unscorable."""
        cfg = self.config
        y = self.labels(pf)
        n_features = len(self.features.columns(pf.frame))
        X = self.features.matrix(pf.frame)
        try:
            fit = cross_validated_auc(X, y, cfg.n_folds, cfg.seed)
        except UnscorableProbe as err:
            self.logger.warning(f"{pf.period}: not scored, {err} ({len(y)} firms, {int(y.sum())} with label 1)")
            return PeriodOutcome(value=float("nan"), records=self.build_sample(pf))

        records: List[Record] = [
            ProbeResult(
                period=pf.period,
                metric="auc",
                value=fit.value,
                n_scored=len(y),
                n_positive=int(y.sum()),
                n_features=n_features,
                n_folds_used=len(fit.folds),
            )
        ]
        records.extend(
            ProbeFold(
                period=pf.period,
                fold=f.fold,
                auc=f.auc,
                n_train=f.n_train,
                n_test=f.n_test,
                n_test_positive=f.n_test_positive,
            )
            for f in fit.folds
        )
        self.logger.info(
            f"{pf.period}: AUC {fit.value:.4f} (n={len(y)}, label 1: {int(y.sum())}, folds {len(fit.folds)})"
        )
        return PeriodOutcome(value=fit.value, records=records + self.build_sample(pf))

    def summarize(self, outcomes: List[PeriodOutcome]) -> PeriodOutcome:
        """Log the mean AUC over the scored periods (NaN when none was scored); no records."""
        values = [o.value for o in outcomes if not np.isnan(o.value)]
        mean = float(np.mean(values)) if values else float("nan")
        self.logger.info(f"Mean {self.task_name} AUC: {mean:.4f} over {len(values)} period(s)")
        return PeriodOutcome(value=mean)
