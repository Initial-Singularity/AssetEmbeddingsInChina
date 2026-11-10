"""Records of the label probe: one AUC per period and one row per scored fold."""

from .base import Detail, RecordField, Result

_PROBE_CONFIG = {
    "model_type": "$.model_type",
    "features": "$.features",
    "align_chars": "$.align_chars",
    "normalization": "$.normalization",
    "seed": "$.seed",
    "label_col": "$.label_col",
    "n_folds": "$.n_folds",
}


class ProbeResult(Result):
    """The probe of one period: the mean test-fold AUC (metric ``auc``) of the label on the features."""

    __table__ = "eval_probe_results"
    __key__ = ("period",)
    __config_columns__ = _PROBE_CONFIG

    period: str = RecordField(required=True, doc="Period label")
    n_scored: int = RecordField(doc="Firms of the period with a label: the last funnel step")
    n_positive: int = RecordField(doc="Firms with label 1")
    n_features: int = RecordField(doc="Feature columns")
    n_folds_used: int = RecordField(
        doc="Folds scored (a fold whose test or training part holds a single class is skipped)"
    )


class ProbeFold(Detail):
    """One scored fold of a period's probe."""

    __table__ = "eval_probe_folds"
    __key__ = ("period", "fold")

    period: str = RecordField(required=True, doc="Period label")
    fold: int = RecordField(required=True, doc="Fold index in the split's order, from 0")
    auc: float = RecordField(required=True, doc="ROC AUC on the fold's test firms")
    n_train: int = RecordField(doc="Training firms")
    n_test: int = RecordField(doc="Test firms")
    n_test_positive: int = RecordField(doc="Test firms with label 1")
