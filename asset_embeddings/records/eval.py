"""Evaluation records: the RV / RC / MP results, the sample funnel and the holdout splits.

What a run was asked to do stays in ``runs.config_content``; the views expose the declared config columns next to
every row. A result row holds the estimate, the penalty actually used and the size of the sample it was fitted on.
"""

from typing import Dict

from .base import Detail, RecordField, Result

#: The steps of the sample funnel, in the order they can occur. File steps give the rows of the file read; the other
#: steps give the rows left after the step, so a drop is the difference to the step before.
SAMPLE_STEPS = (
    "embedding_file",
    "benchmark_file",
    "joined",
    "char_file",
    "char_joined",
    "na_dropped",
    "nonpositive_equity_dropped",
    "group_file",
    "label_file",
    "label_joined",
)

_EVAL_CONFIG: Dict[str, str] = {
    "model_type": "$.model_type",
    "features": "$.features",
    "align_chars": "$.align_chars",
    "scoring": "$.scoring",
    "holdout_split_aggregate": "$.holdout_split_aggregate",
    "fixed_alpha": "$.ridge_alpha",
    "normalization": "$.normalization",
    "seed": "$.seed",
}
_RV_CONFIG = {**_EVAL_CONFIG, "nonpositive_equity": "$.nonpositive_equity"}
# RC and RC_temporal share the RC table; slope_window is NULL for RC and for temporal runs written before the field.
_RC_CONFIG = {**_EVAL_CONFIG, "slope_window": "$.slope_window"}


class RVResult(Result):
    """Relative valuation: the ridge R² of the valuation residual on the features, one row per period."""

    __table__ = "eval_rv_results"
    __key__ = ("period",)
    __config_columns__ = _RV_CONFIG

    period: str = RecordField(required=True, doc="Period label, e.g. 2023Q2")
    ridge_alpha: float = RecordField(doc="Penalty used (under holdout scoring: the median over the splits)")
    n_scored: int = RecordField(doc="Firms in the fit: the last funnel step (every firm of the period under holdout)")
    n_features: int = RecordField(doc="Feature columns: embedding, placebo, characteristics and dummy columns")


class RCResult(Result):
    """Return comovement: the ridge R² of next-quarter returns on the features, one row per period."""

    __table__ = "eval_rc_results"
    __key__ = ("period",)
    __config_columns__ = _RC_CONFIG

    period: str = RecordField(required=True, doc="Period label, e.g. 2023Q2")
    ridge_alpha: float = RecordField(doc="Penalty used (under holdout scoring: the median over the splits)")
    n_scored: int = RecordField(doc="Firms in the fit: the last funnel step (every firm of the period under holdout)")
    n_features: int = RecordField(doc="Feature columns: embedding, placebo, characteristics and dummy columns")


class MPResult(Result):
    """Managed portfolios: the refined model's score on the test split, with the score before refining."""

    __table__ = "eval_mp_results"
    __key__ = ("period",)
    __config_columns__ = {
        "model_type": "$.model.embedding_model",
        "head": "$.model.model_type",
        "freeze_embedding": "$.model.freeze_embedding",
        "seed": "$.train.seed",
    }

    period: str = RecordField(required=True, doc="Period of the holdings, e.g. 2024Q2")
    value_std: float = RecordField(doc="Standard deviation of the per-portfolio scores")
    baseline_value: float = RecordField(doc="Score of the loaded model before refining")
    baseline_value_std: float = RecordField(doc="Standard deviation of the baseline per-portfolio scores")
    n_features: int = RecordField(doc="Embedding dimension of the model")


class SampleStep(Detail):
    """One step of a period's sample funnel: a file read, a join or a drop, with its row count."""

    __table__ = "eval_sample"
    __key__ = ("period", "step_no")
    __table_constraints__ = (f"CHECK (step IN ({', '.join(repr(s) for s in SAMPLE_STEPS)}))",)

    period: str = RecordField(required=True, doc="Period label")
    step_no: int = RecordField(required=True, doc="Position of the step in the period's funnel, from 1")
    step: str = RecordField(required=True, doc="One of SAMPLE_STEPS")
    n_rows: int = RecordField(required=True, doc="Rows of the file (file steps) or rows left after the step")


class RidgeSplit(Detail):
    """One firm split of a holdout fit: the test R² of the split and its penalty."""

    __table__ = "eval_ridge_splits"
    __key__ = ("period", "split")

    period: str = RecordField(required=True, doc="Period label")
    split: int = RecordField(required=True, doc="Split index; split s uses seed holdout_seed + s")
    value: float = RecordField(required=True, doc="Koijen test R² of the split")
    value_sklearn: float = RecordField(doc="sklearn-definition test R² (1 - SS_res / SS_tot) of the split")
    alpha: float = RecordField(doc="Penalty of the split")
    n_train: int = RecordField(doc="Training firms")
    n_test: int = RecordField(doc="Held-out firms")
