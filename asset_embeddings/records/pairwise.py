"""Records of the pairwise similarity: the fit of each period, its coefficients, and their Fama-MacBeth aggregates."""

from .base import Detail, RecordField, Result

_PAIRWISE_CONFIG = {
    "model_type": "$.model_type",
    "label_col": "$.label_col",
    "match_on": "$.match_on",
    "interaction": "$.interaction",
    "seed": "$.seed",
}


class PairwiseFit(Detail):
    """The regression of one period: its fit and sample (one row per period)."""

    __table__ = "eval_pairwise_fits"
    __key__ = ("period",)
    __config_columns__ = _PAIRWISE_CONFIG

    period: str = RecordField(required=True, doc="Period label")
    r2: float = RecordField(doc="R² of the pair-level regression")
    n_pairs: int = RecordField(doc="Unordered firm pairs")
    mean_sim: float = RecordField(doc="Mean cosine similarity over the pairs")
    n_firms: int = RecordField(doc="Firms with an embedding, a label and every match_on value")


class PairwiseCoef(Detail):
    """One coefficient of one period's regression."""

    __table__ = "eval_pairwise_coefs"
    __key__ = ("period", "term")
    __config_columns__ = _PAIRWISE_CONFIG

    period: str = RecordField(required=True, doc="Period label")
    term: str = RecordField(required=True, doc="'const', a match_on column, or an interaction 'a:b'")
    coef: float = RecordField(required=True, doc="Coefficient of the term")


class PairwiseResult(Result):
    """The Fama-MacBeth aggregate of one term (metric ``fm_coef``, value the time-series mean of its coefficient)."""

    __table__ = "eval_pairwise_results"
    __key__ = ("term",)
    __summary_keys__ = ("term",)
    __config_columns__ = _PAIRWISE_CONFIG

    term: str = RecordField(required=True, doc="'const', a match_on column, or an interaction 'a:b'")
    se: float = RecordField(doc="Standard error: sample std of the per-period coefficients over sqrt(n_periods)")
    t: float = RecordField(doc="mean / se (NULL when se is 0 or undefined)")
    p: float = RecordField(doc="Two-sided p, t distribution with n_periods - 1 degrees of freedom (NULL when t is)")
    n_periods: int = RecordField(doc="Periods with a coefficient")
