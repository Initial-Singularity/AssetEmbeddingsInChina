"""Records of the grouped RV / RC evaluations: one result per (period, group) cell, its holdout splits, the tests."""

from .base import Detail, RecordField, Result
from .eval import _EVAL_CONFIG, _RV_CONFIG

_GROUPING_CONFIG = {
    "grouping": "$.grouping",
    "group_col": "$.group_col",
    "n_quantiles": "$.n_quantiles",
    "alpha_scope": "$.alpha_scope",
    "winsor_pct": "$.winsor_pct",
    "cv_alpha_min": "$.cv_alpha_min",
}


class GroupedResult(Result):
    """Score of one (period, group) cell with its bootstrap interval and the group's grouping values."""

    __key__ = ("period", "group_label")
    __summary_keys__ = ("group_label",)

    period: str = RecordField(required=True, doc="Period label")
    group_label: str = RecordField(
        required=True,
        doc="G1..Gn (quantile grouping, G1 = lowest value), a fixed-bin label or a category value (categorical)",
    )
    ci_lower: float = RecordField(doc="Bootstrap 95% interval, lower bound (NULL without the bootstrap)")
    ci_upper: float = RecordField(doc="Bootstrap 95% interval, upper bound")
    ridge_alpha: float = RecordField(doc="Penalty used")
    n_scored: int = RecordField(doc="Firms of the cell in the fit")
    n_features: int = RecordField(doc="Feature columns")
    n_group_firms: int = RecordField(doc="Firms of the group file in the group (the cell before the merge)")
    group_value_mean: float = RecordField(
        doc="Mean grouping value of the group (NULL under the categorical grouping, as are the three below)"
    )
    group_value_median: float = RecordField(doc="Median grouping value")
    group_value_min: float = RecordField(doc="Smallest grouping value")
    group_value_max: float = RecordField(doc="Largest grouping value")


class GroupedRVResult(GroupedResult):
    """Grouped RV cell."""

    __table__ = "eval_grouped_rv_results"
    __config_columns__ = {**_RV_CONFIG, **_GROUPING_CONFIG}


class GroupedRCResult(GroupedResult):
    """Grouped RC cell."""

    __table__ = "eval_grouped_rc_results"
    __config_columns__ = {**_EVAL_CONFIG, **_GROUPING_CONFIG}


class GroupedRidgeSplit(Detail):
    """One firm split of a grouped cell's holdout fit: the test R² of the split and its penalty.

    The grouped counterpart of ``RidgeSplit``, shared by grouped RV and RC (the run's task tells them apart).
    """

    __table__ = "eval_grouped_ridge_splits"
    __key__ = ("period", "group_label", "split")

    period: str = RecordField(required=True, doc="Period label")
    group_label: str = RecordField(required=True, doc="Group of the cell, as in the grouped result table")
    split: int = RecordField(required=True, doc="Split index; split s uses seed holdout_seed + s")
    value: float = RecordField(required=True, doc="Koijen test R² of the split")
    value_sklearn: float = RecordField(doc="sklearn-definition test R² (1 - SS_res / SS_tot) of the split")
    alpha: float = RecordField(doc="Penalty of the split")
    n_train: int = RecordField(doc="Training firms of the cell")
    n_test: int = RecordField(doc="Held-out firms of the cell")


class GroupedTest(Detail):
    """Two-sided permutation test of the difference of two groups' mean per-period scores."""

    __key__ = ("group_a", "group_b")

    group_a: str = RecordField(required=True, doc="First group")
    group_b: str = RecordField(required=True, doc="Second group")
    statistic: float = RecordField(required=True, doc="Absolute difference of the groups' mean per-period scores")
    p_value: float = RecordField(required=True, doc="Share of permutations reaching the observed statistic")
    n_a: int = RecordField(doc="Per-period scores of group_a")
    n_b: int = RecordField(doc="Per-period scores of group_b")


class GroupedRVTest(GroupedTest):
    """Grouped RV permutation test."""

    __table__ = "eval_grouped_rv_tests"


class GroupedRCTest(GroupedTest):
    """Grouped RC permutation test."""

    __table__ = "eval_grouped_rc_tests"
