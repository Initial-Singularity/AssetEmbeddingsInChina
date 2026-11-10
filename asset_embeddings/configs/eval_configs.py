"""Configs of the evaluations (RV, RC and their grouped variants), built from field groups.

`EvalDataConfig` names the periods and the per-period files, `FeatureConfig` the feature columns, `RidgeConfig`
the penalty and the scoring, `GroupingConfig` how firms are grouped, `LabelConfig` a per-period firm label. Each
evaluation config is one combination of them plus its own fields. Every field is documented by its
`ConfigField(doc=...)` string.

A per-period path is a list with one file per entry of `periods`, in the same order, or a single file when there
is one period. Files are never matched to periods by name: which period a file belongs to differs between paths
(the predictive setups read the embedding of period t next to the benchmark of t+1), so only the config can say.
"""

import re
from typing import List, Literal, Optional, Union

from .base import Config, ConfigField
from .constraints import (
    ChoiceConstraint,
    ChoicesConstraint,
    IntersectionConstraints,
    LambdaConstraint,
    LambdaCrossConstraint,
    LengthConstraint,
    OptionalConstraint,
    RangeConstraint,
    TypeConstraint,
    UnionConstraints,
)
from .object_configs import path_preprocessor

#: The feature blocks an evaluation can combine.
FEATURES = ("embedding", "placebo", "chars", "industry")

_PATH = UnionConstraints(TypeConstraint(str), TypeConstraint(list))


def per_period(name: str) -> LambdaCrossConstraint:
    """The cross constraint that path field ``name`` names one file per period (None passes).

    Args:
        name: The path field.

    Returns:
        A constraint named ``<name>_per_period``.
    """

    def check(cfg) -> bool:
        value = getattr(cfg, name)
        if value is None:
            return True
        n_periods = len(cfg.periods)
        return len(value) == n_periods if isinstance(value, list) else n_periods == 1

    return LambdaCrossConstraint(
        check,
        error=f"{name} must be a list with one file per period, or a single file when there is one period",
        name=f"{name}_per_period",
    )


def _is_feature_list(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and len(set(value)) == len(value)
        and all(isinstance(f, str) and f in FEATURES for f in value)
    )


class EvalDataConfig(Config):
    """The periods, the per-period embedding and benchmark files, and the columns read from them."""

    model_type: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="Label of the model under evaluation, recorded with every result, e.g. 'AssetBERT', 'AssetW2V', 'Chars'.",
        required=True,
    )
    periods: List[str] = ConfigField(
        default=None,
        constraint=LambdaConstraint(
            lambda v: isinstance(v, list) and len(v) > 0 and all(isinstance(q, str) for q in v),
            "periods must be a non-empty list of period labels",
        ),
        doc=("Period labels, in the order of the per-period path lists; each label is recorded with its results."),
        required=True,
    )
    embedding_path: Optional[Union[str, List[str]]] = ConfigField(
        default=None,
        constraint=OptionalConstraint(_PATH),
        doc="Per-period embedding files. Set exactly when 'embedding' is in features.",
    )
    benchmark_path: Union[str, List[str]] = ConfigField(
        default=None,
        constraint=_PATH,
        doc=(
            "Per-period benchmark files: the target columns, and the characteristics and IndustryCode unless "
            "char_benchmark_path is set."
        ),
        required=True,
    )
    embedding_token_col: str = ConfigField(
        default="Token",
        constraint=TypeConstraint(str),
        doc="Name of the token (stock ID) column in the embedding files.",
    )
    embedding_value_pattern: str = ConfigField(
        default="Embed*",
        constraint=TypeConstraint(str),
        doc="Wildcard pattern of the embedding value columns, e.g. 'Embed*', 'Emb_*' or 'Feature_*'.",
    )
    benchmark_stock_col: str = ConfigField(
        default="Stkcd", constraint=TypeConstraint(str), doc="Name of the stock ID column in the benchmark files."
    )
    seed: int = ConfigField(
        default=42,
        constraint=TypeConstraint(int),
        doc="Seed of every random draw: the placebo columns (one generator per period, from the seed and the period label), the cross-validation folds of the penalty search, the bootstrap and the permutations.",
    )

    _cross_constraints = [per_period("embedding_path"), per_period("benchmark_path")]


class FeatureConfig(Config):
    """Which columns are the features, which firms enter, and how the features are scaled."""

    features: List[str] = ConfigField(
        default=["embedding"],
        constraint=LambdaConstraint(
            _is_feature_list, f"features must be a non-empty list of distinct entries of {list(FEATURES)}"
        ),
        doc=(
            "Feature blocks, in column order; any non-empty combination of 'embedding' (the columns matching "
            "embedding_value_pattern), 'placebo' (four columns drawn uniformly on [-1, 1] per firm), 'chars' (the "
            "benchmark's characteristics) and 'industry' (IndustryCode dummies, first level dropped). Without "
            "'embedding' the evaluation runs on the firms of the benchmark file."
        ),
    )
    align_chars: bool = ConfigField(
        default=True,
        constraint=TypeConstraint(bool),
        doc=(
            "Read the characteristics even when 'chars' is not a feature, so that firms missing any of them are "
            "dropped and the sample matches the one with 'chars' added. With 'chars' in features the firms are "
            "restricted either way. False: every firm with the other inputs enters."
        ),
    )
    char_benchmark_path: Optional[Union[str, List[str]]] = ConfigField(
        default=None,
        constraint=OptionalConstraint(_PATH),
        doc=(
            "Per-period files that supply the characteristics and IndustryCode instead of benchmark_path, joined "
            "on the stock ID (the predictive setups read the period-t characteristics next to the t+1 target). "
            "None: they come from benchmark_path."
        ),
    )
    normalization: Literal["standard", "minmax", "none"] = ConfigField(
        default="standard",
        constraint=ChoiceConstraint(["standard", "minmax", "none"]),
        doc="Feature scaling per period: 'standard' (StandardScaler), 'minmax' (MinMaxScaler) or 'none'.",
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: (cfg.embedding_path is not None) == ("embedding" in cfg.features),
            error="embedding_path must be set when 'embedding' is in features, and only then",
            name="embedding_path_matches_features",
        ),
        per_period("char_benchmark_path"),
    ]


class RidgeConfig(Config):
    """The ridge penalty and how the fit is scored."""

    ridge_alpha: Optional[float] = ConfigField(
        default=0.5,
        constraint=OptionalConstraint(RangeConstraint(0, None)),
        doc="Ridge penalty; None selects it per fit by 10-fold cross-validation over a fixed grid.",
    )
    scoring: Literal["cross_section", "holdout"] = ConfigField(
        default="cross_section",
        constraint=ChoiceConstraint(["cross_section", "holdout"]),
        doc=(
            "'cross_section': fit and score on every firm of the period (the setting of the main results). "
            "'holdout': fit on the training firms of holdout_n_splits random firm splits and score R² on the "
            "held-out firms, aggregated over the splits by holdout_split_aggregate; with ridge_alpha None the "
            "penalty is cross-validated inside each training set."
        ),
    )
    holdout_test_size: float = ConfigField(
        default=0.2,
        constraint=LambdaConstraint(
            lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < v <= 0.5,
            "holdout_test_size must be a number in (0, 0.5]",
        ),
        doc=(
            "Held-out share of firms per split under holdout scoring; realized as round(1 / holdout_test_size) "
            "hash folds with fold 0 held out."
        ),
    )
    holdout_n_splits: int = ConfigField(
        default=20,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Number of firm splits per period under holdout scoring; split s uses seed holdout_seed + s.",
    )
    holdout_seed: int = ConfigField(
        default=42,
        constraint=TypeConstraint(int),
        doc="Base seed of the firm splits under holdout scoring, independent of seed.",
    )
    holdout_split_aggregate: Literal["median", "mean"] = ConfigField(
        default="median",
        constraint=ChoiceConstraint(["median", "mean"]),
        doc=(
            "How a period's value is aggregated over its firm splits under holdout scoring. The median resists the "
            "occasional split whose held-out R² is driven far below zero by one outlying test firm. Aggregation "
            "across periods is always the mean."
        ),
    )


class GroupingConfig(Config):
    """How firms are grouped by a per-period value, and the interval and tests computed per group.

    The grouped evaluations read the characteristics from benchmark_path, so they refuse char_benchmark_path. Under
    holdout scoring each cell's value is the held-out R² aggregated over the firm splits (holdout_split_aggregate)
    and its interval a bootstrap over the cell's firms of that aggregate.
    """

    group_path: Union[str, List[str]] = ConfigField(
        default=None,
        constraint=_PATH,
        doc="Per-period group files: the stock ID and the value firms are grouped by.",
        required=True,
    )
    group_col: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="Column of the group file that defines the groups, e.g. 'coverage_count' or 'SOE'.",
        required=True,
    )
    group_stock_col: str = ConfigField(
        default="Stkcd",
        constraint=TypeConstraint(str),
        doc="Stock ID column of the group file (the coverage files name it 'Symbol').",
    )
    grouping: Literal["quantile", "fixed", "categorical"] = ConfigField(
        default="quantile",
        constraint=ChoiceConstraint(["quantile", "fixed", "categorical"]),
        doc=(
            "'quantile': n_quantiles groups by the per-period quantiles of the value. 'fixed': the groups of bins. "
            "'categorical': one group per distinct value (an integer-valued number labeled without its decimal, "
            "the groups ordered by value)."
        ),
    )
    n_quantiles: int = ConfigField(
        default=4,
        constraint=RangeConstraint(2, 20),
        doc="Number of quantile groups under grouping 'quantile'.",
    )
    bins: Optional[list] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(list)),
        doc=(
            "Inner cut points under grouping 'fixed' (finite, strictly increasing); [1, 3, 10] gives the groups "
            "<=1, 2-3, 4-10 and >10."
        ),
    )
    bin_labels: Optional[list] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(list)),
        doc="Labels of the fixed groups, one more than bins; None derives them from bins.",
    )
    n_bootstrap: int = ConfigField(
        default=1000,
        constraint=RangeConstraint(0, 100000),
        doc="Bootstrap resamples of the per-group interval; 0 skips the bootstrap and records a null interval.",
    )
    n_permutation: int = ConfigField(
        default=1000,
        constraint=RangeConstraint(100, 100000),
        doc="Permutations of each between-group test.",
    )
    winsor_pct: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 25.0)),
        doc=(
            "Two-sided winsorization percentile of the target within a group (the valuation residual for RV, the "
            "next-quarter return for RC): clipped at [p, 100-p] before the ridge, which keeps heavy tails from "
            "saturating the cross-validated penalty. None: no winsorization."
        ),
    )
    cv_alpha_min: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1e8)),
        doc=(
            "Floor of the cross-validated penalty: only grid points at or above it are searched, so the "
            "well-covered groups are not under-regularized. None: the full grid."
        ),
    )
    alpha_scope: Literal["cell", "cross_section"] = ConfigField(
        default="cell",
        constraint=ChoiceConstraint(["cell", "cross_section"]),
        doc=(
            "Where the cross-validated penalty is chosen: 'cell' within each (group, period) cell, 'cross_section' once "
            "on the period's full cross section and applied to every group (for the characteristics baseline, "
            "whose per-group samples are too small for a stable search). No effect with a fixed ridge_alpha."
        ),
    )

    _cross_constraints = [
        per_period("group_path"),
        LambdaCrossConstraint(
            lambda cfg: cfg.grouping != "fixed"
            or (
                isinstance(cfg.bins, list)
                and len(cfg.bins) >= 1
                and all(cfg.bins[i] < cfg.bins[i + 1] for i in range(len(cfg.bins) - 1))
            ),
            error="grouping 'fixed' requires bins to be a non-empty, strictly increasing list of inner cut points",
            name="fixed_requires_bins",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.grouping != "fixed" or cfg.bin_labels is None or len(cfg.bin_labels) == len(cfg.bins) + 1,
            error="bin_labels must have one entry more than bins",
            name="bin_labels_match_bins",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.grouping != "categorical" or (cfg.bins is None and cfg.bin_labels is None),
            error="grouping 'categorical' takes its groups from the values; unset bins and bin_labels",
            name="categorical_takes_no_bins",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.char_benchmark_path is None,
            error="the grouped evaluation reads the characteristics from benchmark_path; unset char_benchmark_path",
            name="grouped_no_char_benchmark_path",
        ),
    ]


class LabelConfig(Config):
    """A per-period firm label (such as the state-ownership flag), joined onto the frame; unlabeled firms leave."""

    label_path: Union[str, List[str]] = ConfigField(
        default=None,
        constraint=_PATH,
        doc="Per-period label files: the stock ID and the label column, one row per stock.",
        required=True,
    )
    label_col: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="The label column of the label files, e.g. 'SOE'.",
        required=True,
    )
    label_stock_col: str = ConfigField(
        default="Stkcd",
        constraint=TypeConstraint(str),
        doc="Stock ID column of the label files.",
    )

    _cross_constraints = [per_period("label_path")]


class RVEvalConfig(EvalDataConfig, FeatureConfig, RidgeConfig):
    """Config of the Relative Valuation (RV) evaluation: the valuation residual regressed on the features."""

    nonpositive_equity: Literal["drop", "keep"] = ConfigField(
        default="drop",
        constraint=ChoiceConstraint(["drop", "keep"]),
        doc=(
            "Firms with non-positive book or market equity. 'drop': removed from the period before the "
            "log(be) ~ log(me) residual and the ridge. 'keep': the earlier behavior, retained so results recorded "
            "under it can be reproduced: the raw value enters the OLS next to the logs of the other firms, and "
            "such a firm's residual is of order 1e9 and dominates any fit that touches it."
        ),
    )


class RCEvalConfig(EvalDataConfig, FeatureConfig, RidgeConfig):
    """Config of the Return Comovement (RC) evaluation: next-quarter returns regressed on the features."""


def _consecutive_quarters(periods) -> bool:
    """Whether ``periods`` are ``YYYYQn`` labels of consecutive quarters in increasing order."""
    if not isinstance(periods, list) or not all(
        isinstance(p, str) and re.fullmatch(r"\d{4}Q[1-4]", p) for p in periods
    ):
        return False
    index = [int(p[:4]) * 4 + int(p[-1]) for p in periods]
    return all(b - a == 1 for a, b in zip(index, index[1:]))


class TemporalRCEvalConfig(RCEvalConfig):
    """Config of the temporal RC evaluation: RC slopes fitted on earlier periods, applied to a later one."""

    align: Literal["none", "procrustes"] = ConfigField(
        default="none",
        constraint=ChoiceConstraint(["none", "procrustes"]),
        doc=(
            "'none': period T's embedding enters in its own coordinates. 'procrustes': period T's raw embedding is "
            "rotated onto period T-1's by the orthogonal Procrustes rotation fitted on the firms present in both "
            "(centered on them), before each period is normalized. Under 'expanding' the rotation is fitted onto "
            "each earlier period whose slope is applied."
        ),
    )
    slope_window: Literal["previous", "expanding"] = ConfigField(
        default="previous",
        constraint=ChoiceConstraint(["previous", "expanding"]),
        doc=(
            "'previous': period T is scored with the slope fitted on period T-1. 'expanding': with the mean of the "
            "predictions of the slopes fitted on every earlier period of the config, each with its own penalty; in "
            "a fixed frame this is the prediction of the mean slope. The first scored period has one earlier slope, "
            "so both choices agree there."
        ),
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.scoring == "cross_section",
            error="temporal RC fits on one period and scores on the next; scoring must be 'cross_section'",
            name="temporal_cross_section",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.features in (["embedding"], ["chars"]),
            error=(
                "temporal RC carries the slope across periods, so features must be ['embedding'] or ['chars'] "
                "(industry dummies can differ between periods and placebo columns are drawn afresh every period)"
            ),
            name="temporal_features",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.align == "none" or cfg.features == ["embedding"],
            error="align 'procrustes' rotates embeddings; it requires features ['embedding']",
            name="temporal_align_embedding",
        ),
        LambdaCrossConstraint(
            lambda cfg: _consecutive_quarters(cfg.periods),
            error=(
                "temporal RC fits each period on the one before, so periods must be consecutive quarters 'YYYYQn' "
                "in increasing order (otherwise the previous period's target overlaps the next one's)"
            ),
            name="temporal_consecutive_periods",
        ),
    ]


class LabelProbeEvalConfig(EvalDataConfig, FeatureConfig, LabelConfig):
    """Config of the label probe: a binary firm label recovered from the features by a cross-validated logit."""

    n_folds: int = ConfigField(
        default=5,
        constraint=TypeConstraint(int) & RangeConstraint(2, 20),
        doc="Folds of the stratified split; each fold's test AUC is scored and the period's value is their mean.",
    )


def _is_match_on(value) -> bool:
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 4
        and len(set(value)) == len(value)
        and all(isinstance(c, str) for c in value)
    )


class PairwiseSimilarityEvalConfig(EvalDataConfig, LabelConfig):
    """Config of the pairwise similarity: embedding cosine similarity of firm pairs on same-value indicators.

    The regression runs on the raw embedding vectors of every firm with an embedding, a label and a value in each
    ``match_on`` column, so there are no feature fields.
    """

    match_on: List[str] = ConfigField(
        default=None,
        constraint=LambdaConstraint(_is_match_on, "match_on must list 1 to 4 distinct column names"),
        doc=(
            "Columns whose shared value makes a same-value indicator of a firm pair: the label column (label_col) "
            "or benchmark columns such as 'IndustryCode'; 1 to 4 distinct names."
        ),
        required=True,
    )
    interaction: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="Add the product of every subset of two or more match_on indicators as a regressor.",
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.embedding_path is not None,
            error="pairwise similarity regresses the embedding similarity; set embedding_path",
            name="pairwise_requires_embedding_path",
        )
    ]


class GroupedRVEvalConfig(GroupingConfig, RVEvalConfig):
    """Config of the grouped RV evaluation: RV per group of firms."""


class GroupedRCEvalConfig(GroupingConfig, RCEvalConfig):
    """Config of the grouped RC evaluation: RC per group of firms."""
