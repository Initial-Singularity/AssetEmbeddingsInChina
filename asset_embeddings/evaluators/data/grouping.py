"""Assign firms to groups by a value of the group file: quantile groups, fixed bins or the value's categories."""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def auto_bin_labels(cuts: Sequence[float]) -> List[str]:
    """Readable labels for fixed bins from their inner cut points.

    ``[1, 3, 10]`` gives ``["1", "2-3", "4-10", ">10"]`` (right-closed bins of integer values).
    """
    labels = [str(int(cuts[0])) if cuts[0] == int(cuts[0]) else f"<={cuts[0]}"]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        lo_disp = int(lo) + 1 if lo == int(lo) else lo
        hi_disp = int(hi) if hi == int(hi) else hi
        labels.append(f"{lo_disp}-{hi_disp}")
    labels.append(f">{int(cuts[-1]) if cuts[-1] == int(cuts[-1]) else cuts[-1]}")
    return labels


def category_label(value: Any) -> str:
    """The label of one categorical value: an integer-valued number prints without a decimal (``1.0`` gives ``"1"``)."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def category_sort_key(label: str) -> Tuple[int, Any]:
    """Order categorical labels by their value: labels that read as numbers first, numerically, then the others."""
    try:
        return (0, float(label))
    except ValueError:
        return (1, label)


def assign_groups(
    values: pd.Series,
    mode: str,
    n_quantiles: Optional[int] = None,
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], pd.Series]:
    """Assign each value a group label.

    Args:
        values: The grouping value per firm (``coverage_count``).
        mode: ``"quantile"``: ``pd.qcut`` into ``n_quantiles`` groups labeled ``G1..Gn`` (``G1`` lowest; the ``G``
            keeps the labels apart from period labels such as ``2023Q2``). ``"fixed"``: bins with edges
            ``[-inf, *bins, inf]``, right-closed, labeled ``labels`` or the automatic labels of ``bins``.
            ``"categorical"``: one group per distinct value, labeled by ``category_label`` and ordered by
            ``category_sort_key``.
        n_quantiles: Number of quantile groups.
        bins: Inner cut points of the fixed bins.
        labels: Explicit fixed-bin labels (one more than ``bins``).

    Returns:
        ``(labels, groups)``: every label in order (including groups that end up empty) and the categorical
        group of each value (NaN where the value is NaN).

    Raises:
        ValueError: When the values have fewer distinct values than quantiles, when fixed mode has no bins, or on
            an unknown mode.
    """
    if mode == "quantile":
        names = [f"G{i + 1}" for i in range(n_quantiles)]
        try:
            groups = pd.qcut(values, q=n_quantiles, labels=names, duplicates="drop")
        except ValueError as e:
            raise ValueError(
                f"Cannot form {n_quantiles} quantiles: only {values.nunique()} unique "
                f"coverage_count values. Reduce n_quantiles or check coverage data quality."
            ) from e
        return names, groups
    if mode == "fixed":
        if not bins:
            raise ValueError("fixed grouping requires non-empty bins")
        names = list(labels) if labels is not None else auto_bin_labels(bins)
        groups = pd.cut(values, bins=[-np.inf, *bins, np.inf], labels=names, right=True)
        return names, groups
    if mode == "categorical":
        groups = values.map(lambda v: None if pd.isna(v) else category_label(v))
        names = sorted(groups.dropna().unique(), key=category_sort_key)
        return names, groups
    raise ValueError(f"Unknown binning mode: {mode!r}")
