"""Resampling statistics: a bootstrap interval around a frame-level value, and a permutation test of two means."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    """A two-sided percentile interval."""

    lower: float
    upper: float


def bootstrap_interval(
    value_of: Callable[[pd.DataFrame], Optional[float]],
    frame: pd.DataFrame,
    n: int,
    seed: int,
    level: float = 0.95,
) -> Optional[Interval]:
    """Percentile interval of ``value_of`` over ``n`` resamples of the rows of ``frame`` with replacement.

    The resamples come from one ``RandomState(seed)`` per call, each a ``frame.sample(n=len(frame), replace=True)``
    in sequence, so the same frame and seed give the same interval. A resample whose value is None is left out; the
    resamples after it are the same either way.

    Args:
        value_of: Computes the value of one (resampled) frame, or None when that resample cannot be scored.
        frame: The rows to resample.
        n: Number of resamples; 0 skips the interval.
        seed: Seed of the resampling RNG.
        level: Coverage of the interval.

    Returns:
        The interval, or None when ``n`` is 0 or no resample has a value.
    """
    if n == 0:
        return None
    rng = np.random.RandomState(seed)
    values = [value_of(frame.sample(n=len(frame), replace=True, random_state=rng)) for _ in range(n)]
    values = [v for v in values if v is not None]
    if not values:
        return None
    # 100 * level first keeps the default percentiles at exactly 2.5 and 97.5.
    tail = (100.0 - 100.0 * level) / 2.0
    return Interval(float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail)))


def permutation_test(a: Sequence[float], b: Sequence[float], n: int, seed: int) -> Tuple[float, float]:
    """Two-sided permutation test for a difference in means.

    The pooled values are shuffled in place ``n`` times by one ``RandomState(seed)``; the p-value is the share of
    shuffles whose absolute difference of group means reaches the observed one.

    Args:
        a: Values of the first group.
        b: Values of the second group.
        n: Number of permutations.
        seed: Seed of the permutation RNG.

    Returns:
        ``(statistic, p_value)``: the observed absolute difference of means and its p-value.
    """
    observed = abs(np.mean(a) - np.mean(b))
    pooled = np.array(list(a) + list(b))
    n_a = len(a)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])) >= observed:
            count += 1
    return float(observed), count / n
