"""Panel aggregation of per-period estimates (Fama-MacBeth)."""

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def fama_macbeth(values: Sequence[float]) -> Dict[str, float]:
    """Aggregate per-period estimates: time-series mean, its standard error, t and two-sided p.

    Args:
        values: One estimate per period; NaN entries are skipped.

    Returns:
        ``mean``, ``se`` (std with ddof 1 over sqrt(T)), ``t`` (mean / se), ``p`` (two-sided, t distribution with
        T - 1 degrees of freedom) and ``n`` (T). ``se``, ``t`` and ``p`` are NaN when T < 2; ``t`` and ``p`` are
        NaN when ``se`` is 0.
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    T = arr.size
    nan = float("nan")
    mean = float(arr.mean()) if T else nan
    se = float(arr.std(ddof=1) / np.sqrt(T)) if T > 1 else nan
    if T > 1 and se > 0:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), T - 1))
    else:
        t = p = nan
    return {"mean": mean, "se": se, "t": t, "p": p, "n": T}
