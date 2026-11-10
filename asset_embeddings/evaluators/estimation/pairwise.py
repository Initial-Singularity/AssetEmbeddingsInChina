"""The pairwise-similarity regression: cosine similarity of every firm pair on same-value indicators.

Every regressor is a function of the pair's indicator pattern, so the OLS is solved from the count, sum and sum of
squares of the similarity in each of the ``2**k`` patterns; the pair-level design matrix is never built.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Cells = Dict[Tuple[bool, ...], Tuple[int, float, float]]


def term_names(match_on: Sequence[str], interaction: bool) -> List[str]:
    """Names of the regression terms, in coefficient order.

    Args:
        match_on: The matched columns.
        interaction: Whether the products of two or more indicators are included.

    Returns:
        ``["const", *match_on]``, followed under ``interaction`` by ``"a:b"``, ``"a:b:c"``, ... for every subset
        of size two or more, in ``itertools.combinations`` order.
    """
    names = ["const", *match_on]
    if interaction:
        for size in range(2, len(match_on) + 1):
            names += [":".join(combo) for combo in itertools.combinations(match_on, size)]
    return names


def cell_statistics(E: np.ndarray, keys: np.ndarray) -> Cells:
    """Sufficient statistics of the pairwise cosine similarity per indicator pattern.

    Args:
        E: Embedding matrix, one row per firm. Rows are scaled to unit norm; a zero row is left as is.
        keys: One column per matched variable, one row per firm; indicator j of a pair is 1 when the two firms'
            values in column j are equal.

    Returns:
        For every pattern of the ``k`` indicators (a tuple of bools, all ``2**k`` present), the pair count, the
        sum of the similarities and the sum of their squares, over the unordered pairs (the upper triangle).
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = E / norms
    iu = np.triu_indices(len(E), k=1)
    S = unit @ unit.T
    sim = S[iu]
    del S  # n x n; the pair vectors below are of the same order of size
    k = keys.shape[1]
    code = np.zeros(sim.shape, dtype=np.int64)
    for j in range(k):
        code |= (keys[iu[0], j] == keys[iu[1], j]).astype(np.int64) << j
    size = 2**k
    counts = np.bincount(code, minlength=size)
    sums = np.bincount(code, weights=sim, minlength=size)
    squares = np.bincount(code, weights=sim * sim, minlength=size)
    return {
        tuple(bool((c >> j) & 1) for j in range(k)): (int(counts[c]), float(sums[c]), float(squares[c]))
        for c in range(size)
    }


def _term_value(term: str, pattern: Tuple[bool, ...], position: Dict[str, int]) -> float:
    """Value of one regressor in one indicator pattern: 1 for the constant, else the product of its indicators."""
    if term == "const":
        return 1.0
    return float(all(pattern[position[name]] for name in term.split(":")))


def ols_from_cells(
    cells: Cells, terms: Sequence[str], match_on: Sequence[str], period: Optional[str] = None
) -> Tuple[np.ndarray, float, int, float]:
    """Solve the pair-level OLS from the per-pattern statistics.

    Args:
        cells: Output of ``cell_statistics``.
        terms: Regression terms from ``term_names``.
        match_on: The matched columns, in the order of the pattern tuples.
        period: Period label, for the error message.

    Returns:
        ``(coefficients in terms order, R², pair count, mean similarity)``.

    Raises:
        ValueError: When the design is singular, e.g. every firm shares the label so that no pair differs on it.
    """
    position = {name: j for j, name in enumerate(match_on)}
    p = len(terms)
    XtX = np.zeros((p, p))
    Xty = np.zeros(p)
    N = S = Q = 0.0
    for pattern, (n, s, q) in cells.items():
        if n == 0:
            continue
        x = np.array([_term_value(t, pattern, position) for t in terms])
        XtX += n * np.outer(x, x)
        Xty += s * x
        N += n
        S += s
        Q += q
    if np.linalg.matrix_rank(XtX) < p:
        empty = [dict(zip(match_on, pattern)) for pattern, (n, _, _) in cells.items() if n == 0]
        raise ValueError(
            f"Pairwise design is singular in period {period}: indicator patterns without pairs {empty}; "
            f"a matched column may hold a single value"
        )
    beta = np.linalg.solve(XtX, Xty)
    rss = Q - beta @ Xty
    tss = Q - S * S / N
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    return beta, r2, int(N), S / N
