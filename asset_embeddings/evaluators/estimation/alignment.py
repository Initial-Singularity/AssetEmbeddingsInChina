"""Orthogonal Procrustes alignment of one period's embedding onto another's."""

import numpy as np


def procrustes_rotation(X_src: np.ndarray, X_dst: np.ndarray) -> np.ndarray:
    """The orthogonal R minimizing ``||X_src @ R - X_dst||_F`` (rotation and reflection, no scaling).

    The rows of both matrices are the same firms in the same order; the caller centers them. R = U V' from the SVD
    X_src' X_dst = U S V' (Schönemann 1966).

    Args:
        X_src: Embeddings to rotate, ``(n, d)``.
        X_dst: Target embeddings, ``(n, d)``.

    Returns:
        The ``(d, d)`` orthogonal matrix R.

    Raises:
        ValueError: On mismatched shapes, fewer rows than columns, or a rank-deficient cross-product, where R is not
            determined.
    """
    X_src, X_dst = np.asarray(X_src, dtype=float), np.asarray(X_dst, dtype=float)
    if X_src.shape != X_dst.shape:
        raise ValueError(f"procrustes_rotation: shapes differ, {X_src.shape} vs {X_dst.shape}")
    n, d = X_src.shape
    if n < d:
        raise ValueError(f"procrustes_rotation: {n} common firms for dimension {d}; R is not determined")
    U, s, Vt = np.linalg.svd(X_src.T @ X_dst)
    # numpy's matrix_rank tolerance: singular values below it are numerically zero.
    rank = int(np.sum(s > s[0] * max(n, d) * np.finfo(float).eps)) if s[0] > 0 else 0
    if rank < d:
        raise ValueError(f"procrustes_rotation: cross-product rank {rank} < dimension {d} over {n} common firms")
    return U @ Vt
