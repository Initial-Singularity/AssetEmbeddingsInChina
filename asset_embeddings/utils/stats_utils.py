"""Streaming mean/variance and min/max trackers used by the trainers to summarize losses.

Also the early-stop tracker that decides when a training loop ends.
"""

import math
from typing import Optional, Self
import numpy as np


class MeanVarianceTracker:
    """Running mean and variance of a stream of values by Welford's algorithm; NaN values are ignored.

    Usable as a context manager: entering the `with` block resets the tracker.
    """

    def __init__(self):
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from the running mean (Welford's M2)

    def update(self, x: float) -> None:
        """Fold one value in; a NaN is skipped and leaves the statistics untouched.

        Args:
            x: The new value.
        """
        if not np.isnan(x):
            self._n += 1

            delta = x - self._mean
            self._mean += delta / self._n

            delta2 = x - self._mean
            self._m2 += delta * delta2

    def clear(self) -> None:
        """Forget every value seen so far."""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def count(self) -> int:
        """Number of non-NaN values folded in."""
        return self._n

    @property
    def sum(self) -> float:
        """Sum of the values folded in, recovered as mean times count."""
        return self._mean * self._n

    def var(self, ddof: int = 0) -> float:
        """Variance of the values folded in.

        Args:
            ddof: Delta degrees of freedom: 0 for the population variance, 1 for the sample variance.

        Returns:
            The variance, or NaN with fewer than two values.
        """
        if self._n < 2:
            return float("nan")
        return self._m2 / (self._n - ddof)

    def std(self, ddof: int = 0) -> float:
        """Standard deviation, the square root of `var(ddof)`.

        Args:
            ddof: Delta degrees of freedom, as for `var`.

        Returns:
            The standard deviation, or NaN with fewer than two values.
        """
        return math.sqrt(self.var(ddof=ddof))

    @property
    def mean(self) -> float:
        """Mean of the values folded in; 0.0 before the first."""
        return self._mean

    def __enter__(self) -> Self:
        """Reset the tracker on entering a `with` block."""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nothing to finalize; the statistics stay readable after the block."""
        pass


class MinMaxTracker:
    """Running minimum and maximum of a stream of values, with the index at which each was seen.

    The trainers feed it one loss per epoch to find the best epoch. NaN values are ignored. Usable as a
    context manager: entering the `with` block resets extremes and indices.
    """

    def __init__(self):
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._min_idx: Optional[int] = None
        self._max_idx: Optional[int] = None

    def update(self, x: float, idx: Optional[int] = None) -> None:
        """Fold one value in; a NaN is skipped.

        Args:
            x: The new value.
            idx: Its position (an epoch, say); recorded as `min_idx` / `max_idx` when `x` sets a new extreme.
        """
        if not np.isnan(x):
            if idx is not None:
                if x < self._min:
                    self._min = x
                    self._min_idx = idx
                if x > self._max:
                    self._max = x
                    self._max_idx = idx
            else:
                if x < self._min:
                    self._min = x
                if x > self._max:
                    self._max = x

    def clear(self) -> None:
        """Reset the two extremes; the recorded indices are left as they are (only `__enter__` resets them)."""
        self._min = float("inf")
        self._max = float("-inf")

    @property
    def min(self) -> float:
        """Smallest value seen; `inf` before the first."""
        return self._min

    @property
    def max(self) -> float:
        """Largest value seen; `-inf` before the first."""
        return self._max

    @property
    def min_idx(self) -> Optional[int]:
        """Index passed with the smallest value, or `None` when no index was given."""
        return self._min_idx

    @property
    def max_idx(self) -> Optional[int]:
        """Index passed with the largest value, or `None` when no index was given."""
        return self._max_idx

    def __enter__(self) -> Self:
        """Reset extremes and indices on entering a `with` block."""
        self._min = float("inf")
        self._max = float("-inf")
        self._min_idx = None
        self._max_idx = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nothing to finalize; the extremes stay readable after the block."""
        pass


class EarlyStopTracker:
    """Early stopping for a training loop: stop once `patience` epochs pass without a large enough improvement.

    The loop calls `update(metric, epoch)` once per epoch and breaks when it returns `True`; `improved` says
    whether the last epoch set a new best. Improvement is measured against the best value so far, relative to
    it or absolute, for a metric to minimize (a loss) or maximize (an accuracy).

    Args:
        patience: Epochs without improvement before stopping.
        min_delta: The improvement an epoch must exceed to count; relative when `relative`, absolute otherwise.
        mode: `min` when lower is better, `max` when higher is better.
        relative: Whether `min_delta` is a fraction of the best value rather than a difference.

    Raises:
        ValueError: If `mode` is not `min` / `max`, `patience` is not positive or `min_delta` is negative.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min", relative: bool = True):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.relative = relative

        self._best_metric = None
        self._best_epoch = 0
        self._patience_counter = 0
        self._improved = False
        self._improvement = 0.0
        self._stop = False

    def update(self, metric: float, epoch: int) -> bool:
        """Record this epoch's metric and say whether the loop should stop.

        Args:
            metric: The metric of the epoch just finished.
            epoch: Its index, remembered as `best_epoch` when the metric is a new best.

        Returns:
            `True` when the patience is exhausted or the metric is NaN, else `False`.
        """
        # A NaN metric means training has diverged: stop at once rather than waiting out the patience.
        if np.isnan(metric):
            self._improved = False
            self._stop = True
            return self.should_stop

        if self._best_metric is None:
            self._best_metric = metric
            self._best_epoch = epoch
            self._patience_counter = 0
            self._improved = True
            self._stop = False
            return self.should_stop

        self._improvement = self._calculate_improvement(metric)

        if self._improvement > self.min_delta:
            self._best_metric = metric
            self._best_epoch = epoch
            self._patience_counter = 0
            self._improved = True
            self._stop = False
        else:
            self._patience_counter += 1
            self._improved = False
            self._stop = True if self._patience_counter >= self.patience else False

        return self.should_stop

    def _calculate_improvement(self, current_metric: float) -> float:
        """Calculate improvement based on mode and relative setting.

        Args:
            current_metric: Current metric value to compare against best.

        Returns:
            Improvement value (positive means improvement, negative means degradation).
        """
        if self.mode == "min":
            raw_improvement = self._best_metric - current_metric
        else:
            raw_improvement = current_metric - self._best_metric

        if self.relative:
            # The floor keeps a best metric of exactly zero from dividing by zero.
            return raw_improvement / max(abs(self._best_metric), 1e-8)
        else:
            return raw_improvement

    @property
    def best_metric(self) -> float:
        """Best metric seen so far; `None` before the first update."""
        return self._best_metric

    @property
    def best_epoch(self) -> Optional[int]:
        """Epoch index of the best metric."""
        return self._best_epoch

    @property
    def patience_counter(self) -> int:
        """Epochs since the last improvement."""
        return self._patience_counter

    @property
    def improved(self) -> bool:
        """Whether the last update set a new best."""
        return self._improved

    @property
    def improvement(self) -> float:
        """Improvement measured by the last update, relative or absolute per `relative`."""
        return self._improvement

    @property
    def should_stop(self) -> bool:
        """Whether the last update asked the loop to stop."""
        return self._stop

    def clear(self) -> None:
        """Forget the best metric, its epoch and the patience count (`improvement` and `should_stop` are kept)."""
        self._best_metric = None
        self._best_epoch = None
        self._patience_counter = 0
        self._improved = False

    def __enter__(self) -> Self:
        """Reset the tracker on entering a `with` block."""
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nothing to finalize; the state stays readable after the block."""
        pass
