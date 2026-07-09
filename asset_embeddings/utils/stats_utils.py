import math
from typing import Optional, Self
import numpy as np


class MeanVarianceTracker:
    """Welford's algorithm for iterative mean and variance calculation."""

    def __init__(self):
        self._n: int = 0  # 计数器
        self._mean: float = 0.0  # 均值
        self._m2: float = 0.0  # 用于计算方差的中间量

    def update(self, x: float) -> None:
        if not np.isnan(x):
            # 更新计数
            self._n += 1

            # 计算新的均值
            delta = x - self._mean
            self._mean += delta / self._n

            # 更新M2，用于方差计算
            delta2 = x - self._mean
            self._m2 += delta * delta2

    def clear(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0  # 重置计数器和中间量

    @property
    def count(self) -> int:
        return self._n

    @property
    def sum(self) -> float:
        return self._mean * self._n

    def var(self, ddof: int = 0) -> float:
        if self._n < 2:
            return float("nan")  # 当样本数小于2时，方差无定义
        return self._m2 / (self._n - ddof)  # 使用无偏估计

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.var(ddof=ddof))

    @property
    def mean(self) -> float:
        return self._mean

    def __enter__(self) -> Self:
        """进入上下文时，初始化/重置计算"""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开上下文时存储最终的均值和方差"""
        pass


class MinMaxTracker:
    def __init__(self):
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._min_idx: Optional[int] = None
        self._max_idx: Optional[int] = None

    def update(self, x: float, idx: Optional[int] = None) -> None:
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
        self._min = float("inf")
        self._max = float("-inf")

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def min_idx(self) -> Optional[int]:
        return self._min_idx

    @property
    def max_idx(self) -> Optional[int]:
        return self._max_idx

    def __enter__(self) -> Self:
        """进入上下文时，初始化/重置计算"""
        self._min = float("inf")
        self._max = float("-inf")
        self._min_idx = None
        self._max_idx = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开上下文时存储最终的均值和方差"""
        pass


class EarlyStopTracker:
    """Early stopping tracker for training loops.

    Monitors metric improvements and triggers stopping when patience is exhausted.
    Supports both absolute and relative improvement thresholds, and both minimization
    and maximization objectives.

    Usage:
        tracker = EarlyStopTracker(patience=5, min_delta=0.001, mode='min', relative=True)

        for epoch in range(max_epochs):
            loss = train_epoch()

            should_stop = tracker.update(loss, epoch)

            if tracker.improved:
                logger.info(f"Metric improved to {loss:.6f}")

            if should_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    Attributes:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement threshold.
        mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better).
        relative: If True, use relative improvement; if False, use absolute improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min", relative: bool = True):
        """Initialize early stopping tracker.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement threshold (relative if relative=True, absolute otherwise).
            mode: 'min' (lower is better, e.g., loss) or 'max' (higher is better, e.g., accuracy).
            relative: If True, use relative improvement; if False, use absolute improvement.
        """
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
        """Update tracker with new metric value.

        Args:
            metric: Current metric value.
            epoch: Current epoch index.

        Returns:
            True if early stopping should be triggered (patience exhausted), False otherwise.
        """

        # Handle NaN metric values
        if np.isnan(metric):
            # Skip NaN values
            self._improved = False
            self._stop = True
            return self.should_stop

        # First metric observed
        if self._best_metric is None:
            # First metric observed
            self._best_metric = metric
            self._best_epoch = epoch
            self._patience_counter = 0
            self._improved = True
            self._stop = False
            return self.should_stop

        # Calculate improvement
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
        else:  # mode == 'max'
            raw_improvement = current_metric - self._best_metric

        if self.relative:
            # Use max(abs(best_metric), 1e-8) to avoid division by zero
            return raw_improvement / max(abs(self._best_metric), 1e-8)
        else:
            return raw_improvement

    @property
    def best_metric(self) -> float:
        """Best metric value observed so far."""
        return self._best_metric

    @property
    def best_epoch(self) -> Optional[int]:
        """Epoch index with best metric."""
        return self._best_epoch

    @property
    def patience_counter(self) -> int:
        """Current patience counter (epochs without improvement)."""
        return self._patience_counter

    @property
    def improved(self) -> bool:
        """Whether improvement was detected in last update."""
        return self._improved

    @property
    def improvement(self) -> float:
        """Improvement value from last update."""
        return self._improvement

    @property
    def should_stop(self) -> bool:
        """Whether early stopping should be triggered."""
        return self._stop

    def clear(self) -> None:
        """Reset tracker to initial state."""
        self._best_metric = None
        self._best_epoch = None
        self._patience_counter = 0
        self._improved = False

    def __enter__(self) -> Self:
        """Context manager entry: reset tracker."""
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
