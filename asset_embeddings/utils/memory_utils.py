"""Process and GPU memory readings.

One-shot RAM/VRAM snapshots, the `ResourceProbe` a `Run` carries, and a chunk-size estimate for reading a large
file in pieces.
"""

import os
from typing import Dict, Optional

import psutil
import torch


def check_vram(device: torch.device) -> Dict[str, float]:
    """Allocated and reserved CUDA memory on `device`, in MB.

    Args:
        device: The CUDA device to read.

    Returns:
        `{"allocated_memory": MB, "cached_memory": MB}`, the second being the allocator's reserved total.
    """
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
    cached_memory = torch.cuda.memory_reserved(device) / (1024**2)
    return {"allocated_memory": allocated_memory, "cached_memory": cached_memory}


def get_memory_usage() -> Dict[str, float]:
    """Resident memory of the current process, in MB and as a percentage of total system memory.

    Returns:
        `{"used_memory_mb": MB, "percent_of_total": percent}`.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss
    total_memory = psutil.virtual_memory().total
    used_memory_mb = mem_info / 1024 / 1024
    percent = mem_info / total_memory * 100
    return {"used_memory_mb": used_memory_mb, "percent_of_total": percent}


class ResourceProbe:
    """Peak resource use of the current process, read cheaply and reported only when it matters.

    ``sample()`` reads the resident memory of the process tree (the process and its children, i.e. the
    DataLoader workers) and, once CUDA has been initialized by the run, the CUDA allocator counters (no device
    sync). ``changed()`` returns one readable line the first time and whenever the reserved VRAM or the peak
    allocation has grown since the last line, else ``None``: a training loop calls it every check step and the
    log shows the warm-up growth and nothing afterwards. ``summary()`` gives the peaks for the ``runs`` row. A run
    that never touches the GPU never initializes CUDA through the probe: its VRAM figures stay ``None``.
    """

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._peak_ram_mb = 0.0
        self._reported = (0.0, 0.0)  # (reserved_mb, peak_vram_mb) of the last line from changed()
        self.start()

    @staticmethod
    def _cuda_ready() -> bool:
        return torch.cuda.is_available() and torch.cuda.is_initialized()

    def _tree_rss_mb(self) -> float:
        """RSS of the process plus its live children (DataLoader workers hold their own dataset copies)."""
        rss = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return rss / 2**20

    def start(self) -> None:
        """Reset the baselines: the RSS peak from now, the CUDA peak counters too if CUDA is up."""
        self._peak_ram_mb = self._tree_rss_mb()
        self._reported = (0.0, 0.0)
        if self._cuda_ready():
            torch.cuda.reset_peak_memory_stats()

    def sample(self) -> Dict[str, Optional[float]]:
        """Current and peak figures in MB; the VRAM entries are ``None`` while CUDA is not initialized."""
        ram = self._tree_rss_mb()
        self._peak_ram_mb = max(self._peak_ram_mb, ram)
        out: Dict[str, Optional[float]] = {
            "ram_mb": ram,
            "peak_ram_mb": self._peak_ram_mb,
            "vram_mb": None,
            "reserved_mb": None,
            "peak_vram_mb": None,
        }
        if self._cuda_ready():
            out["vram_mb"] = torch.cuda.memory_allocated() / 2**20
            out["reserved_mb"] = torch.cuda.memory_reserved() / 2**20
            out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        return out

    def changed(self) -> Optional[str]:
        """One line when the VRAM high-water mark moved since the last line, else ``None``."""
        s = self.sample()
        if s["reserved_mb"] is None:
            return None
        mark = (s["reserved_mb"], s["peak_vram_mb"])
        if mark[0] <= self._reported[0] and mark[1] <= self._reported[1]:
            return None
        self._reported = mark
        return (
            f"VRAM {s['vram_mb']:.0f} MB allocated, {s['reserved_mb']:.0f} MB reserved, "
            f"peak {s['peak_vram_mb']:.0f} MB; RAM {s['ram_mb']:.0f} MB"
        )

    def summary(self) -> Dict[str, Optional[float]]:
        """``peak_ram_mb`` and ``peak_vram_mb`` (``None`` without CUDA): the resource columns of ``runs``."""
        s = self.sample()
        return {"peak_ram_mb": s["peak_ram_mb"], "peak_vram_mb": s["peak_vram_mb"]}

    @staticmethod
    def describe(summary: Dict[str, Optional[float]]) -> str:
        """``peak RAM 2105 MB[, peak VRAM 1210 MB]``."""
        parts = [f"peak RAM {summary['peak_ram_mb']:.0f} MB"]
        if summary.get("peak_vram_mb") is not None:
            parts.append(f"peak VRAM {summary['peak_vram_mb']:.0f} MB")
        return ", ".join(parts)


def calculate_chunk_size(file_path: str, avg_row_size: Optional[int] = None, memory_fraction: float = 0.5):
    """Row count per chunk for reading `file_path` in pieces, sized to the memory available right now.

    Args:
        file_path: The text file to be read in chunks.
        avg_row_size: Average bytes per row; when `None` it is estimated from the first 100 lines, so the
            file must have at least that many.
        memory_fraction: Share of the available memory one chunk may occupy.

    Returns:
        The chunk size: the smaller of what fits in memory and the estimated row count, and at least 1.
    """
    file_size = os.path.getsize(file_path)
    available_memory = psutil.virtual_memory().available

    if avg_row_size is None:
        with open(file_path, "r") as f:
            sample_lines = [next(f) for _ in range(100)]
            avg_row_size = sum(len(line.encode("utf-8")) for line in sample_lines) // len(sample_lines)

    max_rows_in_memory = (available_memory * memory_fraction) // avg_row_size
    estimated_rows = file_size // avg_row_size
    chunk_size = min(max_rows_in_memory, estimated_rows)

    return max(chunk_size, 1)
