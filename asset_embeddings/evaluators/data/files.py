"""Turn a per-period path config field into its list of files."""

from pathlib import Path
from typing import List, Union


def resolve_period_files(path: Union[str, Path, List[str], None]) -> List[Path]:
    """The files of a per-period path field, in the order of ``config.periods``.

    The config guarantees one file per period (a list as long as ``periods``, or one file with one period), so
    the order is the list's own; files are never matched to periods by name.

    Args:
        path: The config value: a list of files, a single file, or None.

    Returns:
        The files as ``Path`` objects; ``[]`` for None.
    """
    if path is None:
        return []
    if isinstance(path, list):
        return [Path(p) for p in path]
    return [Path(path)]
