"""Small helpers with no better home.

A tensor NaN/Inf check, an attribute-versus-dict assertion, directory creation and cleaning, the common root of
several paths, and list-column expansion for DataFrames.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
import torch


def check(tensor: torch.Tensor) -> torch.Tensor:
    """Whether `tensor` holds any NaN or Inf, as a zero-dimensional bool tensor.

    Args:
        tensor: The tensor to inspect.

    Returns:
        A bool tensor, `True` when at least one element is not finite.
    """
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))


def check_attr_dict_match(obj, dic: Dict, names: List[str]) -> None:
    """Assert that `obj` and `dic` both carry every name in `names` with equal values.

    Args:
        obj: The object whose attributes are compared.
        dic: The dict whose entries are compared.
        names: The attribute / key names to compare.

    Raises:
        AssertionError: If a name is missing on either side or the two values differ.
    """
    for name in names:
        assert hasattr(obj, name) and name in dic
        assert getattr(obj, name) == dic[name]


def ensure_dir(directory: str) -> None:
    """Create `directory`, parents included, when it does not exist yet.

    Args:
        directory: The directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def prepare_output_dir(
    path: Union[str, Path], clean: bool = False, protect: Iterable[Union[str, Path, None]] = ()
) -> Path:
    """Create an output directory and, with `clean`, empty it first.

    A data script writes one file or directory per period into its output directory and never removes the ones an
    earlier run left there, so a period that no longer exists would still be read by the next step. `clean` removes
    the directory's contents before the run writes; without it a non-empty directory only draws a warning.

    Args:
        path: The output directory.
        clean: Whether to delete everything inside it first.
        protect: The run's inputs (files or directories; `None` entries are skipped). Cleaning is refused when the
            output directory is one of them or contains one, so a mistyped output path cannot delete the input.

    Returns:
        The directory as a `Path`.

    Raises:
        ValueError: If `clean` is set and `path` is a filesystem root, the home directory, the working directory or
            one of its parents, or is or contains a path in `protect`.
    """
    directory = Path(path)
    logger = logging.getLogger(__name__)
    if clean and directory.exists():
        resolved = directory.resolve()
        cwd = Path.cwd().resolve()
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve() or resolved in (cwd, *cwd.parents):
            raise ValueError(f"refusing to clean {resolved}: a root, home or working directory")
        for item in protect:
            if item is None:
                continue
            guarded = Path(item).resolve()
            if resolved == guarded or resolved in guarded.parents:
                raise ValueError(f"refusing to clean {resolved}: it holds the input {guarded}")
        entries = list(directory.iterdir())
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        logger.info(f"Cleaned {directory}: removed {len(entries)} entries")
    elif directory.exists() and any(directory.iterdir()):
        logger.warning(
            f"{directory} already holds files; those this run does not overwrite are kept. Pass --clean to empty it first."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_common_root(dirs: str) -> str:
    """Longest path that every entry of `dirs` lies under.

    Args:
        dirs: The paths to compare (a list, despite the annotation).

    Returns:
        The common root path.

    Raises:
        ValueError: If `dirs` is empty.
        RuntimeError: If the paths share no root, for example on different drives.
    """
    if not dirs:
        raise ValueError("The list of directories is empty.")
    try:
        common_root = os.path.commonpath(dirs)
        if not any([os.path.commonprefix([dir_, common_root]) == common_root for dir_ in dirs]):
            raise RuntimeError("No common root directory found.")
        return common_root
    except ValueError:
        raise RuntimeError("No common root directory found.")


def expand_list_columns(df: pd.DataFrame, col: str, new_col: Optional[str] = None) -> pd.DataFrame:
    """Expand a column of lists into one column per position, `{prefix}_1` to `{prefix}_n`.

    Args:
        df: The frame holding the list column.
        col: Name of the column whose cells are lists.
        new_col: Prefix of the new columns; the original column name when `None`.

    Returns:
        A copy of `df` with the new columns joined on and `col` dropped; `n` is the longest list, and
        shorter lists leave NaN in the trailing columns.
    """
    prefix = new_col if new_col is not None else col

    max_len = df[col].str.len().max()

    expanded_df = pd.DataFrame(df[col].tolist(), index=df.index, columns=[f"{prefix}_{i+1}" for i in range(max_len)])

    result_df = df.join(expanded_df).drop(columns=[col])

    return result_df


def quarters(first: str, last: str) -> List[str]:
    """List the quarter labels from `first` to `last`, both included.

    Args:
        first: First quarter, e.g. ``"2005Q1"``.
        last: Last quarter, e.g. ``"2005Q3"``.

    Returns:
        The labels in order, e.g. ``["2005Q1", "2005Q2", "2005Q3"]``.
    """
    y, q = int(first[:4]), int(first[5])
    ly, lq = int(last[:4]), int(last[5])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out
