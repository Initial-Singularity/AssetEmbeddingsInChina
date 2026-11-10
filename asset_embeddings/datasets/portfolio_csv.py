"""The portfolio CSV format: one investor per row, the holdings and their proportions as comma-joined lists.

The data-preparation step writes it and the training side reads it.
Header ``InvestorID,Portfolio[,Proportion1,Proportion2]``; ``Portfolio`` lists the stock codes, the proportion
columns list one value per code with six decimals. The csv module quotes a list field that holds a comma.
"""

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

PathLike = Union[str, Path]

SEPARATOR = ","


def resolve_data_files(path: Union[PathLike, List[PathLike]]) -> List[Path]:
    """Resolve a file, a directory or a list of both into the csv files they name, in order.

    Args:
        path: A csv file, a directory (its ``*.csv`` files sorted by name) or a list of those, concatenated in
            the order given.

    Returns:
        The files.

    Raises:
        FileNotFoundError: A path does not exist, or nothing resolves to a csv file.
        ValueError: A file that is not a ``.csv``.
    """
    paths = path if isinstance(path, list) else [path]
    files: List[Path] = []
    for p in map(Path, paths):
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.suffix.lower() != ".csv":
            raise ValueError(f"File {p} is not a csv file")
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No csv files found in path: {path}")
    return files


def _columns(include_proportion: bool, id_key: str, portfolio_key: str, p1: str, p2: str) -> List[str]:
    return [id_key, portfolio_key] + ([p1, p2] if include_proportion else [])


def _join_codes(codes: Iterable) -> str:
    items = [str(c) for c in codes]
    for item in items:
        if SEPARATOR in item:
            raise ValueError(f"Stock code {item!r} holds the list separator {SEPARATOR!r}")
    return SEPARATOR.join(items)


def _join_proportions(values: Iterable) -> str:
    return SEPARATOR.join(f"{float(v):.6f}" for v in values)


def _split(field: str) -> List[str]:
    return [item.strip() for item in field.split(SEPARATOR) if item.strip()]


def write_portfolio_csv(
    df: pd.DataFrame,
    path: PathLike,
    include_proportion: Optional[bool] = False,
    *,
    id_key: str = "InvestorID",
    portfolio_key: str = "Portfolio",
    proportion1_key: str = "Proportion1",
    proportion2_key: str = "Proportion2",
) -> None:
    """Write portfolio rows to ``path``.

    Args:
        df: One row per investor: the id, the list of codes and, with proportions, the two lists of values.
        path: The file written.
        include_proportion: Whether to write the proportion columns; None writes them when ``df`` has both.
        id_key: Id column.
        portfolio_key: Codes column.
        proportion1_key: Proportion of total shares column.
        proportion2_key: Proportion of tradable A-shares column.

    Raises:
        ValueError: A required column is missing, or a code holds a comma.
    """
    if include_proportion is None:
        include_proportion = proportion1_key in df.columns and proportion2_key in df.columns
    columns = _columns(include_proportion, id_key, portfolio_key, proportion1_key, proportion2_key)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} for {path}; available: {list(df.columns)}")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in df[columns].itertuples(index=False):
            fields = [row[0], _join_codes(row[1])]
            if include_proportion:
                fields += [_join_proportions(row[2]), _join_proportions(row[3])]
            writer.writerow(fields)


def read_portfolio_csv(
    path: PathLike,
    include_proportion: Optional[bool] = False,
    *,
    id_key: str = "InvestorID",
    portfolio_key: str = "Portfolio",
    proportion1_key: str = "Proportion1",
    proportion2_key: str = "Proportion2",
) -> pd.DataFrame:
    """Read a portfolio file into one row per investor with list columns.

    Args:
        path: The file.
        include_proportion: Whether to read the proportion columns; None reads them when the file has both.
        id_key: Id column (read as a string).
        portfolio_key: Codes column (a list of strings).
        proportion1_key: Proportion of total shares column (a list of floats).
        proportion2_key: Proportion of tradable A-shares column (a list of floats).

    Returns:
        The id and codes columns, then the two proportion columns when they are read.

    Raises:
        ValueError: The file is empty, a required column is missing, or a proportion list differs in length from
            its codes.
    """
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Portfolio file {path} is empty")
        if include_proportion is None:
            include_proportion = proportion1_key in header and proportion2_key in header
        columns = _columns(include_proportion, id_key, portfolio_key, proportion1_key, proportion2_key)
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(f"Portfolio file {path} lacks columns {missing}; found {header}")
        at = [header.index(c) for c in columns]
        records = []
        for line, fields in enumerate(reader, start=2):
            codes = _split(fields[at[1]])
            record = [fields[at[0]], codes]
            if include_proportion:
                for key, i in zip((proportion1_key, proportion2_key), at[2:]):
                    values = [float(v) for v in _split(fields[i])]
                    if len(values) != len(codes):
                        raise ValueError(
                            f"Portfolio file {path}, line {line}: {key} holds {len(values)} values for "
                            f"{len(codes)} codes"
                        )
                    record.append(values)
            records.append(record)
    return pd.DataFrame(records, columns=columns)
