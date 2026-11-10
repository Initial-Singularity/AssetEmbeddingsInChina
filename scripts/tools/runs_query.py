"""runs_query — filter inputs for the runners, read from the records database(s).

    uv run python -m scripts.tools.runs_query -d DB [-d DB ...] [--experiment PREFIX] [--task TASK] status
    uv run python -m scripts.tools.runs_query -d DB ... configs [--status completed|error|running] [--compress] [--root DIR]
    uv run python -m scripts.tools.runs_query -d DB ... missing --root DIR [--compress]

`status` prints one row per (experiment, task): completed / error / running counts and the latest start.
`configs` prints the config paths whose LATEST run has the given status, one per line (for --exclude-file
with `completed`, for --filter-file with `error`).
`missing` prints every *.json under DIR that has no completed run in the database(s) (for --filter-file).
`--compress` collapses a directory to `dir/*` when every *.json under it (on disk, recursively) is in the set.
Paths are printed repo-relative with forward slashes, exactly as the runners pass them to --config; the
runners' filter engine lowercases both sides, so case is irrelevant. The tool proposes; the runner never decides.
"""

from __future__ import annotations

import argparse
import errno
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from asset_embeddings.records.query import load_runs

STATUSES = ("completed", "error", "running")


def norm(path: str) -> str:
    """Repo-relative, forward slashes, no leading './'."""
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def load(dbs: Sequence[str], experiment: Optional[str], task: Optional[str]) -> pd.DataFrame:
    """Load the runs of the given databases, filtered by experiment prefix and exact task, config paths normalized.

    Args:
        dbs: Records database paths.
        experiment: Prefix that ``runs.experiment`` must start with, or None for all.
        task: Exact ``runs.task`` value, or None for all.

    Returns:
        One row per run; empty when nothing matches.
    """
    df = load_runs(list(dbs))
    if df.empty:
        return df
    df = df.copy()
    df["experiment"] = df["experiment"].fillna("")
    if experiment:
        df = df[df["experiment"].str.startswith(experiment)]
    if task:
        df = df[df["task"] == task]
    df["config_file"] = df["config_file"].fillna("").map(norm)
    return df


def latest_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config_file: the run with the latest started_at decides the status."""
    if df.empty:
        return df
    d = df[df["config_file"] != ""].sort_values("started_at", kind="stable")
    return d.groupby("config_file", as_index=False).tail(1)


def status_table(df: pd.DataFrame) -> str:
    """Render one line per (experiment, task) with the completed / error / running counts and the latest start.

    Args:
        df: Runs as returned by ``load``.

    Returns:
        The table text, or ``"(no runs)"``.
    """
    if df.empty:
        return "(no runs)"
    rows = []
    for (exp, task), part in df.groupby(["experiment", "task"]):
        counts = part["status"].value_counts()
        rows.append(
            f"{exp:<40} {task:<14} "
            + " ".join(f"{s}={int(counts.get(s, 0))}" for s in STATUSES)
            + f"  latest={part['started_at'].max()}"
        )
    return "\n".join(rows)


def configs_with_status(df: pd.DataFrame, status: str) -> List[str]:
    """Sorted config paths whose latest run has the given status.

    Args:
        df: Runs as returned by ``load``.
        status: One of ``STATUSES``.

    Returns:
        Repo-relative config paths.
    """
    latest = latest_per_config(df)
    if latest.empty:
        return []
    return sorted(latest.loc[latest["status"] == status, "config_file"].tolist())


def on_disk(root: Path) -> List[str]:
    """Sorted, normalized paths of every ``*.json`` under ``root``, recursively."""
    return sorted(norm(str(p)) for p in root.rglob("*.json"))


def missing(df: pd.DataFrame, root: Path) -> List[str]:
    """Config files under ``root`` with no completed run in ``df``, in ``on_disk`` order.

    Args:
        df: Runs as returned by ``load``.
        root: Directory to scan for ``*.json`` files.

    Returns:
        Repo-relative config paths.
    """
    done = set(configs_with_status(df, "completed"))
    return [p for p in on_disk(root) if p not in done]


def compress(paths: Iterable[str], root: Path) -> List[str]:
    """Collapse to `dir/*` where every *.json under dir (on disk) is in `paths`; otherwise list files."""
    wanted: Set[str] = set(paths)
    out: List[str] = []

    def walk(d: Path) -> None:
        files_below = [norm(str(p)) for p in d.rglob("*.json")]
        if files_below and all(f in wanted for f in files_below):
            out.append(norm(str(d)) + "/*")
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            walk(child)
        for f in sorted(d.glob("*.json")):
            if norm(str(f)) in wanted:
                out.append(norm(str(f)))

    walk(root)
    return out


def header(args: argparse.Namespace, mode: str) -> str:
    """Comment line written above the output: the mode, databases, filters and a timestamp."""
    stamp = datetime.now().isoformat(timespec="seconds")
    return f"# runs_query {mode} db={','.join(args.db)} experiment={args.experiment or '*'} task={args.task or '*'} at={stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; ``status`` when no mode is given.

    Args:
        argv: Arguments without the program name, or None for ``sys.argv``.

    Returns:
        Exit code: 0, or 2 when ``--root`` is not a directory.
    """
    parser = argparse.ArgumentParser(
        prog="runs_query", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-d", "--db", action="append", required=True, help="records database (repeatable)")
    parser.add_argument("--experiment", help="prefix filter on runs.experiment")
    parser.add_argument("--task", help="exact filter on runs.task")
    sub = parser.add_subparsers(dest="mode")
    sub.add_parser("status", help="counts per (experiment, task)")
    p_cfg = sub.add_parser("configs", help="config paths whose latest run has --status")
    p_cfg.add_argument("--status", choices=STATUSES, default="completed")
    p_cfg.add_argument("--compress", action="store_true")
    p_cfg.add_argument("--root", default="configs", help="directory used by --compress (default: configs)")
    p_mis = sub.add_parser("missing", help="*.json under --root without a completed run")
    p_mis.add_argument("--root", required=True)
    p_mis.add_argument("--compress", action="store_true")
    args = parser.parse_args(argv)
    mode = args.mode or "status"

    df = load(args.db, args.experiment, args.task)
    if mode == "status":
        print(status_table(df))
        return 0
    if mode == "configs":
        lines = configs_with_status(df, args.status)
        if args.compress:
            lines = compress(lines, Path(args.root))
    else:
        root = Path(args.root)
        if not root.is_dir():
            print(f"runs_query: --root {root} is not a directory", file=sys.stderr)
            return 2
        lines = missing(df, root)
        if args.compress:
            lines = compress(lines, root)
    try:
        print(header(args, mode))
        for line in lines:
            print(line)
        sys.stdout.flush()
    except OSError as exc:  # consumer closed the pipe (`| head`): EPIPE on POSIX, EINVAL on Windows
        if exc.errno not in (errno.EPIPE, errno.EINVAL):
            raise
        try:
            sys.stdout = open(os.devnull, "w")  # silence the interpreter's flush at exit
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
