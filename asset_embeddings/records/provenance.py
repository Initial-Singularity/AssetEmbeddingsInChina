"""Run provenance: experiment derivation and git / host / package info.

Everything here is best-effort and never raises for missing tools or files —
a run must not fail because ``git`` is absent or an input path is unreadable.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from importlib import metadata
from typing import Any, Dict, Iterable, List, Optional

#: Stage directories: a segment equal to one of these, or starting with one plus ``_`` (``finetune_mpdiag``), ends the experiment name.
_STAGE_SEGMENTS = ("eval", "finetune", "pretrained", "refine", "data")
#: Task / model / cell directories that never belong to an experiment name.
_TASK_SEGMENT = re.compile(
    r"^(RV|RC|MP|MP_\w+|Pred-\w+|GROUPED_\w+|Asset\w+|RS_\w+|BERT|SMLM|d\d+(_\w+)?|ridge(\d[\w.]*|None)|\d{4}Q\d|Placebo|Chars)$"
)


def _is_stage(seg: str) -> bool:
    return any(seg == s or seg.startswith(s + "_") for s in _STAGE_SEGMENTS)


def _is_task(seg: str) -> bool:
    return bool(_TASK_SEGMENT.match(seg)) or seg.endswith(".json")


def derive_experiment(config_file: str) -> str:
    """Experiment family from a config path; ``""`` when the path carries no family name.

    Layouts:
    - current, ``configs/<family>[/<split>]/<stage>/...``: ``configs/main/eval/RV/d16/x.json`` -> ``main``,
      ``configs/pt_strategy_ablation/no_pt_init/finetune/...`` -> ``pt_strategy_ablation/no_pt_init``,
      ``configs/holdout/eval/cv/RV/...`` -> ``holdout``;
    - legacy, ``configs/<stage>/<family>/...``: ``configs/eval/timesplit_robust/RV/...`` -> ``timesplit_robust``;
    - legacy without a family, ``configs/eval/RV/...``: ``""`` (callers fall back, e.g. to the database name);
    - no ``configs`` segment: the leading directory, or the file stem for a bare file name.
    """
    parts = [p for p in str(config_file).replace("\\", "/").split("/") if p and p != "."]
    if not parts:
        return ""
    if "configs" not in parts:
        return parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]
    parts = parts[parts.index("configs") + 1 :]
    if not parts:
        return ""
    if _is_stage(parts[0]):  # legacy layout: stage first
        nxt = parts[1] if len(parts) > 1 else ""
        return nxt if nxt and not _is_task(nxt) and not _is_stage(nxt) else ""
    kept: List[str] = []
    for seg in parts:
        if _is_stage(seg) or _is_task(seg):
            break
        kept.append(seg)
    return "/".join(kept)


def _git(args: List[str], cwd: Optional[str]) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def collect_provenance(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Git commit / dirty flag, hostname and package version.

    Returns a dict with exactly the keys ``git_commit``, ``git_dirty``, ``hostname``,
    ``package_version``. Missing information is ``None``.
    """
    head = _git(["rev-parse", "HEAD"], repo_root)
    git_commit = head.strip() if head else None
    git_dirty: Optional[bool] = None
    if git_commit is not None:
        status = _git(["status", "--porcelain"], repo_root)
        git_dirty = bool(status.strip()) if status is not None else None
    try:
        hostname: Optional[str] = socket.gethostname()
    except Exception:
        hostname = None
    package_version: Optional[str] = None
    for dist_name in ("AssetEmbeddingsInChina", "asset_embeddings"):  # pyproject name first, import name as fallback
        try:
            package_version = metadata.version(dist_name)
            break
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            break
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hostname": hostname,
        "package_version": package_version,
    }
