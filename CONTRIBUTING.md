# Contributing

Thanks for your interest. This codebase backs a working paper ("Do Asset Embeddings Need Complete Holdings? Lessons from China", 2026); the core methodology is frozen. We welcome bug reports, reproducibility issues, and small improvements; large architectural changes are unlikely to be merged.

## Dev environment

```bash
uv sync
uv run pre-commit install
```

## Code style

Enforced by pre-commit (ruff lint + format). Run:

```bash
uv run pre-commit run --all-files
```

## Issue triage

Please include: minimal reproducible example, environment (`uv pip freeze`), and CSMAR data scope you tested against (if relevant).
