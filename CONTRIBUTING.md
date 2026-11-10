# Contributing

Thanks for your interest. This codebase backs a working paper ("Do Asset Embeddings Need Complete Holdings? Lessons from China", 2026); the core methodology is frozen. We welcome bug reports, reproducibility issues, and small improvements; large architectural changes are unlikely to be merged.

## Dev environment

```bash
uv sync
uv run pre-commit install
```

## Code style

Formatting and linting are enforced by pre-commit (`ruff format`, `ruff check`), including the docstring rules (`D`, Google convention). Run:

```bash
uv run pre-commit run --all-files
```

### Docstrings

Google style, English, on every public module, class, function and method of `asset_embeddings/` and `scripts/` (public = no leading underscore; `__init__` and magic methods need none).

- Line 1 is one sentence ending with a period that says what the object is or does for its caller. It is what the API reference shows in lists.
- Sections, only when they add something, in this order: `Args:` (one line per parameter, no types: the signature carries them), `Returns:` / `Yields:`, `Raises:`, `Example:` (rare), `Note:`.
- Cross-references use the mkdocstrings form ``[`RecordStore`][asset_embeddings.records.store.RecordStore]`` or plain backticks. No reST roles (`:class:`, `:meth:`, `:func:`, `:param:`).
- A module docstring is one to three sentences on what the module holds and who calls it. A CLI module names the path it runs as (`python -m asset_embeddings.scripts.eval.rv`). No usage essays, no changelog.
- A `ConfigField(doc=...)` string is that field's documentation; the class docstring does not repeat the field list.

### Inline comments

A comment says why, or names an invariant, precondition or trap the code does not make obvious. It never restates what the next line already says.

- Not allowed: comments that narrate the code (`# validate columns`), rule banners (`# ----`, `# ==== Title ====`), commented-out code, generation markers (`v2 behavior`).
- A comment must stand alone: no references to review IDs or private design notes. Write the reason itself. A derivation too long for a comment belongs in `docs/explanation/`.
- `TODO` / `FIXME` carry an owner and a reason on the same line.

### Command-line entry points

Every module under `asset_embeddings/scripts/` that runs as a CLI has the same shape:

- `get_parser()` builds and returns the `argparse.ArgumentParser`, without parsing, so tests can parse a given argument list.
- `main(args)` does the work on the parsed arguments, so tests can run it in process.
- `if __name__ == "__main__": main(get_parser().parse_args())`.

A check that needs the parser after parsing (a start period after the end period, defaults from a `--config` file) goes into a `parse_args(argv=None)` built on `get_parser()`, and the guard calls `main(parse_args())`. The config-driven training and evaluation scripts add the six shared flags with `utils.add_run_args(parser, override_example=...)`; their own parser holds only the description and the config layout of the `--help` epilog.

### Type hints

Not enforced by lint. Public signatures should be annotated; a type-free `Args:` section is complete only when the signature carries the types.

## Issue triage

Please include: minimal reproducible example, environment (`uv pip freeze`), and CSMAR data scope you tested against (if relevant).
