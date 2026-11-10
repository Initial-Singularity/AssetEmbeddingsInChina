# Examples

End-to-end walkthrough of the project's canonical training chain — **W2V → BERT-PT → BERT-FT → RV ×3** — on a synthetic dataset shipped in [`data/`](data/). CPU only, ~1 minute, fully redistributable (every number is generated; see [`data/README.md`](data/README.md)).

Two equivalent entry points share the same pipeline:

| Format | File | Use when |
|---|---|---|
| Jupyter notebook | [`quickstart.ipynb`](quickstart.ipynb) | You want narrative alongside each step. |
| Plain-Python script | [`quickstart.py`](quickstart.py) | You want a one-shot run with no Jupyter setup. |

## Notebook setup

`jupyterlab` and `ipykernel` are **demo-only extras** — not part of the core install. Run once:

```bash
uv sync --extra notebook
uv run jupyter lab examples/quickstart.ipynb
```

The notebook's `kernelspec` is the generic `python3`. Once `ipykernel` is in the project `.venv`, `uv run jupyter lab` picks it up automatically — no manual kernel registration needed.

## Script setup

The script needs nothing beyond the core install:

```bash
uv run python examples/quickstart.py
```

## What you should see

The pipeline produces three asset-embedding artifacts and scores them against the bundled benchmark. Expected RV @ 2024Q2:

| Model | Trained on | RV | Note |
|---|---|---|---|
| W2V | PT data | ~0.53 | baseline |
| BERT-PT | PT data | ~0.53 | matches W2V (same training data) |
| BERT-FT | FT data | ~0.60 | **+0.07 from fine-tuning** |

The +0.07 gap reflects the synthetic data's cluster-level drift between PT and FT periods. On real CSMAR data the lift is typically larger.
