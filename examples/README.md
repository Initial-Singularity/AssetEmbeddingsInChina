# Examples

End-to-end walkthrough of the project's canonical training chain — **W2V-PT → W2V-FT → BERT-PT → BERT-FT** — on a synthetic dataset shipped in [`data/`](data/). CPU only, ~1 minute, fully redistributable (every number is generated; see [`data/README.md`](data/README.md)).

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

The pipeline produces four asset-embedding artifacts:

| Model | Trained on | Artifact |
|---|---|---|
| W2V-PT | PT data | `_output/w2v/w2v_d16_embedding.csv` |
| W2V-FT | FT data | `_output/w2v_ft/w2v_ft_d16_embedding.csv` |
| BERT-PT | PT data | `_output/bert_pt/bert_pt_d16_best_contextual_embedding.csv` |
| BERT-FT | FT data | `_output/bert_ft/bert_ft_d16_best_contextual_embedding.csv` |

Each CSV is a `(Token, Embed_1..Embed_d)` matrix. The final cell prints the four shapes plus the top-5 cosine neighbours of one stock in each space — the synthetic universe has latent cluster structure, so neighbours stay within a stock's cluster, and the two fine-tuned models pick up the FT-period drift together while staying comparable to their pretrained starting points.
