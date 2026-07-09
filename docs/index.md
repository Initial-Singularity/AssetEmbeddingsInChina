# AssetEmbeddingsInChina

> Learning asset embeddings from partial institutional holdings in China's A-share market.
> Companion training code for the working paper **"Do Asset Embeddings Need Complete Holdings? Lessons from China"** (2026).

Following [Gabaix et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4507511), we treat
each investor's portfolio as a *sentence* and each stock as a *token*: rank order carries economic
meaning (the largest position is the primary exposure), just as word order carries meaning in language.
This lets us learn asset representations with NLP techniques for asset pricing and investor modeling.

<p align="center"><img src="figures/portfolio_sentence.svg" alt="Portfolio-sentence isomorphism" width="640"></p>

## What's new

Gabaix et al. fit a single cross-section. We need a sequence of embeddings that moves with the data
*in a coherent coordinate frame*. Fitting each quarter independently (the **naive** procedure) leaves
rotation/reflection/permutation of the latent axes unidentified, so the change between quarters
confounds true asset-level change with optimization-frame drift.

Our **coupled** procedure pretrains a base model on the first 80% of holding observations (the corpus is split chronologically by sequence count, placing the cutoff at 2023Q1: 73 quarters for pretraining, 6 for quarterly fine-tuning) and initializes every quarterly fine-tune from that shared anchor — the central methodological commitment of the paper. Two initialization couplings drive the pipeline (W2V→BERT and PT→FT). See [Method: paper ↔ code](explanation/method.md).

<p align="center"><img src="figures/pipeline.svg" alt="Pretrain-finetune pipeline" width="720"></p>

## What the repo delivers

A training framework for the full embedding pipeline: three architectures (recommender-system SVD, Word2Vec, BERT), the coupled pretrain–finetune procedure, and the supporting data tooling. The output is a quarterly sequence of embedding CSVs — one vector per stock per quarter, all in one coherent coordinate frame. The companion paper studies how well these embeddings explain firm valuations and return correlations.

## Quickstart

```bash
git clone https://github.com/Initial-Singularity/AssetEmbeddingsInChina
cd AssetEmbeddingsInChina
uv sync --extra notebook                      # notebook + ipykernel are demo-only extras
uv run jupyter lab examples/quickstart.ipynb  # or: uv run python examples/quickstart.py
```

The notebook runs the full **W2V → BERT-PT → BERT-FT** chain on a synthetic dataset shipped in `examples/data/` (CPU, ~1 minute), ending with three quarterly embedding CSVs. Real production runs require CSMAR data — see [Prepare data](how-to/prepare-data.md). Full walkthrough: [Quickstart](tutorials/quickstart.md).

## Where to go

- **New here?** [Quickstart](tutorials/quickstart.md)
- **A recipe?** [Install](how-to/install.md) · [Prepare data](how-to/prepare-data.md) · [Train](how-to/train.md)
- **The "why"?** [Method & experiments](explanation/method.md)
- **Just the embeddings?** Download them from Zenodo: [10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781) (CC-BY-4.0)
- **Looking something up?** [CLI reference](reference/cli.md) · [API reference](reference/api/index.md)

## Citation

```bibtex
@unpublished{cheng2026asset,
  title  = {Do Asset Embeddings Need Complete Holdings? Lessons from China},
  author = {Cheng, Zitong and Huang, Dashan and Liu, Xiaobin and Zeng, Tao},
  year   = {2026},
  note   = {Working paper},
}
```

The trained embeddings are archived on Zenodo (CC-BY-4.0):
[10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781).
