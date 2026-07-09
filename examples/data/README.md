# Example data — synthetic

This directory holds the synthetic dataset used by `examples/quickstart.{ipynb,py}`.
**Every number here is generated — none of it
comes from CSMAR or any other commercial data vendor.** It exists purely so the
example can ship inside a public repository.

## Files

| File | Rows | Description |
|---|---|---|
| `portfolio_pretrained.csv` | 1500 | PT-period investor portfolios. Schema: `InvestorID`, `Portfolio` (comma-separated stock codes), `Proportion1`, `Proportion2` (comma-separated weights). |
| `portfolio_finetune.csv`   | 1500 | FT-period portfolios — a drifted snapshot of the same world. Same schema. |

## How it is shaped

The data is sampled from a latent factor model designed to mirror the
*statistical structure* of A-share shareholding data **and** to make the
PT-FT distinction observable in the learned embeddings:

- A universe of ~2500 stock codes drawn from realistic A-share board ranges
  (Shanghai main `600xxx/601xxx/603xxx`, Shenzhen main `000xxx`, SME `002xxx`,
  ChiNext `300xxx/301xxx`, STAR `688xxx`).
- 18 latent "industry-like" clusters with centroids in a 10-D factor space.
- **Two stock-factor matrices**:
  - `stock_factors_pt` = cluster_centroid + per-stock jitter (PT period)
  - `stock_factors_ft` = stock_factors_pt + cluster-level drift + per-stock noise (FT period)
- Investors prefer 1–3 clusters; PT portfolios are sampled via softmax similarity
  against `stock_factors_pt`, FT portfolios against `stock_factors_ft`.

This design makes the PT-FT distinction observable in the learned
representations: W2V and BERT-PT (both PT-trained) recover the PT-period
cluster structure, while BERT-FT — fine-tuned on FT-period portfolios —
additionally picks up the cluster-level drift.

## Reproducing this dataset

The exact generation recipe is **not shipped in the repository** — it lives in
a private `_dev/` directory. The CSVs here are the canonical artifact. Treat
them as a deterministic, redistributable demo fixture; do **not** rely on them
for anything other than verifying the pipeline runs.

For real CSMAR data acquisition, see [`docs/how-to/prepare-data.md`](../../docs/how-to/prepare-data.md).
