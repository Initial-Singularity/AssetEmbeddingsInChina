# Prepare data

Two data regimes: **CSMAR** for production/paper reproduction, and a committed **synthetic** dataset for the demo (no external data needed). For the per-script CLI flags that turn raw data into training inputs, see the [CLI reference](../reference/cli.md).

!!! tip "Just want the embeddings?"
    If you only need the trained asset and text embeddings (not to re-run the pipeline from raw data),
    download them from Zenodo: [10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781)
    (CC-BY-4.0). Those are derived representations, not CSMAR source data — no CSMAR access required.
    CSMAR is only needed to regenerate the embeddings from scratch, as described below.

## CSMAR (production data)

Production runs require CSMAR A-share data — institutional shareholding records, plus company names if you also want the text-embedding pipeline. CSMAR is paywalled; institutional access is typically available via university libraries. CSMAR's license does not permit redistribution of the source data or any derivative slices, so no download mirror is provided — acquire the tables directly from CSMAR.

### Shareholding panel

Institutional holding records (Shanghai / Shenzhen / Beijing exchanges + public-fund periodic reports),
split by year as UTF-8 CSV. Key fields:

| Field | Description |
|---|---|
| `InstitutionID` | Listed-company record code (CSMAR-internal, **1:1 with `Symbol`**). **Not** an investor ID — the investor ID is `ShareHolderID`. See [Data pipeline](../explanation/data-pipeline.md). |
| `Symbol` | Stock code. |
| `EndDate` | Report end date. |
| `ShareHolderID` / `ShareHolderName` | Holding institution ID / name (the real investor identity). |
| `SystematicsID` / `CategoryCode` | Institution classification codes (fund / QFII / brokerage / insurance / social-security / trust / bank / …). |
| `Holdshares` | Shares held. |
| `HoldProportion` | Holding proportion vs **total** shares. |
| `HoldProportion1` | Holding proportion vs **tradable A-shares**. |
| `Price` | Stock price closest to `EndDate`. |
| `IndustryCode` / `IndustryName` | CSRC-2012 industry classification at `EndDate`. |

## Synthetic data (demo)

One committed dataset — no external data needed to reproduce the demo:

- `examples/data/` — synthetic portfolios for the end-to-end W2V-PT → W2V-FT → BERT-PT → BERT-FT walkthrough in `examples/quickstart.{ipynb,py}`. Schema and generation rationale: `examples/data/README.md`.

## Processing pipeline

Raw data is turned into training inputs by the `asset_embeddings.scripts.data.*` tools — `sharehold` (holdings → portfolios), `distributor` (train-split, e.g. 80/20 pretrained/finetune), and `text_embedding` (company names → LLM text embeddings, see [Text embeddings](text-embeddings.md)). Each is flag-driven; see the [CLI reference](../reference/cli.md) for the full argument tables. The preset-driven batch path is in [Install → Quickstart with presets](install.md).
