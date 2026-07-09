# CLI reference

Every entry point runs as a module:

```bash
uv run python -m asset_embeddings.scripts.<group>.<name> [args...]
```

Common flags: `-c`/`--config` (JSON config; training is config-driven),
`-v`/`--verbose` (console DEBUG), `-l`/`--log` (log file path). Pass `-h` to any script for its full
help. Training scripts also accept `--override KEY=VALUE ...` to patch config fields on the command line.

The **data** scripts below are flag-driven. The **training** scripts (`train.rs` / `train.w2v` / `train.bert`)
are config-driven and share one [common option set](#train-eval-shared); see [Train](../how-to/train.md)
for the per-model config-field references.

---

## `data.sharehold` — shareholding → portfolios {#data-sharehold}

Converts raw CSMAR holding records into per-period investor→stock portfolio sequences.

| Parameter | Type | Description |
|---|---|---|
| `--input_folder`, `-i` | `str` | Folder of raw ShareHoldings CSVs. **Required.** |
| `--output_folder`, `-o` | `str` | Output folder for processed portfolios. **Required.** |
| `--intermediate_folder`, `-inter` | `str` | Optional folder for intermediate per-period valid samples/stocks/investors. |
| `--output_format` | `{csv,json,binary}` | Output format. Default: `csv`. |
| `--frequency`, `-freq` | `{Y,HY,Q,M,N}` | Grouping frequency: yearly / half-year / quarterly / monthly / no-grouping. Custom period mappers via the Python API. Default: `Q`. |
| `--date_format` | `str` | Date parse format (ignored when `frequency=N`). Default: auto-detect. |
| `--aggregation`, `-agg` | `{all,last,first,max,mean}` | Dedup strategy for repeated (investor, stock) pairs in a period. `last`/`first` need a date column. Default: `all`. |
| `--include_proportion`, `-p` | `bool` | Emit the two holding proportions alongside the portfolio. Default: `False`. |
| `--investor_lowerbound`, `-il` | `int` | Drop investors holding fewer than N stocks. Default: `10`. |
| `--stock_lowerbound`, `-sl` | `int` | Drop stocks held fewer than N times. Default: `10`. |
| `--source_shareholder_id_column` | `str` | Source investor-ID column. Default: `ShareHolderID`. |
| `--source_stock_symbol_column` | `str` | Source stock-symbol column. Default: `Symbol`. |
| `--source_end_date_column` | `str` | Source end-date column. Default: `EndDate`. |
| `--source_proportion1_column` | `str` | Proportion-1 source column (vs total shares). Default: `HoldProportion`. |
| `--source_proportion2_column` | `str` | Proportion-2 source column (vs tradable A-shares). Default: `HoldProportion1`. |
| `--id_key` / `--portfolio_key` | `str` | Output keys. Defaults: `InvestorID` / `Portfolio`. |
| `--proportion1_key` / `--proportion2_key` | `str` | Output proportion keys. Defaults: `Proportion1` / `Proportion2`. |
| `--stats` | `bool` | Write per-period stats (`statistics.json`, `stock_coverage.csv`) with the intermediate data. Default: `True`. |
| `--plot` | `bool` | Render distribution-analysis plots (`distribution_analysis.png`) with the intermediate data. Default: `False`. |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | Console DEBUG / log-file path. |

## `data.distributor` — train/val/test split {#data-distributor}

Splits processed data into datasets by ratio, file- or row-level, in-memory or streaming.

| Parameter | Type | Description |
|---|---|---|
| `--input_path`, `-i` / `--output_path`, `-o` | `str` | Input (file or folder) / output path. |
| `--file_format`, `-f` | `{csv,json,binary}` | Input/output format. |
| `--split_ratios`, `-r` | `list[float]` | Target split ratios (sum to 1.0). Default: `[0.8, 0.2]`. |
| `--dir_names` | `list[str]` | Target subdir names. Default: `['pretrained', 'finetune']`. |
| `--keep_file_integrity` | `bool` | Split by whole files (`True`) vs by rows (`False`). Default: `True`. |
| `--strategy` | `{concat,streaming}` | Row-split strategy (only when `keep_file_integrity=False`): `concat` (in-memory, faster) vs `streaming` (low-memory). Default: `concat`. |
| `--split_shuffle` | `bool` | Shuffle before row-splitting (only when `keep_file_integrity=False`). Default: `True`. |
| `--max_lines_per_file` | `int` | Cap lines per output file (row-split only). Default: `None`. |
| `--seed` | `int` | Seed for reproducible splits. Default: `42`. |
| `--use_portfolio_encoder` | `bool` | Parse with `PortfolioDataEncoder`. Default: `False`. |
| `--include_proportion` | `bool` | Holding data includes proportions (only with `use_portfolio_encoder=True`). |
| `--id_key` / `--portfolio_key` / `--proportion1_key` / `--proportion2_key` | `str` | Output keys. Defaults: `InvestorID` / `Portfolio` / `Proportion1` / `Proportion2`. |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | Console DEBUG / log-file path. |

## `data.text_embedding` — LLM text embeddings

Encodes CSMAR firm names (zh/en) into asset embeddings via OpenAI / Cohere / Voyage / Gemini / local HF (Qwen, BGE), with native + PCA (universal or per-period) reduction and a resumable cache. Provider rate-limit defaults and output layout: [Text embeddings](../how-to/text-embeddings.md).

| Parameter | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | JSON config; its keys become argparse defaults (hyphen and underscore keys both accepted), explicit CLI flags still override. |
| `--mode` | `{names,full}` | `names` runs the name-resolution stage only; `full` runs the full embedding pipeline. Default: `full`. |
| `--csmar-info` | `str` | Path to CSMAR `TRD_Co.csv`. **Required** (via CLI or `--config`). |
| `--valid-stocks-dir` | `str` | Directory of `{period}/valid_stocks.json` (from `data.sharehold`). **Required** (via CLI or `--config`). |
| `--periods` | `list[str]` | Subset of periods to process. Default: all periods under `--valid-stocks-dir`. |
| `--names-output` | `str` | Names-stage output dir (`stock_names.csv`, `missing_stocks.csv`). Default: `data/text_embeddings/names`. |
| `--cache-folder` | `str` | JSONL embedding-cache dir. Default: `data/text_embeddings/cache`. |
| `--output-folder` | `str` | Root for embedding artifacts. Default: `data/text_embeddings/embeddings`. |
| `--provider` | `{openai,cohere,voyage,gemini,local}` | Embedding provider. **Required for `--mode full`.** |
| `--model` | `str` | Provider-specific model id. **Required for `--mode full`.** |
| `--language` | `{zh,zh-short,en}` | `zh`=Conme, `zh-short`=Stknme, `en`=Conme_en. Default: `zh`. |
| `--transforms` | `{native,pca-universal,pca-perperiod}` | Dimensionality strategies to produce (combinable). Default: `native pca-universal`. |
| `--target-dims` | `list[int]` | Target embedding dimensions. Default: `4 8 10 16 32 64 128`. |
| `--seed` | `int` | PCA `random_state`. Default: `42`. |
| `--batch-size` | `int` | Texts per request (hosted) and GPU mini-batch (local); clamped to each provider's max. Default: `100`. |
| `--max-retries` | `int` | Max retries per batch. Default: `6`. |
| `--api-key` | `str` | Literal API key; highest precedence over `--api-key-env` and the provider-default env var. |
| `--api-key-env` | `str` | Env var name to read the key from (used only when `--api-key` is unset). |
| `--device` | `str` | Device for local HF models (ignored for hosted). Default: `cuda:0`. |
| `--hf-dtype` | `{bfloat16,float16,float32,int8,int4}` | Local model dtype; `int8`/`int4` not yet implemented (raise `NotImplementedError`). Default: `bfloat16`. |
| `--missing-name-fallback` | `{skip,akshare,error}` | Stocks in valid_stocks but missing from CSMAR (full mode): `skip` drops + warns, `error` aborts, `akshare` reserved. Default: `skip`. |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | Console DEBUG / log-file path. |

## Training (shared options) {#train-eval-shared}

Training (`train.rs`, `train.w2v`, `train.bert`) is config-driven: every model and run parameter lives in the JSON config. All three scripts expose the **same** five flags — there are no per-script options.

| Parameter | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | Path to the JSON config file. **Required.** |
| `--log`, `-l` | `str` (0+) | Log-file path(s). Default: no file logging. |
| `--result_file`, `-r` | `str` | SQLite path for structured training records. Default: `None`. |
| `--override`, `-o` | `str` (1+) | Patch config fields on the command line as `key=value`, dot notation for nested keys (e.g. `train.max_epoches=10 optimizer.learning_rate=0.001`). |
| `--verbose`, `-v` | flag | Console DEBUG output. Default: `INFO`. |

For the per-model **config-field** references, see [Train](../how-to/train.md).
