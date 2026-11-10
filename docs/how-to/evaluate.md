# Evaluation

> Part of [AssetEmbeddingsInChina](../index.md). See also [Installation](install.md), [Training](train.md).

All evaluation scripts accept `--result_file` (`-r`) pointing to a SQLite
database. Per-period scores and diagnostics are persisted via the structured
[records system](../explanation/records-system.md). Each run produces:

- One `runs` row — the config stored once, provenance (script, git commit,
  hostname), status, wall time and the `peak_ram_mb` / `peak_vram_mb` peaks.
- A per-task **result** row (`metric`, `value`, the penalty used, `n_scored`, `n_features`) —
  `eval_rv_results`, `eval_rc_results` or `eval_mp_results` (the grouped evaluations write one row per group and period, see
  [Grouped evaluation](#grouped-evaluation); the [label probe](#label-probe) and the [pairwise similarity](#pairwise-similarity) have tables of their own); the views
  `v_<table>` add the run's config fields (model type, features, scoring, ...) and `v_<table>_summary`
  the mean over periods.
- The **sample funnel** of every period (`eval_sample`): the rows of each file read and the rows left
  after each join and drop.
- For the MP scripts, one `run_epochs` row per refine epoch and one `train_summary` row.
- With `scoring: "holdout"`, one `eval_ridge_splits` row per period and firm split (see
  [Holdout evaluation](#holdout-evaluation)).

Results are saved incrementally per period, so partial runs are preserved if
the script is interrupted. See the [records system](../explanation/records-system.md)
for every table's full schema.

All evaluation entry points share the same six-flag CLI (also in the [CLI reference](../reference/cli.md#train-eval-shared)):

```bash
uv run python -m asset_embeddings.scripts.eval.<task> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--experiment EXPERIMENT] [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | Path to the JSON config file (examples below). **Required.** |
| `--log`, `-l` | `str` (0+) | Log-file path(s). Optional. |
| `--result_file`, `-r` | `str` | SQLite path for structured results (see above). Optional. |
| `--experiment`, `-e` | `str` | Experiment family stored in the `runs` table (default: derived from the config path, e.g. `configs/main/...` gives `main`). The runners pass `-e "$EXP"`. Optional. |
| `--override`, `-o` | `str` (1+) | Override config fields, `key=value` with dot notation. |
| `--verbose`, `-v` | flag | DEBUG console output (default `INFO`). |

## RV — Relative Valuations

`asset_embeddings.scripts.eval.rv` evaluates whether embedding vectors (e.g.
from AssetBERT, AssetW2V, AssetRS) can explain **relative valuation
differences** among firms. This corresponds to the first benchmark in the
paper: regressing company valuation residuals on asset embeddings.

$$
\begin{gathered}
b_{a,t}=\alpha_{0,t} + \alpha_{1,t} m_{a,t} + b_{a,t}^{\perp}\\
b_{a,t}^{\perp} = \gamma_{0,t} + \boldsymbol{\gamma}_{1,t}^T\boldsymbol{x}_{a,t} + \epsilon_{a,t}\\
\text{RV} = 1 - \frac{1}{T}\sum_{t=1}^{T}\frac{\text{Var}(b_{a,t}^{\perp} - \hat{\gamma}_{0,t} - \boldsymbol{\hat{\gamma}}_{1,t}^T\boldsymbol{x}_{a,t})}{\text{Var}(b_{a,t}^{\perp})}
\end{gathered}
$$

Here $b_{a,t}$ and $m_{a,t}$ are log book equity and log market equity (the `be` and `me` columns of the benchmark file). The target $b_{a,t}^{\perp}$ is the residual of the period's cross-sectional OLS of $\log(\text{be})$ on $\log(\text{me})$, and the score is the cross-sectional explanatory $R^2$ of the ridge fit of that residual on the embedding.

```bash
uv run python -m asset_embeddings.scripts.eval.rv -c config.json [-r results.db]
```

Outputs per-period RV plus the average.

### Config example

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2023Q3/AssetBERT_d4_2023Q3_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2023Q4/AssetBERT_d4_2023Q4_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q1/AssetBERT_d4_2024Q1_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q2/AssetBERT_d4_2024Q2_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q3/AssetBERT_d4_2024Q3_best_embedding.csv"
    ],
    "benchmark_path": [
        "data/processed/Benchmark/2023Q2.csv",
        "data/processed/Benchmark/2023Q3.csv",
        "data/processed/Benchmark/2023Q4.csv",
        "data/processed/Benchmark/2024Q1.csv",
        "data/processed/Benchmark/2024Q2.csv",
        "data/processed/Benchmark/2024Q3.csv"
    ],
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "features": ["embedding"],
    "align_chars": true,
    "ridge_alpha": 0.5,
    "seed": 42,
    "embedding_token_col": "Token",
    "embedding_value_pattern": "Embed_*",
    "benchmark_stock_col": "Stkcd",
    "normalization": "standard"
}
```

### Config fields

Every per-period path is a list with one file per entry of `periods`, in the same order (a single file is accepted when there is one period); the config refuses any other length. Files are never matched to periods by name, because which period a file belongs to differs between paths: the predictive setups read the period-t embedding next to the t+1 benchmark.

| Field            | Type                 | Description                                                  |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| `model_type`     | `str`                | Label of the model under evaluation, recorded with every result, e.g. `"AssetBERT"`. **Required.** |
| `periods`       | `list[str]`          | Period labels, in the order of the path lists. **Required.** |
| `embedding_path` | `list[str]` / `null` | One embedding file per period. Set exactly when `"embedding"` is in `features`. |
| `benchmark_path` | `list[str]`          | One benchmark file per period: the target columns, and the characteristics and `IndustryCode` unless `char_benchmark_path` is set. **Required.** |
| `char_benchmark_path` | `list[str]` / `null` | Optional, one file per period supplying the characteristics and `IndustryCode` instead of `benchmark_path`, joined on the stock ID. `null` (default): they come from `benchmark_path`. |
| `features`       | `list[str]`          | Feature blocks, in column order; any non-empty combination of `"embedding"` (the columns matching `embedding_value_pattern`), `"placebo"` (four columns drawn uniformly on [-1, 1] per firm), `"chars"` (the characteristics) and `"industry"` (industry dummies, `drop_first=True`). Without `"embedding"` the evaluation runs on the firms of the benchmark file: the full available sample, restricted only by the columns it reads, so a baseline's sample can differ from the rows of an embedding evaluation; the recorded `n_scored` and the `eval_sample` funnel show the sample each fit used. RV's characteristics are beta, asset growth, profitability and dividend-to-asset (`beta_60m`, `at_gr1`, `gp_at`, `div_at`); RC uses its own set (listed under RC below). Default: `["embedding"]`. |
| `align_chars`    | `bool`               | Read the characteristics even when `"chars"` is not a feature, so that firms missing any of them are dropped and the sample matches the one with `"chars"` added. With `"chars"` in `features` the firms are restricted either way. `false`: every firm with the other inputs enters. Recorded as the `align_chars` config column of the result views. Default: `true`. |
| `ridge_alpha`    | `float` / `null`     | Ridge regularization strength; `null` selects it by 10-fold cross-validated grid search maximizing $R^2$. Default: `0.5`. |
| `scoring`        | `str`                | `"cross_section"` fits and scores on every firm of the period (the setting of the main results); `"holdout"` scores on Koijen-style firm splits, see [Holdout evaluation](#holdout-evaluation). Default: `"cross_section"`. |
| `holdout_test_size` | `float`           | Held-out share of firms per split under holdout scoring, realized as `round(1 / holdout_test_size)` hash folds with fold 0 held out. Default: `0.2`. |
| `holdout_n_splits` | `int`              | Number of firm splits per period under holdout scoring; split `s` uses seed `holdout_seed + s`. Default: `20`. |
| `holdout_seed`   | `int`                | Base seed of the firm splits under holdout scoring (independent of `seed`, which drives the cross-validation folds and the placebo draw). Default: `42`. |
| `holdout_split_aggregate` | `str`     | How a period's value is aggregated over its firm splits under holdout scoring: `"median"` or `"mean"`. Aggregation across periods is always the mean. Default: `"median"`. |
| `nonpositive_equity` | `str`            | RV only. Firms with non-positive book or market equity in the RV target: `"drop"` (default) removes them from the period before the `log(be) ~ log(me)` residual and the ridge; `"keep"` is the earlier behavior, retained so results recorded under it can be reproduced (such a firm's residual is of order 1e9 and dominates any fit that touches it). |
| `seed`           | `int`                | Seed of every random draw: the placebo columns (one generator per period, from the seed and the period label), the cross-validation folds of the penalty search, the bootstrap and the permutations. Default: `42`. |
| `embedding_token_col`     | `str`       | Name of the token (stock ID) column in the embedding files. Default: `"Token"`. |
| `embedding_value_pattern` | `str`       | Wildcard pattern of the embedding value columns. Default: `"Embed*"`, which matches the `Embed_1 … Embed_N` columns the trainers write; text embeddings with `Emb_0 …` columns need `"Emb_*"`. |
| `benchmark_stock_col`     | `str`       | Name of the stock ID column in the benchmark files. Default: `"Stkcd"`. |
| `normalization`  | `str`                | Feature scaling: `"standard"` (StandardScaler), `"minmax"` (MinMaxScaler) or `"none"`. Default: `"standard"`. |

## RC — Return Comovement

`asset_embeddings.scripts.eval.rc` evaluates whether asset embeddings explain
future return comovement structure — a core test of whether embeddings capture
the **systematic risk factor structure** of assets.

$$
\begin{gathered}
r_{a,t+1} = c_{t} + \boldsymbol{f}_t^T\boldsymbol{x}_{a,t} + \epsilon_{a,t}\\
\text{RC} = 1 - \frac{1}{T}\sum_{t=1}^{T}\frac{\text{Var}(r_{a,t+1} - \hat{c}_{t} - \boldsymbol{\hat{f}}_t^T\boldsymbol{x}_{a,t})}{\text{Var}(r_{a,t+1})}
\end{gathered}
$$

```bash
uv run python -m asset_embeddings.scripts.eval.rc -c config.json [-r results.db]
```

Config schema is identical to RV (see above), except that `nonpositive_equity` does not exist: the RC target is the next-quarter return, not a valuation residual. The characteristic set also differs: `"chars"` and `align_chars` use beta, size, book-to-market, asset growth, profitability and momentum (`beta_60m`, `me`, `be_me`, `at_gr1`, `gp_at`, `ret_12_1`).

## MP — Managed Portfolios

The MP score measures the **model's ability to predict masked assets at fixed
positions** in investor portfolios, reflecting how well embeddings capture
portfolio-construction preferences.

$$
MP = 1 - \frac{\sum_{b_i(k)\in\mathcal{B}}\ln (\text{Pr}(b_i(k)=a_i(k) | \mathcal{A}_i(k)))}{-\lvert\mathcal{B}\rvert \ln V}
$$

Where $\mathcal{B}$ is the masked-asset evaluation set, $\mathcal{A}_i(k)$ is investor $i$'s holdings with position $k$ masked, $a_i(k)$ is the asset originally held by investor $i$ at position $k$, and $V$ is the size of the tokenizer vocabulary. Each masked token scores $1 + \ln p / \ln V$, so a uniform guess over the whole vocabulary scores 0 and certainty scores 1; the reported score is the mean over the masked tokens of the test split. The metric ranges over $(-\infty, 1]$, with values closer to 1 indicating better prediction.

Because W2V uses a fixed window, it is not natively suited for this fixed-position MLM task. For RS and W2V, the embedding is scored through a simple feedforward head (SimplifiedMLM); the softmax runs over the whole vocabulary $\mathcal{V}$ (of size $V$), not only over the assets the investor does not hold:

$$
\text{Pr}(b_i(k)=a | \mathcal{A_i}(k)) = \frac{\exp(\boldsymbol{\lambda}_i^T\boldsymbol{x}_a+\delta_a)}{\sum_{b \in \mathcal{V}}\exp(\boldsymbol{\lambda}_i^T\boldsymbol{x}_b+\delta_b)}
$$

To handle time-varying market structure, MP evaluation refines a model per period on that period's holdings, split into train, validation and test sets. For BERT, the refine starts from the period's finetuned AssetBERT checkpoint (`model.model_checkpoint`, loaded with its encoder and prediction head) and its embedding file; the main grid freezes the embedding layer (`freeze_embedding: true`) and trains the encoder and head. For W2V and RS, a SimplifiedMLM head is trained on the period's embedding file with the embedding frozen.

End-to-end train-val-test scripts are provided for SimplifiedMLM and BERT.

Both read their `train` block as `MPTrainConfig`: the loop fields of the [training config](train.md) plus `test_split`, without the pretraining-only `calculate_contextualized_embeddings` and `keep_best_only`. `best_metric` and `early_stop_metric` take `"train_loss"`, `"val_loss"` or `"test_loss"`; `"val_loss"` needs `validation_split > 0` and `"test_loss"` needs `test_split > 0`, checked when the config loads. Each epoch scores the validation split when there is one, and the test split when a metric reads it. The main-grid MP templates run without a validation split and select the epoch on `"test_loss"`.

> Because MP introduces an extra training procedure, it takes substantially
> longer than RV or RC.

### Pretrain baseline

Both evaluation scripts call `prepare()` and then `train()`, which first scores the original model (loaded from `model_checkpoint`) on the test set before any refining, recording the baseline in the `baseline_value` / `baseline_value_std` columns of `eval_mp_results`. It then refines, saving the `accelerate` state to `<save_name>_accelerator_best` whenever the selection loss improves, reloads that state and scores the test set again, enabling a direct comparison before and after refining. The per-epoch states are deleted at the end; the best state stays.

### SimplifiedMLM (RS, W2V)

```bash
uv run python -m asset_embeddings.scripts.eval.mp_smlm -c config.json [-r results.db]
```

#### Config example

```json
{
    "run": {
        "period": "2023Q2"
    },
    "model": {
        "model_type": "SimplifiedMLM",
        "model_checkpoint": null,
        "embedding_file": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2_embedding.csv",
        "vocab_size": 4940,
        "hidden_size": 4,
        "max_position_embeddings": 256,
        "type_vocab_size": 1,
        "freeze_embedding": true,
        "freeze_head": false,
        "embedding_model": "AssetW2V"
    },
    "tokenizer": {
        "w2v_model": null,
        "embedding_file": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2_embedding.csv",
        "vocab_file": null,
        "pretrained_tokenizer_file": null,
        "alias_file": null
    },
    "dataset": {
        "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
        "data_format": "csv",
        "id_key": "InvestorID",
        "portfolio_key": "Portfolio",
        "proportion1_key": "Proportion1",
        "proportion2_key": "Proportion2",
        "include_proportion": false,
        "max_length": 256,
        "mask_prob": null,
        "mask_indices": [2],
        "num_repeats": 1
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
        "pin_memory": true,
        "drop_last": false
    },
    "optimizer": {
        "optimizer_type": "PagedAdam8bit",
        "optimizer_kwargs": null,
        "learning_rate": 0.001,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_train_steps": null,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 0
    },
    "accelerator": {
        "mixed_precision": "auto",
        "gradient_accumulation_steps": 1,
        "accelerator_checkpoint": null,
        "use_compile": false,
        "compile_backend": "inductor",
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": false,
        "compile_dynamic": false,
        "compile_cache_dir": null,
        "allow_tf32": true
    },
    "train": {
        "max_epochs": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/refine/SMLM/AssetW2V/d4/2023Q2",
        "save_name": "SMLM_AssetW2V_d4_2023Q2",
        "save_format": ".safetensors",
        "seed": 42,
        "early_stop_enabled": false,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.0,
        "early_stop_mode": "min",
        "early_stop_relative": true,
        "early_stop_metric": "val_loss"
    }
}
```

The `tokenizer`, `dataset`, `dataloader`, `optimizer` and `accelerator` blocks are those of AssetBERT training. The `train` block is `MPTrainConfig`: AssetBERT's `train` block without `calculate_contextualized_embeddings` and `keep_best_only`. The `run` and `model` blocks are:

| Field                     | Type   | Description                                  |
| ------------------------- | ------ | -------------------------------------------- |
| `run.period`             | `str`  | Period label written to the `eval_mp_results` row, e.g. `"2023Q2"`. **Required.** |
| `model_type`              | `str`  | Fixed as `"SimplifiedMLM"`. |
| `model_checkpoint`        | `str`  | Optional pretrained path. |
| `embedding_file`          | `str`  | Asset embedding CSV. |
| `vocab_size`              | `int`  | Number of stock types. |
| `hidden_size`             | `int`  | Hidden dim. |
| `max_position_embeddings` | `int`  | Max holding-sequence length. |
| `freeze_embedding`        | `bool` | Freeze input embedding layer. |
| `freeze_head`             | `bool` | Freeze LM head. |
| `embedding_model`         | `str`  | Name of the upstream embedding model, e.g. `"AssetW2V"` or `"AssetRS_Binary"`; written to `eval_mp_results.model_type`, a required column, so a run without it fails when the result is saved. |

### AssetBERT

```bash
uv run python -m asset_embeddings.scripts.eval.mp_bert -c config.json [-r results.db]
```

#### Config example

```json
{
    "run": {
        "period": "2023Q2"
    },
    "model": {
        "model_type": "BERT",
        "model_checkpoint": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best/model.safetensors",
        "embedding_file": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
        "vocab_size": 5093,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 256,
        "type_vocab_size": 1,
        "freeze_embedding": true,
        "freeze_encoder": false,
        "embedding_model": "AssetBERT"
    },
    "tokenizer": {
        "w2v_model": null,
        "embedding_file": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
        "vocab_file": null,
        "pretrained_tokenizer_file": null,
        "alias_file": null
    },
    "dataset": {
        "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
        "data_format": "csv",
        "id_key": "InvestorID",
        "portfolio_key": "Portfolio",
        "proportion1_key": "Proportion1",
        "proportion2_key": "Proportion2",
        "include_proportion": false,
        "max_length": 256,
        "mask_prob": null,
        "mask_indices": [2],
        "num_repeats": 1
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
        "pin_memory": true,
        "drop_last": false
    },
    "optimizer": {
        "optimizer_type": "PagedAdam8bit",
        "optimizer_kwargs": null,
        "learning_rate": 0.001,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_train_steps": null,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 0
    },
    "accelerator": {
        "mixed_precision": "auto",
        "gradient_accumulation_steps": 1,
        "accelerator_checkpoint": null,
        "use_compile": false,
        "compile_backend": "inductor",
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": false,
        "compile_dynamic": false,
        "compile_cache_dir": null,
        "allow_tf32": true
    },
    "train": {
        "max_epochs": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/refine/BERT/AssetBERT/d4/2023Q2",
        "save_name": "AssetBERT_d4_2023Q2",
        "save_format": ".safetensors",
        "seed": 42,
        "early_stop_enabled": false,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.0,
        "early_stop_mode": "min",
        "early_stop_relative": true,
        "early_stop_metric": "val_loss"
    }
}
```

The `tokenizer`, `dataset`, `dataloader`, `optimizer` and `accelerator` blocks are those of AssetBERT training. The `train` block is `MPTrainConfig`, as for SimplifiedMLM. The `run` and `model` blocks are:

| Field                     | Type   | Description                                  |
| ------------------------- | ------ | -------------------------------------------- |
| `run.period`             | `str`  | Period label written to the `eval_mp_results` row, e.g. `"2023Q2"`. **Required.** |
| `model_type`              | `str`  | Fixed as `"BERT"`. |
| `model_checkpoint`        | `str`  | Checkpoint the refine starts from (the period's finetuned AssetBERT `model.safetensors`). Optional. |
| `w2v_model`               | `str`  | Optional Word2Vec `.model` whose vectors initialize the token embeddings. |
| `embedding_file`          | `str`  | Asset embedding CSV. |
| `vocab_size`              | `int`  | Number of stock types (overwritten by the tokenizer's size). |
| `hidden_size`             | `int`  | Hidden dim. |
| `num_hidden_layers`       | `int`  | Transformer layers. Default: `4`. |
| `num_attention_heads`     | `int`  | Attention heads. Default: `2`. |
| `intermediate_size`       | `int`  | Feed-forward width. Default: `512`. |
| `max_position_embeddings` | `int`  | Max holding-sequence length. |
| `type_vocab_size`         | `int`  | Token-type vocabulary size. Default: `1`. |
| `freeze_embedding`        | `bool` | Freeze input embedding layer. |
| `freeze_encoder`          | `bool` | Freeze the Transformer encoder. |
| `embedding_model`         | `str`  | Name of the upstream embedding model, e.g. `"AssetBERT"`; written to `eval_mp_results.model_type`, a required column, so a run without it fails when the result is saved. |

## Holdout evaluation

By default RV and RC fit and score the ridge on every firm of the period: RV reports a cross-sectional explanatory $R^2$ and RC a cross-sectional predictive $R^2$ (its target is the next-quarter return). Setting `scoring: "holdout"` switches either task to Koijen-style firm splits. Per split $s$ (seed `holdout_seed + s`) the firms are hashed into `round(1 / holdout_test_size)` folds; fold 0 is the test set and the rest the training set. The penalty is `ridge_alpha` when set, otherwise a 10-fold `RidgeCV` on the training rows selects it. A `Ridge` fitted on the training rows predicts the held-out firms, and the split's Koijen $R^2$ is $1 - \text{Var}(\text{resid}) / \text{Var}(y_{\text{test}})$; the sklearn definition $1 - SS_{\text{res}} / SS_{\text{tot}}$ is recorded alongside. The period's score is the median over the `holdout_n_splits` splits (`holdout_split_aggregate: "mean"` takes the mean) and the reported `ridge_alpha` is the median of the per-split penalties (the fixed value itself under a fixed penalty). A split with fewer than 10 training rows, an empty test set or zero test-target variance raises instead of being skipped, so the score is always taken over the recorded set of firms. The median is the default because a held-out $R^2$ is bounded above but not below: one outlying firm in a test fold can drive a split far below zero, and the mean with it.

What such a run writes: the result row carries `metric = "r2_holdout"`, the split aggregate as `value` and the median split penalty as `ridge_alpha`; one `eval_ridge_splits` row per period and split (`value`, `value_sklearn`, `alpha`, `n_train`, `n_test`) is read back with `load_splits`. The spread over the splits, or the other aggregate, is a query on those rows. Runs written before `holdout_split_aggregate` existed hold the mean; their `holdout_split_aggregate` view column is NULL. See [Holdout splits](../explanation/records-system.md#holdout-splits-eval_ridge_splits).

The `holdout` experiment family runs the main RV/RC grid this way. `templates/eval_holdout_main.json` generates `configs/holdout/eval/{fixed,cv}/{RV,RC}/...` — the placebo and characteristics baselines, six models × seven dimensions × `features` `["embedding"]` / `["embedding", "chars"]`, and the full-universe variants (`align_chars: false`) — with `holdout_test_size: 0.2`, `holdout_n_splits: 20`, `holdout_seed: 42`, in two penalty tiers: `fixed` (`ridge_alpha: 0.5`, the in-sample value; seconds per config) and `cv` (`ridge_alpha: null`, a 10-fold `RidgeCV` inside each training split; about 20× the cost). `scripts/run/eval_holdout_main.sh` runs both tiers into `results/db/holdout_main.db` with `runs.experiment = "holdout"`; the `fixed_alpha` config column of the views tells the tiers apart. The jobs are tagged `rv` / `rc`, `holdout`, `fixed` / `cv`, the model name, `addchars`, `full` and `d{dim}`, so the CV tier can be sharded by model:

```bash
bash scripts/run/run_notify.sh scripts/run/eval_holdout_main.sh
bash scripts/run/run_notify.sh -k fixed -k cv,chars -k cv,placebo scripts/run/eval_holdout_main.sh
bash scripts/run/run_notify.sh -k cv,assetbert scripts/run/eval_holdout_main.sh
```

## Temporal RC

`asset_embeddings.scripts.eval.rc_temporal` asks whether the RC mapping persists across periods. For each period $T$ after the first, the ridge of period $T-1$'s target (the period-$T$ return) on $T-1$'s features gives a slope $\hat{\boldsymbol{f}}_{T-1}$; that slope is applied unchanged to period $T$'s features and scored against period $T$'s target (the period-$T+1$ return) with the Koijen $R^2$:

$$
\text{RC}^{\text{temporal}}_T = 1 - \frac{\text{Var}(r_{a,T+1} - \boldsymbol{\hat{f}}_{T-1}^T\boldsymbol{x}_{a,T})}{\text{Var}(r_{a,T+1})}
$$

with population variances. Under `slope_window: "expanding"` the slopes of every earlier period $s < T$ are applied instead and their predictions averaged, so that in a fixed frame $\hat{\boldsymbol{f}}_{T-1}$ is replaced by the mean slope $\frac{1}{T-1}\sum_{s<T}\hat{\boldsymbol{f}}_s$, the Fama–MacBeth estimate of the premia. The residual is demeaned, so the intercept is free in every period: it absorbs period $T$'s common return, which a cross-sectional $R^2$ does not credit in any case. Only the slope is carried over. In the cross-sectional RC the slope (the factor realizations $\boldsymbol{f}_t$) is re-estimated every period; holding it fixed tests whether those realizations persist, which is an out-of-period question, and it needs embeddings whose coordinates mean the same thing in both periods. Because an unrelated slope adds prediction variance, a negative value means no predictive power, not an error.

```bash
uv run python -m asset_embeddings.scripts.eval.rc_temporal -c config.json [-r results.db]
```

The config is the RC config (see above) with two more fields, `align` and `slope_window`:

| `align` | Period $T$'s embedding |
|---|---|
| `"none"` (default) | Enters in its own coordinates. |
| `"procrustes"` | Rotated onto period $T-1$'s frame before normalization: over the firms present in both periods, each period's raw embedding is centered on those firms and the orthogonal $R$ minimizing $\lVert X_T R - X_{T-1}\rVert_F$ is solved (rotation and reflection, no scaling); every firm of period $T$ is then multiplied by $R$. Only embeddings enter this step, no return. |

| `slope_window` | Slopes applied to period $T$ |
|---|---|
| `"previous"` (default) | The slope fitted on period $T-1$. |
| `"expanding"` | The slopes fitted on every earlier period of the config, each with its own penalty; their predictions are averaged. Under `align: "procrustes"` period $T$ is rotated onto each earlier period directly, over the firms common to the two, and each slope is applied in its own frame. The first scored period has one earlier slope, so both choices agree there. |

Alignment comes before the per-period normalization because column standardization does not commute with a rotation unless the columns' covariance is isotropic. Each period is then normalized on its own full sample with `normalization`, as in RC. The penalty is `ridge_alpha` (fixed tier), or with `ridge_alpha: null` a 10-fold CV over the firms of each fitted period with folds seeded by `seed` (CV tier). Restrictions, each checked when the config is loaded:

- `features` is `["embedding"]` or `["chars"]` (industry dummies can differ between periods, placebo columns are drawn afresh every period); `align: "procrustes"` requires `["embedding"]`.
- `scoring` is `"cross_section"`.
- `periods` are consecutive quarters `YYYYQn` in increasing order, so that no earlier period's target overlaps period $T$'s.

At run time the feature columns of consecutive periods must be identical (this covers different embedding dimensions), and under alignment the common firms must determine the rotation (at least as many as the dimension, full-rank cross-product); otherwise the run raises. The number of firms each slope is fitted on, the periods whose slopes score period $T$ and, under alignment, the number of common firms and the smallest singular value of the cross-product are logged at INFO.

What a run writes: the first period has no predecessor, so it gets its sample funnel but no result row. Every later period writes one `eval_rc_results` row under task `RC_temporal`: `metric` is `r2_temporal` (raw) or `r2_temporal_aligned`, `ridge_alpha` is the mean penalty of the slopes applied (under `"previous"` the penalty of $T-1$), `n_scored` the firms of period $T$. The views carry `slope_window` as a config column; runs written before the field read NULL there and used the previous period. The run's mean is over the scored periods. `load_results(db, "RC_temporal")` reads only these runs; `load_results(db, "RC")` reads only the cross-sectional and holdout ones.

The `rc_temporal` family runs both grids: `templates/eval_rc_temporal.json` generates `configs/rc_temporal/{main,timesplit}/{previous,expanding}/{fixed,cv}/...`: per window, slope window and tier six models × seven dimensions × `raw` / `aligned` on the finetuned embeddings, the characteristics baseline, and the pretrained BERT and Word2Vec embeddings without finetuning (`AssetBERT_static`, `AssetW2V_static`: the same embedding in every period, `raw` only) at seven dimensions; 99 configs each, 792 in all, on `main` (2023Q2–2024Q3) and `timesplit_robust` (2020Q4–2024Q3). `scripts/run/eval_rc_temporal.sh` writes them to `results/db/rc_temporal.db` with `runs.experiment = "rc_temporal"`; the jobs are tagged `rc_temporal`, `main` / `timesplit`, `previous` / `expanding`, `fixed` / `cv`, `raw` / `aligned`, `chars`, `static`, the model name and `d{dim}`.

## Grouped evaluation

`asset_embeddings.scripts.eval.rv_grouped` and `asset_embeddings.scripts.eval.rc_grouped` repeat the RV / RC ridge fit per group and period, driving `GroupedRVEvaluator` / `GroupedRCEvaluator` (`asset_embeddings.evaluators.grouped`). Firms are grouped by a value of a per-period group file: institutional coverage in §7.4 (how many funds hold a firm, a `coverage_count` column, grouped into quantiles or fixed bins) or state ownership in §7.1 (an `SOE` column of 1 for state-owned and 0 for private firms, one group per value under `grouping: "categorical"`). The score of each (group, period) cell is bootstrapped for a confidence interval, and permutation tests compare the groups. Each cell is one row of `eval_grouped_rv_results` / `eval_grouped_rc_results` (key `period`, `group_label`); each test is one row of `eval_grouped_rv_tests` / `eval_grouped_rc_tests`; the period's funnel ends with the group file.

```bash
uv run python -m asset_embeddings.scripts.eval.rv_grouped -c config.json [-r results.db]
uv run python -m asset_embeddings.scripts.eval.rc_grouped -c config.json [-r results.db]
```

`GroupedRVEvalConfig` / `GroupedRCEvalConfig` (`asset_embeddings.configs.eval_configs`) extend the RV / RC configs with the grouping fields below, declared once on `GroupingConfig`. A set `char_benchmark_path` is refused by the config (the characteristics come from `benchmark_path`). A (group, period) cell with fewer than 10 stocks, or fewer than 10 rows after the merge, is skipped with a warning. Two experiment families run these evaluations. `coverage`: `templates/eval_coverage.json` generates `configs/coverage/eval/...`, and `scripts/run/eval_coverage.sh` runs them into `results/db/coverage.db`, choosing the grouping at run time with `-o grouping=quantile` or `-o grouping=fixed bins=...`. `state_ownership`: `templates/eval_state_ownership.json` generates `configs/state_ownership/eval/grouped/{RV,RC}/...` with `grouping: "categorical"` on `SOE`, and `scripts/run/eval_state_ownership.sh` runs them, with the [label probe](#label-probe) and the [pairwise similarity](#pairwise-similarity), into `results/db/state_ownership.db`.

The groups are assigned on the group file's own universe (every firm it lists), not on the firms of the frame, and the file is joined as a lookup that keeps every row, since the penalty chosen on the whole cross section (`alpha_scope: "cross_section"`) needs every row. Under `grouping: "categorical"` each distinct value is a group: an integer-valued number is labeled without its decimal (`0`, `1`), the groups are ordered by value, and the per-group statistics of the value (`group_value_mean`, `_median`, `_min`, `_max`) are null, since every firm of a group holds the same value. The groups of a period are those its file holds; the permutation tests run over the groups that have scores.

Under `scoring: "holdout"` a cell's value is the held-out R² aggregated over the firm splits by `holdout_split_aggregate` (metric `r2_holdout`), with one penalty per split: selected on the split's training firms of the cell (`alpha_scope: "cell"`) or of the whole period (`"cross_section"`), and recorded as their median. Every split of a scored cell is one `eval_grouped_ridge_splits` row (key `period`, `group_label`, `split`; read back with `load_grouped_splits`). The interval is then a bootstrap over the cell's firms of that split aggregate: each resample is rescored at the cell's penalties, and the copies of a firm drawn twice fall in the same fold, since the folds hash the stock id. A cell whose splits cannot be fitted or scored (fewer than 10 training firms, no test firm, or a constant target on a split's test firms) is skipped with a warning; a resample that cannot be scored (no test firm in a split, or a constant target on its test firms, which small cells meet often) is left out of the interval, and the log gives how many were. Every resample rescores all splits, so the bootstrap costs about `holdout_n_splits` ridge fits per resample. The permutation tests compare adjacent groups, then the first group against the last when there are more than two groups; each p-value is recorded with its statistic.

| Field                 | Type          | Description                                                  |
| --------------------- | ------------- | ------------------------------------------------------------ |
| `group_path`          | `list[str]`   | One group file per period: the stock ID and the value firms are grouped by (`stock_coverage.csv` for coverage, the SOE label files for state ownership). A file that lists a stock twice is refused. **Required.** |
| `group_col`           | `str`         | Column of the group file that defines the groups, e.g. `"coverage_count"` or `"SOE"`. **Required.** |
| `group_stock_col`     | `str`         | Stock ID column of the group file (the coverage files name it `"Symbol"`); ids are zero-padded to six digits. Default: `"Stkcd"`. |
| `n_quantiles`         | `int`         | Number of quantile groups under `grouping: "quantile"` (`G1` = lowest value). Default: `4`. |
| `n_bootstrap`         | `int`         | Number of bootstrap resamples for the CI (`0` skips the bootstrap; the CI is recorded as null). Default: `1000`. |
| `n_permutation`       | `int`         | Number of permutation-test iterations. Default: `1000`. |
| `grouping`            | `str`         | `"quantile"` (per-period quantiles of the value), `"fixed"` (the absolute groups of `bins`) or `"categorical"` (one group per distinct value; `bins` and `bin_labels` must be unset). Default: `"quantile"`. |
| `bins`                | `list` / `null` | Under `grouping: "fixed"`: inner cut points (finite, strictly increasing), e.g. `[1, 3, 10]` gives the groups `<=1` / `2-3` / `4-10` / `>10`. |
| `bin_labels`          | `list` / `null` | Labels of the fixed groups; the length must be `len(bins) + 1`. `null` derives the labels from `bins`. |
| `winsor_pct`          | `float` / `null` | Two-sided winsorization percentile of the target (the OLS valuation residual for RV, the next-quarter return for RC): clip at `[p, 100-p]` before the ridge, taming heavy tails that would otherwise saturate the CV penalty. `null` = no winsorization. |
| `cv_alpha_min`        | `float` / `null` | Lower bound of the CV ridge-alpha grid, so the well-covered groups are not under-regularized. `null` = no floor. Applies to RV and RC. |
| `alpha_scope`         | `str`         | Scope of the CV ridge-alpha resolution: `"cell"` resolves alpha within each (group, period) cell; `"cross_section"` resolves it once per period on the full cross-section and applies that penalty to every group (used by the characteristics baseline, whose per-group subsamples are too small for a stable per-cell CV). No effect when `ridge_alpha` is fixed. Default: `"cell"`. |

## Label probe

`asset_embeddings.scripts.eval.label_probe` asks how well a binary firm label (state ownership in §7.1) can be recovered from the features, driving `LabelProbeEvaluator` (`asset_embeddings.evaluators.label_probe`). Per period the label file is inner-joined onto the period's frame, so unlabeled firms leave the sample. The features are scaled on the whole cross section as in RV and RC, split into `n_folds` stratified folds (`StratifiedKFold(n_folds, shuffle=True, random_state=seed)`), and a logistic regression (`LogisticRegression(max_iter=1000)`, L2, C = 1) fitted on the training folds is scored by the ROC AUC of the test fold. The period's value is the mean test-fold AUC. A fold whose test or training part holds a single class is skipped. A period whose label holds a single class, that has fewer labeled firms than folds, or none of whose folds can be scored writes no result row; its funnel is still recorded and a warning names the reason. A label with values other than 0 and 1 stops the run with an error.

```bash
uv run python -m asset_embeddings.scripts.eval.label_probe -c config.json [-r results.db]
```

`LabelProbeEvalConfig` (`asset_embeddings.configs.eval_configs`) combines the data and feature fields of RV (the paths, `periods`, `features`, `align_chars`, `char_benchmark_path`, `normalization`, `seed` and the column names, as in the [RV config fields](#config-fields)) with the label fields below; it has no penalty and no scoring fields. The characteristics are RV's four (beta, asset growth, profitability, dividend-to-asset), so `features: ["chars"]` without `embedding_path` is the characteristics baseline: the benchmark firms with a label and complete characteristics.

| Field             | Type        | Description |
| ----------------- | ----------- | ----------- |
| `label_path`      | `list[str]` | One label file per period: the stock ID and the label column, one row per stock (a stock listed twice is refused). **Required.** |
| `label_col`       | `str`       | The label column of the label files, e.g. `"SOE"`. **Required.** |
| `label_stock_col` | `str`       | Stock ID column of the label files; ids are zero-padded to six digits. Default: `"Stkcd"`. |
| `n_folds`         | `int`       | Folds of the stratified split per period (2 to 20); each fold's test AUC is scored and the period's value is their mean. Default: `5`. |

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d128/2023Q2/AssetBERT_d128_2023Q2_best_embedding.csv",
        "..."
    ],
    "benchmark_path": ["data/processed/Benchmark/2023Q2.csv", "..."],
    "label_path": ["data/processed/FirmLabels/SOE/2023Q2.csv", "..."],
    "label_col": "SOE",
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "features": ["embedding"],
    "align_chars": false,
    "normalization": "standard",
    "embedding_value_pattern": "Embed_*",
    "n_folds": 5,
    "seed": 42
}
```

Each scored period writes one `eval_probe_results` row (task `PROBE`, metric `auc`, `value` the mean test-fold AUC, `n_scored` the labeled firms, `n_positive` those with label 1, `n_features`, `n_folds_used`) and one `eval_probe_folds` row per scored fold (`fold`, `auc`, `n_train`, `n_test`, `n_test_positive`). The funnel ends with the label steps `label_file` (labeled rows of the file) and `label_joined` (rows left by the join). `load_results(db, "PROBE")` reads the results with the config columns (`model_type`, `features`, `align_chars`, `label_col`, `n_folds`, ...); `load_details(db, ProbeFold)` the folds.

## Pairwise similarity

`asset_embeddings.scripts.eval.pairwise_similarity` asks how much closer two firms' embeddings are when the firms share a label or an industry, driving `PairwiseSimilarityEvaluator` (`asset_embeddings.evaluators.pairwise_similarity`). Per period, the cosine similarity of the raw embedding vectors (no feature selection, no scaling) of every unordered firm pair is regressed by OLS on a constant and one indicator per `match_on` column, 1 when the two firms share the value; `interaction: true` adds the product of every subset of two or more indicators. Every regressor is a function of the pair's indicator pattern, so the fit is solved from the count, sum and sum of squares of the similarity in each of the `2**k` patterns and the pair-level design is never built (a period of 4,700 firms has 11 million pairs and needs well under 1 GB). After the last period the per-period coefficients are aggregated Fama-MacBeth style: each term's value is their time-series mean, the standard error their standard deviation over the square root of the number of periods, and the p-value two-sided from a t distribution with one degree of freedom fewer than periods. A design that cannot be solved, for example when every firm shares the label in a period, stops the run with an error naming the period.

```bash
uv run python -m asset_embeddings.scripts.eval.pairwise_similarity -c config.json [-r results.db]
```

`PairwiseSimilarityEvalConfig` combines the data fields of RV (the paths, `periods`, `seed` and the column names) with the label fields and the two fields below; it has no feature fields, since the regression runs on the raw embedding, and `embedding_path` is required. The sample is every firm with an embedding, a label and a value in each `match_on` column: the `match_on` columns other than the label are read from the benchmark file, and a firm missing one of them leaves the sample.

| Field             | Type        | Description |
| ----------------- | ----------- | ----------- |
| `label_path`      | `list[str]` | One label file per period: the stock ID and the label column, one row per stock (a stock listed twice is refused). **Required.** |
| `label_col`       | `str`       | The label column of the label files, e.g. `"SOE"`. **Required.** |
| `label_stock_col` | `str`       | Stock ID column of the label files; ids are zero-padded to six digits. Default: `"Stkcd"`. |
| `match_on`        | `list[str]` | Columns whose shared value makes a same-value indicator of a firm pair: the label column or benchmark columns such as `IndustryCode`; 1 to 4 distinct names, in coefficient order. **Required.** |
| `interaction`     | `bool`      | Add the product of every subset of two or more `match_on` indicators as a regressor. Default: `false`. |

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d128/2023Q2/AssetBERT_d128_2023Q2_best_embedding.csv",
        "..."
    ],
    "benchmark_path": ["data/processed/Benchmark/2023Q2.csv", "..."],
    "label_path": ["data/processed/FirmLabels/SOE/2023Q2.csv", "..."],
    "label_col": "SOE",
    "match_on": ["SOE", "IndustryCode"],
    "interaction": false,
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "embedding_value_pattern": "Embed_*",
    "seed": 42
}
```

A run writes three tables, nothing repeated between them: one `eval_pairwise_fits` row per period (`r2`, `n_pairs`, `mean_sim` the mean similarity of all pairs, `n_firms`), one `eval_pairwise_coefs` row per period and term (`term` is `const`, a `match_on` column or an interaction such as `SOE:IndustryCode`; `coef`), and after the last period one `eval_pairwise_results` row per term (task `PAIRWISE`, metric `fm_coef`, `value` the Fama-MacBeth mean, `se`, `t`, `p`, `n_periods`). The funnel ends with the label steps as in the probe. A premium relative to the average similarity divides a term's value by the mean of the per-period `mean_sim`:

```python
from asset_embeddings.records import PairwiseFit, load_details, load_results

results = load_results(db, "PAIRWISE")  # one row per run and term, with model_type, match_on, ...
fits = load_details(db, PairwiseFit)
mean_sim = fits.groupby("run_id")["mean_sim"].mean().rename("mean_sim")
results = results.join(mean_sim, on="run_id")
results["premium"] = results["value"] / results["mean_sim"]
```
