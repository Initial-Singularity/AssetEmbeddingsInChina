# Training

> Part of [AssetEmbeddings](../index.md). See also [Installation](install.md), [Data acquisition](prepare-data.md).

Three model families are supported:

- **Recommender Systems (RS)** — traditional methods like PCA, suitable for low-dimensional representations.
- **Word2Vec (W2V)** — shallow neural network that uses the order of investor holdings (sorted by weight from largest to smallest) to predict "masked" assets.
- **BERT** — Transformer model that generates context-aware embeddings, suitable for modeling ambiguous scenarios like "Is Apple a tech company or a fruit?"

All training entry points share the same five-flag CLI (also in the [CLI reference](../reference/cli.md#train-eval-shared)):

```bash
uv run python -m asset_embeddings.scripts.train.<model> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | Path to the JSON config file (examples below). **Required.** |
| `--log`, `-l` | `str` (0+) | Log-file path(s). Optional. |
| `--result_file`, `-r` | `str` | SQLite path for structured results. Optional. |
| `--override`, `-o` | `str` (1+) | Override config fields, `key=value` with dot notation (e.g. `train.max_epoches=10 optimizer.learning_rate=0.001`). |
| `--verbose`, `-v` | flag | DEBUG console output (default `INFO`). |

With `--result_file`, each run also writes a per-run training summary (`train_{bert,w2v,rs}_recordings`) via the [records system](../explanation/records-system.md).

## AssetRS

`asset_embeddings.scripts.train.rs` trains recommender-system-type embedding
models, corresponding to the first method in *Asset Embeddings*. It learns
low-dimensional asset embeddings from an investor-asset matrix using PCA.

```bash
uv run python -m asset_embeddings.scripts.train.rs -c config.json
```

### Config example

```json
{
    "model_type": "RS_Binary",
    "n_components": 4,
    "whiten": false,
    "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
    "data_format": "csv",
    "id_key": "InvestorID",
    "portfolio_key": "Portfolio",
    "proportion1_key": "Proportion1",
    "proportion2_key": "Proportion2",
    "save_folder": "checkpoints/finetune/AssetRS/RS_Binary/d4/2023Q2",
    "save_name": "AssetRS_Binary_d4_2023Q2",
    "save_format": ".pkl",
    "seed": 42
}
```

### Config fields

| Field             | Type   | Description                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `model_type`      | `str`  | RS embedding model type. Supported: `RS_Binary`, `RS_Ranks`, `RS_Level0`, `RS_LevelMin`. |
| `n_components`    | `int`  | Embedding dimension (target reduced dimension). |
| `whiten`          | `bool` | Whether to whiten the dimensionality-reduction result. |
| `data_path`       | `str` or `list[str]` | Single file, directory, or list of file paths. |
| `data_format`     | `str`  | Input data format: `csv`, `json`, `binary`. |
| `id_key`          | `str`  | Investor ID key. |
| `portfolio_key`   | `str`  | Portfolio key. |
| `proportion1_key` | `str`  | Proportion (relative to total shares) key. |
| `proportion2_key` | `str`  | Proportion (relative to tradable A-shares) key. |
| `save_folder`     | `str`  | Save directory. |
| `save_name`       | `str`  | Model save filename (no extension). |
| `save_format`     | `str`  | Recommended `.pkl`. |
| `seed`            | `int`  | Random seed. |

> Note: RS uses proportion data when constructing the investor-asset matrix,
> so the source data must have been produced with the `--include_proportion True`
> flag in `asset_embeddings.scripts.data.sharehold`.

## AssetW2V

`asset_embeddings.scripts.train.w2v` trains asset-level Word2Vec models.
Holding sequences are treated as sentences and stock codes as words; low-dimensional
representations are learned through contextual co-occurrence. Implemented via
Gensim with Skip-Gram / CBOW, negative sampling, and high-frequency downsampling.

This corresponds to *Asset Word2Vec* in the paper, suitable for modeling asset
co-occurrence structure and initializing more complex models (e.g. AssetBERT).

```bash
uv run python -m asset_embeddings.scripts.train.w2v -c config.json
```

### Config example

```json
{
    "pretrained_model": "checkpoints/pretrained/AssetW2V/d4/AssetW2V_d4_pretrained.model",
    "embedding_dim": 4,
    "epochs": 10,
    "window": 5,
    "min_count": 1,
    "sg": 1,
    "sample": 0.001,
    "negative_sample": 10,
    "seed": 42,
    "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
    "data_format": "csv",
    "id_key": "InvestorID",
    "portfolio_key": "Portfolio",
    "workers": 1,
    "save_folder": "checkpoints/finetune/AssetW2V/d4",
    "save_name": "AssetW2V_d4_2023Q2",
    "save_format": ".model"
}
```

### Config fields

| Field              | Type    | Description                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `pretrained_model` | `str`   | Optional pretrained model path (incremental training); `null` for from-scratch. |
| `embedding_dim`    | `int`   | Word embedding dimension. |
| `epochs`           | `int`   | Number of training epochs. |
| `window`           | `int`   | Context window size. |
| `min_count`        | `int`   | Ignore stocks appearing fewer times than this (recommended: 1). |
| `sg`               | `int`   | Skip-Gram (1) or CBOW (0). |
| `sample`           | `float` | High-frequency downsampling threshold. |
| `negative_sample`  | `int`   | Number of negatives per positive sample. |
| `seed`             | `int`   | Random seed. |
| `data_path`        | `str` or `list[str]` | Input data path(s). |
| `data_format`      | `str`   | `csv`, `json`, `binary`. |
| `id_key`           | `str`   | Investor ID key. |
| `portfolio_key`    | `str`   | Portfolio key. |
| `workers`          | `int`   | Number of parallel threads. |
| `save_folder`      | `str`   | Save directory. |
| `save_name`        | `str`   | Save filename (no extension). |
| `save_format`      | `str`   | `.model` or `.txt`. |

> W2V does not consume proportion data, so the `proportion*_key` fields are
> not present.

## AssetBERT

AssetBERT is a Transformer-based embedding model that learns context-aware
representations (contextualized asset embeddings) of assets from holding
sequences. It is the AssetBERT model of the paper (shortened to BERT in
figures and tables).

We use AssetW2V embeddings to initialize BERT's embedding layer; the
tokenizer vocabulary (excluding special tokens) is shared with AssetW2V.

```bash
uv run python -m asset_embeddings.scripts.train.bert -c config.json
```

### Config example

```json
{
    "model": {
        "model_type": "BERT",
        "model_checkpoint": null,
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
        "embedding_file": null,
        "vocab_size": 5236,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 64,
        "type_vocab_size": 1,
        "freeze_embedding": false,
        "freeze_encoder": false
    },
    "tokenizer": {
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
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
        "max_length": 64,
        "mask_prob": 0.15,
        "mask_indices": null,
        "num_repeats": 1,
        "cache_size": 400
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
        "persistent_workers": true,
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
        "max_epoches": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "check_per_step": 200,
        "report_per_epoch": 1,
        "calculate_contextualized_embeddings": true,
        "save_per_epoch": 1,
        "save_folder": "checkpoints/finetune/AssetBERT/d4/2023Q2",
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

### Config fields

The config is in JSON format and groups fields into modules:

- `model`: model structure and initialization.
- `tokenizer`: asset sequence encoding.
- `dataset`: dataset definition and preprocessing.
- `dataloader`: PyTorch DataLoader behavior.
- `optimizer`: optimizer and LR scheduler.
- `accelerator`: HuggingFace Accelerator, mixed precision, `torch.compile`.
- `train`: training control, validation/test split, early stopping.

| Module         | Field                         | Type        | Description                                                                 |
| -------------- | ----------------------------- | ----------- | --------------------------------------------------------------------------- |
| `model`        | `model_type`                  | `str`       | Fixed as `"BERT"`. |
|                | `model_checkpoint`            | `str`       | Pretrained path; `null` from scratch. |
|                | `w2v_model`                   | `str`       | Word2Vec model for vocab + embedding init; `null` for random. |
|                | `embedding_file`              | `str`       | Embedding init CSV; `null` if not used. |
|                | `vocab_size`                  | `int`       | Asset vocabulary size. |
|                | `hidden_size`                 | `int`       | BERT embedding dim. |
|                | `num_hidden_layers`           | `int`       | Number of Transformer encoder layers. |
|                | `num_attention_heads`         | `int`       | Number of attention heads; must divide `hidden_size`. |
|                | `intermediate_size`           | `int`       | FFN hidden size, typically `hidden_size * 4`. |
|                | `max_position_embeddings`     | `int`       | Max sequence length (max holdings per investor). |
|                | `type_vocab_size`             | `int`       | Type embedding dim. Set to 1. |
|                | `freeze_embedding`            | `bool`      | Freeze embedding-layer weights. |
|                | `freeze_encoder`              | `bool`      | Freeze encoder; fine-tune embeddings + LM head only. |
| `tokenizer`    | `w2v_model`                   | `str`       | Word2Vec path for vocab/index mapping. |
|                | `vocab_file`                  | `str`       | HuggingFace-format vocab file. |
|                | `embedding_file`              | `str`       | Embedding-init CSV consistent with `model.embedding_file`. |
|                | `alias_file`                  | `str`       | Optional `{ticker: name}` alias mapping. |
| `dataset`      | `data_path`                   | `str` or `list[str]` | Input path(s). |
|                | `data_format`                 | `str`       | `csv` / `json` / `binary`. |
|                | `id_key`                      | `str`       | Investor ID key. |
|                | `portfolio_key`               | `str`       | Portfolio key. |
|                | `proportion1_key`             | `str`       | Proportion (total shares) key. |
|                | `proportion2_key`             | `str`       | Proportion (tradable A-shares) key. |
|                | `include_proportion`          | `bool`      | Emit `(id, proportion)` tuples per item. Requires `proportion*_key`. |
|                | `max_length`                  | `int`       | Max holdings per investor; should match `model.max_position_embeddings`. |
|                | `mask_prob`                   | `float`     | MLM mask ratio. Mutually exclusive with `mask_indices`. |
|                | `mask_indices`                | `list[int]` | Fixed mask positions. Mutually exclusive with `mask_prob`. |
|                | `num_repeats`                 | `int`       | Times to repeat dataset per epoch. |
|                | `cache_size`                  | `int`       | Streaming-loader cache size. |
| `dataloader`   | `batch_size`                  | `int`       | Batch size. |
|                | `shuffle`                     | `bool`      | Whether to shuffle. |
|                | `num_workers`                 | `int`       | Loader processes. |
|                | `persistent_workers`          | `bool`      | Reduce per-epoch worker restart. Default: `true`. |
|                | `pin_memory`                  | `bool`      | Pin to RAM for GPU transfer. Default: `true`. |
|                | `drop_last`                   | `bool`      | Drop last incomplete batch. Default: `false`. |
| `optimizer`    | `optimizer_type`              | `str`       | `Adam`, `AdamW`, `Adam8bit`, `PagedAdam8bit`, `AdamW8bit`, `PagedAdamW8bit`, `Lion`, `Lion8bit`, `PagedLion8bit`, `DAdaptation`, `SGDNesterov`, `SGD8bit`, `Adafactor`. |
|                | `optimizer_kwargs`            | `dict`      | Optional extras. |
|                | `learning_rate`               | `float`     | Initial LR. |
|                | `lr_scheduler_type`           | `str`       | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `adafactor`. |
|                | `lr_scheduler_warmup_steps`   | `int`       | Warmup steps. |
|                | `lr_scheduler_train_steps`    | `int`       | Total steps; `null` auto-computes. |
|                | `lr_scheduler_num_cycles`     | `float`     | For cosine schedulers. |
|                | `lr_scheduler_power`          | `float`     | For polynomial scheduler. |
| `accelerator`  | `mixed_precision`             | `str`       | `'no'`, `'fp16'`, `'bf16'`, `'auto'`. |
|                | `gradient_accumulation_steps` | `int`       | Grad accumulation. |
|                | `accelerator_checkpoint`      | `str`       | Accelerator checkpoint path. |
|                | `use_compile`                 | `bool`      | Enable `torch.compile`. |
|                | `compile_backend`             | `str`       | `"inductor"`, `"eager"`, `"aot_eager"`. |
|                | `compile_mode`                | `str`       | `"default"`, `"reduce-overhead"`, `"max-autotune"`. |
|                | `compile_fullgraph`           | `bool`      | Require full-graph compile. |
|                | `compile_dynamic`             | `bool`      | Enable dynamic shapes. |
|                | `compile_cache_dir`           | `str`       | Persistent compile cache. |
|                | `allow_tf32`                  | `bool`      | TF32 on Ampere/Ada. |
| `train`        | `max_epoches`                 | `int`       | Max training epochs. |
|                | `validation_split`            | `float`     | Validation split (`null` / 0 disables). |
|                | `test_split`                  | `float`     | Test split (`null` / 0 disables). `validation + test < 1.0`. |
|                | `split_method`                | `str`       | `"random"` or `"sequential"`. |
|                | `best_metric`                 | `str`       | `"train_loss"` or `"val_loss"`. |
|                | `clip_grad_norm`              | `float`     | L2 grad-clip. |
|                | `clip_grad_value`             | `float`     | Abs grad-clip. |
|                | `detect_anomaly`              | `bool`      | Anomaly detection (debugging). |
|                | `check_per_step`              | `int`       | Log frequency (steps). |
|                | `report_per_epoch`            | `int`       | Intermediate report frequency. |
|                | `calculate_contextualized_embeddings` | `bool` | Extract size-weighted contextualized embeddings post-training. Needs `include_proportion=true`. |
|                | `save_per_epoch`              | `int`       | Save frequency. |
|                | `keep_best_only`              | `bool`      | Drop non-best checkpoints/embeddings at end. |
|                | `save_folder`                 | `str`       | Save directory. |
|                | `save_name`                   | `str`       | Filename (no extension). |
|                | `save_format`                 | `str`       | Recommended `.safetensors`. |
|                | `seed`                        | `int`       | Random seed. |
|                | `early_stop_enabled`          | `bool`      | Enable early stopping. |
|                | `early_stop_patience`         | `int`       | Patience epochs. |
|                | `early_stop_min_delta`        | `float`     | Improvement threshold (rel if `early_stop_relative=true`). |
|                | `early_stop_mode`             | `str`       | `"min"` / `"max"`. |
|                | `early_stop_relative`         | `bool`      | Relative vs absolute improvement. |
|                | `early_stop_metric`           | `str`       | `"train_loss"` / `"val_loss"`. |

### Static vs. contextualized embeddings

AssetBERT provides two complementary asset representations.

**Static embeddings** are read directly from the trained embedding layer.
Each token maps to a fixed vector, analogous to standard BERT word embeddings,
summarizing asset-level characteristics across all training data. Vocabulary
matches AssetW2V (modulo BERT special tokens).

**Contextualized embeddings** are derived from BERT hidden states processing
real investor holding sequences. Following Gabaix et al. (2024), we compute the
size-weighted average

$$\sum_i S_{i,a} \cdot x_{i,a}$$

over investors $i$ for each asset $a$, where:

- $x_{i,a}$ is the BERT hidden state for asset $a$ in investor $i$'s sequence,
- $S_{i,a}$ is the investor's shareholding proportion relative to tradable
  A-shares (the `proportion2_key` field).

To enable contextualized extraction:

1. Set `dataset.include_proportion=true`.
2. Set `train.calculate_contextualized_embeddings=true`.
3. Configure both `proportion1_key` and `proportion2_key`.

Both kinds of embeddings are saved as CSV with `Token` plus `Embed_1...Embed_N`
columns. Naming distinguishes them:

- Static: `{save_name}_best_embedding.csv`
- Contextualized: `{save_name}_best_contextual_embedding.csv`
