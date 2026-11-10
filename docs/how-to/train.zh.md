# 训练

> 隶属于 [AssetEmbeddingsInChina](../index.md)。另见 [安装](install.md)、[评估](evaluate.md)。

支持三个模型族：

- **推荐系统（Recommender Systems, RS）**——PCA 等传统方法，适合低维表示。
- **Word2Vec（W2V）**——浅层神经网络，把每个组合（持仓按权重从大到小排序）当作一个句子，在上下文窗口内预测与某资产同时被持有的资产。
- **BERT**——Transformer 模型，生成上下文感知的嵌入，适合刻画“Apple 是科技公司还是水果？”这类含歧义的情形。

所有训练入口共享同一套六参数 CLI（亦见 [CLI 参考](../reference/cli.md#train-eval-shared)）：

```bash
uv run python -m asset_embeddings.scripts.train.<model> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--experiment EXPERIMENT] [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置文件路径（示例见下）。**必填。** |
| `--log`, `-l` | `str`（0+） | 日志文件路径。可选。 |
| `--result_file`, `-r` | `str` | 存放结构化结果的 SQLite 路径。可选。 |
| `--experiment`, `-e` | `str` | 存入 `runs` 表的实验族（默认由配置路径推导，如 `configs/main/...` 得到 `main`）。runner 会传 `-e "$EXP"`。可选。 |
| `--override`, `-o` | `str`（1+） | 覆盖配置字段，`key=value` 且支持点号嵌套（如 `train.max_epochs=10 optimizer.learning_rate=0.001`）。 |
| `--verbose`, `-v` | flag | DEBUG 级控制台输出（默认 `INFO`）。 |

**一次运行写入什么。** 每次运行都会把生效的配置写成 `save_folder` 下的 `config.json`，与下文各模型列出的模型文件放在一起；BERT 运行还会把 TensorBoard 标量写到 `results/tensorboard/` 下。指定 `--result_file` 时，每次运行插入一条 `runs` 行（配置只存一次、脚本/git commit/主机名等来源信息、状态、耗时、`peak_ram_mb` / `peak_vram_mb`），以及一条逐运行的训练行：BERT 为 `train_summary`（已跑 epoch 数、最优 epoch 及其损失、末轮损失、最优检查点与嵌入路径），其余为 `train_w2v_summary` 或 `train_rs_summary`（另加每个成分一条 `train_rs_components` 行）；BERT 训练还会每个 epoch 写一条 `run_epochs` 行（`train_loss`、`val_loss`、`test_loss`）。检查点与嵌入 CSV 本身写到 `save_folder` 下，按惯例为 `checkpoints/...`。表 schema 与读取接口（`load_runs`、`load_epochs`、`load_details`）见 [记录系统](../explanation/records-system.md)。

## AssetRS

`asset_embeddings.scripts.train.rs` 训练推荐系统类的嵌入模型，对应 *Asset Embeddings* 中的第一种方法。它用 PCA 从投资者-资产矩阵中学习低维资产嵌入。

```bash
uv run python -m asset_embeddings.scripts.train.rs -c config.json
```

### 配置示例

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

### 配置字段

| Field             | Type   | 说明                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `model_type`      | `str`  | RS 嵌入模型类型。支持：`RS_Binary`、`RS_Ranks`、`RS_Level0`、`RS_LevelMin`。 |
| `n_components`    | `int`  | 嵌入维度（目标降维维数）。 |
| `whiten`          | `bool` | 是否对降维结果做白化。 |
| `data_path`       | `str`  | 单个文件或目录（不接受列表）。 |
| `data_format`     | `str`  | `csv`，唯一可接受的取值。 |
| `id_key`          | `str`  | 投资者 ID 键。 |
| `portfolio_key`   | `str`  | 组合键。 |
| `proportion1_key` | `str`  | 比例（相对总股本）键。 |
| `proportion2_key` | `str`  | 比例（相对流通 A 股）键。 |
| `save_folder`     | `str`  | 保存目录。 |
| `save_name`       | `str`  | 文件名前缀：运行写出 `<save_name>_model.pkl`（拟合好的 PCA）与 `<save_name>_embedding.csv`。 |
| `save_format`     | `str`  | 必须为 `.pkl`（唯一允许的取值）。 |
| `seed`            | `int`  | 随机种子。 |

> 注：RS 在构造投资者-资产矩阵时会用到比例数据，因此源组合 CSV 必须带有比例列。

## AssetW2V

`asset_embeddings.scripts.train.w2v` 训练资产级 Word2Vec 模型。持仓序列被当作句子、股票代码当作词，通过上下文共现学习低维表示。基于 Gensim 实现，支持 Skip-Gram / CBOW、负采样与高频下采样。

它对应论文中的 *Asset Word2Vec*，适合刻画资产共现结构、并为更复杂的模型（如 AssetBERT）做初始化。

```bash
uv run python -m asset_embeddings.scripts.train.w2v -c config.json
```

### 配置示例

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
    "save_folder": "checkpoints/finetune/AssetW2V/d4/2023Q2",
    "save_name": "AssetW2V_d4_2023Q2"
}
```

### 配置字段

| Field              | Type    | 说明                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `pretrained_model` | `str`   | 可选，之前某次 W2V 运行保存的 `.model` 文件（增量训练）；`null` 表示从零训练。 |
| `embedding_dim`    | `int`   | 词嵌入维度。 |
| `epochs`           | `int`   | 训练轮数。 |
| `window`           | `int`   | 上下文窗口大小。 |
| `min_count`        | `int`   | 忽略出现次数少于此值的股票（推荐：1）。 |
| `sg`               | `int`   | Skip-Gram（1）或 CBOW（0）。 |
| `sample`           | `float` | 高频下采样阈值。 |
| `negative_sample`  | `int`   | 每个正样本对应的负样本数。 |
| `seed`             | `int`   | 随机种子。 |
| `data_path`        | `str` 或 `list[str]` | 输入数据路径。 |
| `data_format`      | `str`   | `csv`，唯一可接受的取值。 |
| `id_key`           | `str`   | 投资者 ID 键。 |
| `portfolio_key`    | `str`   | 组合键。 |
| `workers`          | `int`   | 并行线程数。 |
| `save_folder`      | `str`   | 保存目录。 |
| `save_name`        | `str`   | 文件名前缀：模型写为 `<save_name>.model`，嵌入写为 `<save_name>_embedding.csv`。 |

> W2V 不消费比例数据，因此不存在 `proportion*_key` 字段。

## AssetBERT

AssetBERT 是基于 Transformer 的嵌入模型，从持仓序列中学习资产的上下文感知表示（contextualized asset embeddings）。即论文中的 AssetBERT（图表中简写为 BERT）。

我们用 AssetW2V 嵌入初始化 BERT 的嵌入层；分词器词表（不含特殊词元）与 AssetW2V 共享。

```bash
uv run python -m asset_embeddings.scripts.train.bert -c config.json
```

### 配置示例

下例是 `templates/train_main.json` 中 d4、2023Q2 的微调阶段：它从预训练 BERT 检查点和该季度的 W2V 模型出发，并提取上下文嵌入。

```json
{
    "model": {
        "model_type": "BERT",
        "model_checkpoint": "checkpoints/pretrained/AssetBERT/d4/AssetBERT_d4_pretrained_best/model.safetensors",
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
        "embedding_file": null,
        "vocab_size": 5236,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 256,
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
        "include_proportion": true,
        "max_length": 256,
        "num_repeats": 1,
        "mask_prob": 0.15,
        "mask_indices": null
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
        "use_compile": true,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_fullgraph": false,
        "compile_dynamic": false,
        "compile_cache_dir": null,
        "allow_tf32": true
    },
    "train": {
        "max_epochs": 30,
        "validation_split": null,
        "split_method": "random",
        "best_metric": "train_loss",
        "clip_grad_norm": 1.0,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "calculate_contextualized_embeddings": true,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/finetune/AssetBERT/d4/2023Q2",
        "save_name": "AssetBERT_d4_2023Q2",
        "save_format": ".safetensors",
        "seed": 42,
        "keep_best_only": true
    }
}
```

### 配置字段

配置为 JSON 格式，按模块分组字段：

- `model`：模型结构与初始化。
- `tokenizer`：资产序列编码。
- `dataset`：数据集定义与预处理。
- `dataloader`：PyTorch DataLoader 行为。
- `optimizer`：优化器与学习率调度。
- `accelerator`：HuggingFace Accelerator、混合精度、`torch.compile`。
- `train`：训练控制、验证/测试切分、早停。

| 模块         | Field                         | Type        | 说明                                                                 |
| -------------- | ----------------------------- | ----------- | --------------------------------------------------------------------------- |
| `model`        | `model_type`                  | `str`       | 固定为 `"BERT"`。 |
|                | `model_checkpoint`            | `str`       | 预训练路径；`null` 表示从零训练。 |
|                | `w2v_model`                   | `str`       | 用于词表 + 嵌入初始化的 Word2Vec 模型；`null` 表示随机初始化。 |
|                | `embedding_file`              | `str`       | 嵌入初始化 CSV；不用则为 `null`。 |
|                | `vocab_size`                  | `int`       | 资产词表大小。 |
|                | `hidden_size`                 | `int`       | BERT 嵌入维度。 |
|                | `num_hidden_layers`           | `int`       | Transformer 编码器层数。 |
|                | `num_attention_heads`         | `int`       | 注意力头数；必须整除 `hidden_size`。 |
|                | `intermediate_size`           | `int`       | FFN 隐层大小；本项目在所有嵌入维度下均取 512。 |
|                | `max_position_embeddings`     | `int`       | 最大序列长度，含 `[CLS]` 与 `[SEP]`。 |
|                | `type_vocab_size`             | `int`       | token 类型（segment）id 的个数。设为 1。 |
|                | `freeze_embedding`            | `bool`      | 冻结嵌入层权重。 |
|                | `freeze_encoder`              | `bool`      | 冻结编码器；仅微调嵌入层 + LM 头。 |
| `tokenizer`    | `w2v_model`                   | `str`       | 用于词表/索引映射的 Word2Vec 路径。 |
|                | `vocab_file`                  | `str`       | HuggingFace 格式的词表文件。 |
|                | `pretrained_tokenizer_file`   | `str`       | 已保存的分词器目录，用 `from_pretrained` 加载。最先检查，其后依次为 `vocab_file`、`w2v_model`、`embedding_file`。 |
|                | `embedding_file`              | `str`       | 与 `model.embedding_file` 一致的嵌入初始化 CSV。 |
|                | `alias_file`                  | `str`       | 保留字段；没有代码读取它，因此不起作用。 |
| `dataset`      | `data_path`                   | `str` 或 `list[str]` | 组合 CSV 文件或目录，或由 `PortfolioCorpus.save` 写出的语料目录（以内存映射方式打开）。 |
|                | `data_format`                 | `str`       | `csv`，唯一可接受的取值。 |
|                | `id_key`                      | `str`       | 投资者 ID 键。 |
|                | `portfolio_key`               | `str`       | 组合键。 |
|                | `proportion1_key`             | `str`       | 比例（总股本）键。 |
|                | `proportion2_key`             | `str`       | 比例（流通 A 股）键。 |
|                | `include_proportion`          | `bool`      | 用 `proportion*_key` 列填充每个样本的 `proportions1` / `proportions2` 张量（为 false 时全为零）。 |
|                | `max_length`                  | `int`       | 序列长度，含 `[CLS]` 与 `[SEP]`，因此最多容纳 `max_length - 2` 个持仓，更长的组合会被截断（截掉的是最小的持仓，因为组合按持股比例排序；截断会去掉 `[SEP]`）。数据集会记录被截断的组合数，以及有多少持仓的代码不在 tokenizer 词表中（它们变成 `[UNK]`）。应与 `model.max_position_embeddings` 一致。 |
|                | `mask_prob`                   | `float`     | MLM 遮蔽比例。与 `mask_indices` 互斥。训练样本每个 epoch 重新抽取遮蔽；验证与测试样本的遮蔽由 `train.seed` 与样本序号决定的生成器抽取，因此每次评估留出损失时看到的遮蔽输入相同，各 epoch 之间可以直接比较。 |
|                | `mask_indices`                | `list[int]` | 固定遮蔽位置。与 `mask_prob` 互斥。 |
|                | `num_repeats`                 | `int`       | 每个 epoch 经由采样器重放训练切分的次数；验证与测试切分不重复。默认：`1`。 |
| `dataloader`   | `batch_size`                  | `int`       | 批大小。 |
|                | `shuffle`                     | `bool`      | 是否打乱。 |
|                | `num_workers`                 | `int`       | 加载进程数。 |
|                | `persistent_workers`          | `bool`      | 减少每个 epoch 的 worker 重启。默认：`true`。 |
|                | `pin_memory`                  | `bool`      | 锁页内存以加速 GPU 传输。默认：`true`。 |
|                | `drop_last`                   | `bool`      | 丢弃最后一个不完整批。默认：`false`。 |
| `optimizer`    | `optimizer_type`              | `str`       | `Adam`、`AdamW`、`Adam8bit`、`PagedAdam8bit`、`AdamW8bit`、`PagedAdamW8bit`、`Lion`、`Lion8bit`、`PagedLion8bit`、`DAdaptation`、`SGDNesterov`、`SGD8bit`、`Adafactor`。 |
|                | `optimizer_kwargs`            | `dict`      | 可选附加参数。 |
|                | `learning_rate`               | `float`     | 初始学习率。 |
|                | `lr_scheduler_type`           | `str`       | `constant`、`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`adafactor`。 |
|                | `lr_scheduler_warmup_steps`   | `int`       | 预热步数。 |
|                | `lr_scheduler_train_steps`    | `int`       | 总步数；`null` 自动计算。 |
|                | `lr_scheduler_num_cycles`     | `float`     | 用于 cosine 类调度器。 |
|                | `lr_scheduler_power`          | `float`     | 用于 polynomial 调度器。 |
| `accelerator`  | `mixed_precision`             | `str`       | `'no'`、`'fp16'`、`'bf16'`、`'fp8'`、`'auto'`（`'auto'` 交由 accelerate 的默认设置决定）。 |
|                | `gradient_accumulation_steps` | `int`       | 梯度累积。 |
|                | `accelerator_checkpoint`      | `str`       | Accelerator 检查点路径。 |
|                | `use_compile`                 | `bool`      | 启用 `torch.compile`。 |
|                | `compile_backend`             | `str`       | `"inductor"`、`"eager"`、`"aot_eager"`。 |
|                | `compile_mode`                | `str`       | `"default"`、`"reduce-overhead"`、`"max-autotune"`。 |
|                | `compile_fullgraph`           | `bool`      | 要求全图编译。 |
|                | `compile_dynamic`             | `bool`      | 启用动态形状。 |
|                | `compile_cache_dir`           | `str`       | 持久化编译缓存。 |
|                | `allow_tf32`                  | `bool`      | 在 Ampere/Ada 上启用 TF32。 |
| `train`        | `max_epochs`                  | `int`       | 最大训练轮数（至少为 1）。 |
|                | `validation_split`            | `float`     | 验证集比例，如 `0.1`；`null` 或 `0` 表示不切分验证集。AssetBERT 的 `train` 块（`AssetBERTTrainConfig`，预训练与微调相同）没有测试集切分；MP 的 `train` 块（`MPTrainConfig`）加上 `test_split`。 |
|                | `split_method`                | `str`       | `"random"` 或 `"sequential"`。 |
|                | `best_metric`                 | `str`       | `"train_loss"` 或 `"val_loss"`；`"val_loss"` 需要 `validation_split > 0`，在加载配置时检查。 |
|                | `clip_grad_norm`              | `float`     | L2 梯度裁剪。 |
|                | `clip_grad_value`             | `float`     | 绝对值梯度裁剪。 |
|                | `detect_anomaly`              | `bool`      | 异常检测（调试用）。 |
|                | `log_every_steps`             | `int`       | 每 n 步记录一次训练损失。 |
|                | `report_every_epochs`         | `int`       | 每 n 个 epoch 记录各 epoch 损失并写入 TensorBoard。 |
|                | `calculate_contextualized_embeddings` | `bool` | 训练后提取按规模加权的上下文嵌入。数据须包含两列持股比例；提取过程会自行开启比例读取。 |
|                | `save_every_epochs`           | `int`       | 每 n 个 epoch 保存一次检查点（及嵌入）。 |
|                | `keep_best_only`              | `bool`      | 结束时丢弃非最优的检查点/嵌入。 |
|                | `save_folder`                 | `str`       | 保存目录。 |
|                | `save_name`                   | `str`       | 文件名（无扩展名）。 |
|                | `save_format`                 | `str`       | `.safetensors` 或 `.pt`，决定逐 epoch 的 accelerate 状态格式。最优模型总是保存为 `model.safetensors`，因此 `.pt` 与 `calculate_contextualized_embeddings` 不兼容（提取时会去找 `model.pt`）。 |
|                | `seed`                        | `int`       | 随机种子。 |
|                | `early_stop_enabled`          | `bool`      | 启用早停。 |
|                | `early_stop_patience`         | `int`       | 容忍轮数。 |
|                | `early_stop_min_delta`        | `float`     | 改善阈值（若 `early_stop_relative=true` 则为相对值）。 |
|                | `early_stop_mode`             | `str`       | `"min"` / `"max"`。 |
|                | `early_stop_relative`         | `bool`      | 相对改善还是绝对改善。 |
|                | `early_stop_metric`           | `str`       | `"train_loss"` 或 `"val_loss"`，切分规则与 `best_metric` 相同。 |

### 静态嵌入 vs. 上下文嵌入

AssetBERT 提供两种互补的资产表示。

**静态嵌入（static embeddings）**直接从训练好的嵌入层读取。每个词元映射到一个固定向量，类比于标准 BERT 的词嵌入，概括了跨全部训练数据的资产级特征。词表与 AssetW2V 一致（除 BERT 特殊词元外）。

**上下文嵌入（contextualized embeddings）**由 BERT 处理真实投资者持仓序列时的隐状态导出。沿用 Gabaix et al. (2024)，我们对每个资产 $a$、在投资者 $i$ 上计算按规模加权的均值

$$\frac{\sum_i S_{i,a} \cdot x_{i,a}}{\sum_i S_{i,a}}$$

其中：

- $x_{i,a}$ 是资产 $a$ 在投资者 $i$ 序列中的 BERT 隐状态，
- $S_{i,a}$ 是该投资者相对流通 A 股的持股比例（即 `proportion2_key` 字段）。

没有出现在任何组合中的资产保留其静态嵌入。

要启用上下文嵌入提取：

1. 设 `train.calculate_contextualized_embeddings=true`。
2. 确保数据包含 `dataset.proportion1_key` 与 `dataset.proportion2_key` 指定的两列持股比例。无论 `dataset.include_proportion` 取何值，提取过程都会自行读取这两列。
3. `train.save_format` 保持为 `.safetensors`。

两种嵌入都保存为带 `Token` 加 `Embed_1...Embed_N` 列的 CSV。命名上加以区分：

- 静态：`{save_name}_best_embedding.csv`
- 上下文：`{save_name}_best_contextual_embedding.csv`

### 输出文件

所有文件都写在 `train.save_folder` 下：

- `{save_name}_best/`：最优 epoch 的 HuggingFace 模型（`config.json`、`model.safetensors`）；微调运行以 `{save_name}_best/model.safetensors` 作为 `model.model_checkpoint` 加载。
- `{save_name}_best_embedding.csv`，以及启用时的 `{save_name}_best_contextual_embedding.csv`。
- 每个保存的 epoch 的 `{save_name}_accelerator_epoch{n}/`（accelerate 状态）与 `{save_name}_epoch{n}_embedding.csv`；`keep_best_only` 会在结束时删除它们。
- `config.json`：生效的配置。
