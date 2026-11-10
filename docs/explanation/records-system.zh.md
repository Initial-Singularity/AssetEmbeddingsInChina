# 记录系统

本文档介绍 AssetEmbeddingsInChina 的实验记录系统（schema v7）：`Record` 类及其定义的表、样本漏斗、逐表视图、`RecordStore` 持久化层、所有脚本共用的 `Run` 对象，以及查询层。

## 目录

- [概述](#概述)
- [Record 类](#record-类)
- [表](#表)
- [视图](#视图)
- [RecordStore 持久化](#recordstore-持久化)
- [数据库文件布局](#数据库文件布局)
- [运行生命周期与状态](#run-lifecycle-and-status)
- [查询层](#查询层)
- [扩展系统](#extending-the-system)

---

## 概述

记录系统位于 `asset_embeddings/records/`。每个训练与评估脚本每次调用写一行 `runs`，并经 `run_id` 关联写入描述该次运行观测到或估计出的内容的各行。

### 设计原则

- **一个事实，一个归处。** 一次运行被要求做什么，存在 `runs.config_content` 里；各行只存观测值或估计值。没有哪一行重复模型类型、特征集或惩罚设置：视图把声明的配置字段放在行旁边展示。
- **只存观测，不存派生。** 比率、占比、跨期均值以及留出切分的汇总都由查询与视图计算。
- **结果行就是论文表格读取的内容。** 在表的自然粒度上给出 `metric` / `value`、实际使用的惩罚、拟合所用样本的大小（`n_scored`）及其特征数（`n_features`）。
- **变长内容用行，不用列。** 样本漏斗、留出切分与主成分都是行。
- **一套命名规则。** 计数一律为 `n_<noun>`；估计量按其本身命名；表按其内容命名。
- **一个关联键加一个自然键。** 每张表经 `run_id` 关联到 `runs`，并声明一个在运行内 `UNIQUE` 的自然键；`run_no` 只是 `runs` 的一列，用作显示序号。
- **按需建表，带约束。** 一个文件只包含实际跑过的任务对应的表；每张表由其类生成，带 `NOT NULL`、`UNIQUE`、`CHECK` 以及（只声明、不强制的）`FOREIGN KEY` 子句，并一同创建其视图。
- **每个脚本一个 `Run` 对象。** 身份、生命周期（`runs` 行，收尾为 `completed` / `error`）以及所有写入都经过 `asset_embeddings.records.Run`；脚本从不直接接触 `RecordStore`。

### 文件结构

```
asset_embeddings/records/
├── __init__.py         # exports all public names
├── base.py             # RecordField, Record, RunRecord, Detail, Result, METRICS, SCHEMA_VERSION, record_classes
├── eval.py             # RVResult, RCResult, MPResult, SampleStep (SAMPLE_STEPS), RidgeSplit
├── grouped.py          # GroupedRVResult / GroupedRCResult, GroupedRVTest / GroupedRCTest
├── probe.py            # ProbeResult, ProbeFold
├── pairwise.py         # PairwiseFit, PairwiseCoef, PairwiseResult
├── train.py            # Epoch, TrainingSummary, W2VTraining, RSFit, RSComponent
├── store.py            # RecordStore, the schema check, the view DDL
├── run.py              # Run (identity + lifecycle + writes), module_name, is_local_main_process
├── provenance.py       # derive_experiment, collect_provenance
└── query.py            # load_runs, load_results, load_summary, load_details, load_sample, load_splits, load_grouped_splits, load_epochs, load_config
```

### 与 CLI 的关系

每个评估与训练脚本都通过 `--result_file`（`-r`）参数接受一个 SQLite 数据库路径。省略时不记录任何内容。设置时，脚本在该文件上打开一个 `Run`，并在运行中边算边写。可选参数 `--experiment`（`-e`）指定写入 `runs.experiment` 的实验族；不给时从配置路径推导（`configs/main/eval/RV/...` 得到 `main`，`configs/<family>/<split>/finetune/...` 得到 `<family>/<split>`）。

```bash
uv run python -m asset_embeddings.scripts.eval.rv -c configs/main/eval/RV/d16/AssetBERT_d16.json -r results/db/main.db -e main
uv run python -m asset_embeddings.scripts.train.bert -c configs/main/finetune/AssetBERT/d16/AssetBERT_d16_2023Q2.json -r results/db/main.db -e main
```

`scripts/run/` 下的 runner 脚本统一设置 `DB` 与 `EXP`，每次调用都传 `-r "$DB" -e "$EXP"`。

---

## Record 类

```
Record                      mechanism: typed RecordFields, the table DDL; a concrete class names its __table__
├── RunRecord               the runs row
└── Detail                  a row owned by one run: run_id, a natural key __key__, optional __config_columns__
    ├── Result              a headline row: metric / value, __summary_keys__
    │   ├── RVResult, RCResult, MPResult, ProbeResult, PairwiseResult
    │   └── GroupedResult → GroupedRVResult, GroupedRCResult
    ├── SampleStep, RidgeSplit
    ├── GroupedTest → GroupedRVTest, GroupedRCTest
    ├── ProbeFold, PairwiseFit, PairwiseCoef
    └── Epoch, TrainingSummary, W2VTraining, RSFit, RSComponent
```

`RecordField` 声明一列：`type_hint`（从注解推断）、`default`、`required`（构造时检查，并生成 `NOT NULL`）、`nullable`（覆盖前者）、`primary_key` 与 `doc`。赋值时做类型检查（`float` 列接受 `int`；`bool` 不算数值）。SQLite 类型遵循 `SQLITE_TYPE_AFFINITY`（`int` 为 `INTEGER`、`float` 为 `REAL`、`str` 为 `TEXT`、`bool` 为以 0 / 1 存储的 `INTEGER`、`bytes` 为 `BLOB`）；`Optional[X]` 取 `X` 的类型。

记录只接受声明过的字段：未知关键字参数抛出 `TypeError`，因此不会出现类未声明的列。构造记录的代码不设置 `run_id`，由 `Run.save` 填入。每个具体类都以其表名注册（`record_classes()`），存储层与查询层就是这样由表找到类的。

一个 `Detail` 类声明：

- `__table__`：它的表。
- `__key__`：`run_id` 之外的键列；DDL 生成 `UNIQUE (run_id, <key>)`，这同时也是表的索引。每次运行一行的类（`TrainingSummary`、`W2VTraining`、`RSFit`）键为空：`UNIQUE (run_id)`。
- `__config_columns__`：列名到 `runs.config_content` 中 JSON 路径的映射（MP 运行的容器配置用嵌套路径，如 `$.model.embedding_model`）。这些名字不得与行的字段重名。

`Result` 增加 `metric`（`METRICS` 的一个键）与 `value`，以及 `__summary_keys__`：汇总视图除期之外用来分组的键列。

`METRICS`，即 `metric` 的取值表：

| metric | `value` 是什么 |
|---|---|
| `r2` | 在该期全部公司上的横截面 ridge R²（RV 为解释性，RC 为预测性） |
| `r2_holdout` | 留出公司上的 Koijen 测试 R²，`1 - Var(resid) / Var(y)`，对各公司切分汇总（`holdout_split_aggregate`；该字段出现之前写入的运行为均值） |
| `mp_score` | 管理型组合分数：测试集上被遮蔽持仓的归一化对数似然 |
| `auc` | 标签探针：交叉验证逻辑回归的测试折 ROC AUC 均值 |
| `fm_coef` | 成对相似度：某项逐期系数的 Fama-MacBeth 均值 |
| `r2_temporal` | 时序 RC：在更早各期拟合的斜率（`slope_window`；该字段出现之前写入的运行只用前一期）不经重估用于本期，所得的 Koijen R² |
| `r2_temporal_aligned` | 同 `r2_temporal`，但本期嵌入先经 Procrustes 对齐到每个更早的期 |

---

## 表

下列每张表的第一列都是 `run_id`（指向 `runs` 的关联）；键列与 `run_id` 一起构成 `UNIQUE`。

### `runs`

每次脚本调用一行，是配置与状态唯一的归处。

| 列 | 类型 | 说明 |
|---|---|---|
| `run_no` | `int` | 文件内的显示序号（SQLite rowid 别名），插入时分配。 |
| `run_id` | `str` | UUID4：其余所有表的关联键，也出现在日志与 `runs_query` 中。 |
| `experiment` | `str` | 实验族，如 `main`、`timesplit_robust`。 |
| `task` | `str` | `RV`、`RC`、`MP`、`GROUPED_RV`、`GROUPED_RC`、`PROBE`、`PAIRWISE`、`pretrain_BERT`、`finetune_BERT`、`pretrain_W2V`、`finetune_W2V`、`fit_RS`。 |
| `script` | `str` | 模块名，如 `asset_embeddings.scripts.eval.rv`。 |
| `config_file` | `str` | 命令行给出的配置路径。 |
| `config_hash` | `str` | 规范化配置 JSON 的 sha1。 |
| `config_content` | `str` | 配置快照；视图从中读取配置列。 |
| `started_at` / `finished_at` | `str` | ISO 8601。 |
| `duration_sec` | `float` | 运行耗时。 |
| `peak_ram_mb` / `peak_vram_mb` | `float` | 进程树常驻内存峰值；CUDA 显存峰值（没有 CUDA 时为 NULL）。 |
| `status` | `str` | `running`、`completed` 或 `error`。 |
| `error_message` | `str` | `status` 为 `error` 时的 traceback。 |
| `git_commit` / `git_dirty` | `str` / `bool` | 运行时的代码检出状态。 |
| `hostname`、`package_version` | `str` | 来源信息。 |
| `schema_version` | `int` | 7（`CHECK`）。 |

### 评估结果

| 表（类） | 键 | `metric`、`value` 之外的列 |
|---|---|---|
| `eval_rv_results` (`RVResult`) | `period` | `ridge_alpha`, `n_scored`, `n_features` |
| `eval_rc_results` (`RCResult`) | `period` | `ridge_alpha`, `n_scored`, `n_features` |
| `eval_grouped_rv_results` / `eval_grouped_rc_results` (`GroupedRVResult` / `GroupedRCResult`) | `period`, `group_label` | `ci_lower`, `ci_upper`, `ridge_alpha`, `n_scored`, `n_features`, `n_group_firms`, `group_value_mean`, `group_value_median`, `group_value_min`, `group_value_max` |
| `eval_mp_results` (`MPResult`) | `period` | `value_std`, `baseline_value`, `baseline_value_std`, `n_features` |
| `eval_probe_results` (`ProbeResult`) | `period` | `n_scored`, `n_positive`, `n_features`, `n_folds_used` |
| `eval_pairwise_results` (`PairwiseResult`) | `term` | `se`, `t`, `p`, `n_periods` |

- `ridge_alpha` 是所用的惩罚；留出评分下为各切分惩罚的中位数，时序 RC 下为所用各斜率惩罚的均值（`slope_window: "previous"` 下即前一期的惩罚）。
- `eval_rc_results` 存放两个任务：`RC`（metric 为 `r2`、`r2_holdout`）与 `RC_temporal`（`r2_temporal`、`r2_temporal_aligned`；第一期之后每期一行，`n_scored` 为被打分那一期的公司数）。向查询层传入任务名时只读该任务的运行。
- `n_scored` 是进入拟合的公司数：漏斗最后一步的行数（分组评估：该单元的行数）。留出评分下它计入该期的全部公司，由各切分把它们分成训练公司与留出公司（`eval_ridge_splits` 中的 `n_train`、`n_test`）。
- `n_features` 计入每一个特征列：嵌入、placebo、公司特征与行业哑变量。
- `group_label` 在分位分组下为 `G1` … `Gn`（`G1` 取值最低），在固定分箱下为分箱标签（如 `1`、`2-3`、`4-10`、`>10`，或配置的 `bin_labels`），在类别分组下为取值本身（整数值的数字不带小数，如 `0`、`1`）；该列不叫 `group`，因为那是 SQL 关键字。`n_group_firms` 是分组文件中落在该组的公司数；`group_value_*` 汇总这些公司的分组变量取值（`coverage_count`），在类别分组下为 NULL。
- 无法打分的探针期（标签只有一类、有标签的公司少于折数、没有可打分的折）没有 `eval_probe_results` 行；其漏斗照常记录。`n_scored` 是有标签的公司数，`n_positive` 是其中标签为 1 的公司数。
- `eval_pairwise_results` 在最后一期之后写入，每项一行（`const`、某个 `match_on` 列，或交互项 `a:b`）；其汇总键为 `term`，因此 `v_eval_pairwise_results_summary` 每次运行每项一行。标准误为 0 或无定义（只有一期）时，`t` 与 `p` 为 NULL。
- `baseline_value` 是加载的模型在 refine 之前的 MP 分数；MP 运行的训练摘要是一条 `train_summary` 行。

各类在视图中的配置列：

| 类 | 配置列 |
|---|---|
| `RVResult` | `model_type`、`features`、`align_chars`、`scoring`、`fixed_alpha`（`$.ridge_alpha`：惩罚经交叉验证选出时为 NULL）、`normalization`、`seed`、`nonpositive_equity` |
| `RCResult` | RV 的各列，去掉 `nonpositive_equity` |
| `GroupedRVResult` / `GroupedRCResult` | RV / RC 的各列，加上 `grouping`、`n_quantiles`、`alpha_scope`、`winsor_pct`、`cv_alpha_min` |
| `MPResult` | `model_type`（`$.model.embedding_model`）、`head`（`$.model.model_type`：`BERT` 或 `SimplifiedMLM`）、`freeze_embedding`、`seed`（`$.train.seed`） |
| `ProbeResult` | `model_type`、`features`、`align_chars`、`normalization`、`seed`、`label_col`、`n_folds` |
| `PairwiseResult`、`PairwiseFit`、`PairwiseCoef` | `model_type`、`label_col`、`match_on`、`interaction`、`seed` |

`features` 与 `match_on` 在 `config_content` 中是 JSON 列表；查询层把它们解码为 Python 列表。

### 样本漏斗（`eval_sample`） { #the-sample-funnel-eval_sample }

`SampleStep`，键为 `period`、`step_no`；列为 `step`、`n_rows`。一期的行被读入或发生变化的每一处写一行，按各步实际运行的顺序编号。文件步给出文件的行数；其余步给出该步之后剩下的行数，因此一次丢弃就是与前一步的差。每次评估都写漏斗，基线也不例外。

| 步 | 写入方 | `n_rows` |
|---|---|---|
| `embedding_file` | 组装器，带 `"embedding"` 特征时 | 嵌入文件的行数 |
| `benchmark_file` | 组装器 | 基准文件的行数 |
| `joined` | 组装器，带 `"embedding"` 特征时 | 嵌入与基准 inner join 之后的行数 |
| `char_file` | 组装器，读取 `char_benchmark_path` 时 | 特征文件的行数 |
| `char_joined` | 组装器，读取 `char_benchmark_path` 时 | 与特征连接之后的行数 |
| `na_dropped` | 组装器 | 丢弃含缺失值的行之后的行数 |
| `nonpositive_equity_dropped` | RV | 剔除非正权益之后的行数（`nonpositive_equity: "keep"` 下不变） |
| `label_file` | 标签探针与成对相似度 | 该期标签文件中有标签的行数 |
| `label_joined` | 标签探针与成对相似度 | 标签 inner join 之后的行数（无标签的公司离开样本） |
| `group_file` | 分组评估 | 该期分组文件的行数 |

取值表是 `SAMPLE_STEPS`，由 `CHECK` 强制；新的步骤名加入 `SAMPLE_STEPS`。

### 留出切分（`eval_ridge_splits`） { #holdout-splits-eval_ridge_splits }

`RidgeSplit`，键为 `period`、`split`；列为 `value`（该切分的 Koijen 测试 R²）、`value_sklearn`（`1 - SS_res / SS_tot`）、`alpha`、`n_train`、`n_test`。留出拟合的每个公司切分一行；横截面拟合不写。一期各切分 `value` 的汇总值（`holdout_split_aggregate`：默认为中位数，该字段出现之前写入的运行为均值）就是结果行的 `value`。

### 分组留出切分（`eval_grouped_ridge_splits`）

`GroupedRidgeSplit`，键为 `period`、`group_label`、`split`；列与 `eval_ridge_splits` 相同。留出打分下每个已打分分组单元的每次公司切分一行，分组 RV 与 RC 共用（由运行的任务区分）；横截面拟合与被跳过的单元不写。一个单元各切分的汇总值就是该单元的 `value`。

### 分组检验（`eval_grouped_rv_tests`、`eval_grouped_rc_tests`）

`GroupedRVTest` / `GroupedRCTest`，键为 `group_a`、`group_b`；列为 `statistic`（两组逐期分数的均值之差的绝对值）、`p_value`、`n_a`、`n_b`。先是相邻分组，分组多于两个时再加首组对末组。

### 探针折（`eval_probe_folds`） { #probe-folds-eval_probe_folds }

`ProbeFold`，键为 `period`、`fold`；列为 `auc`（测试折 ROC AUC）、`n_train`、`n_test`、`n_test_positive`。标签探针每个已打分的折一行；一期各折 `auc` 的均值就是结果行的 `value`。

### 成对拟合与系数（`eval_pairwise_fits`、`eval_pairwise_coefs`） { #pairwise-fits-and-coefficients-eval_pairwise_fits-eval_pairwise_coefs }

`PairwiseFit`，键为 `period`；列为 `r2`（公司对层面回归的 R²）、`n_pairs`、`mean_sim`（各公司对余弦相似度的均值）、`n_firms`。`PairwiseCoef`，键为 `period`、`term`；列为 `coef`。某项 `coef` 跨期的 Fama-MacBeth 汇总就是它的 `eval_pairwise_results` 行；相对平均相似度的溢价，是用该取值除以 `mean_sim` 的均值（见[评估：成对相似度](../how-to/evaluate.md#pairwise-similarity)）。

### 训练

| 表（类） | 键 | 列 |
|---|---|---|
| `run_epochs` (`Epoch`) | `epoch` | `train_loss`、`val_loss`（没有验证集时为 NULL）、`test_loss`（除非选择指标读取测试集，否则为 NULL） |
| `train_summary` (`TrainingSummary`) | 无：每次运行一行 | `epochs_run`, `best_epoch`, `best_loss`, `final_train_loss`, `final_val_loss`, `final_test_loss`, `early_stop_epoch`, `n_parameters`, `tensorboard_dir`, `checkpoint_path`, `embedding_path` |
| `train_w2v_summary` (`W2VTraining`) | 无 | `epochs_run`, `n_vocab`, `n_sentences`, `embedding_path` |
| `train_rs_summary` (`RSFit`) | 无 | `n_investors`, `n_assets`, `embedding_path` |
| `train_rs_components` (`RSComponent`) | `component` | `explained_variance_ratio` |

`train_summary` 对 AssetBERT 的预训练与微调以及两种 MP refine 形状相同：`checkpoint_path` 是最优的 HF 模型（AssetBERT）或最优的 `accelerate` 状态（MP），`embedding_path` 是最优的静态嵌入（仅 AssetBERT）。选择指标本身是配置中的 `train.best_metric`。W2V 与 RS 的行在模型构建时写入，并随训练推进而刷新（W2V：每个 epoch 之后），因此失败的运行保留它已达到的状态。

---

## 视图

每张 `Detail` 表都有一个行视图 `v_<table>`：表中各行，加上来自 `runs` 的 `run_no`、`experiment`、`task`、`config_file`、`git_commit`、`run_status`，以及该类的配置列（用 `json_extract` 从 `runs.config_content` 读取）。列表值的配置字段（`features`）以其 JSON 文本出现，如 `["embedding","chars"]`。

每张结果表还有一个汇总视图 `v_<table>_summary`，每个 `run_id`、汇总键与 `metric` 一行：`mean_value`、`var_value`（跨期的样本方差；只有一期时为 NULL）、`n_periods`，外加 runs 字段与配置列。方差由各值与组均值之差计算，因此不会为负，各期的值彼此接近时也保持精确。并非所有 SQLite 构建都带 `sqrt`，因此视图给出方差；`load_summary` 再加上 `std_value`。视图不存数据：用 `RecordStore` 打开文件时，凡是存储的定义与当前代码不同的视图都会重建；查询 API 从不写文件，读取这类文件时改用按当前定义建立的临时视图，因此文件写入之后新增的配置列读出来为 NULL。

```sql
SELECT model_type, features, mean_value, var_value, n_periods
FROM v_eval_rv_results_summary WHERE run_status = 'completed' AND align_chars = 1;

SELECT group_label, mean_value FROM v_eval_grouped_rv_results_summary WHERE run_id = ?;
```

---

## RecordStore 持久化

`RecordStore(db_path)` 打开一个文件（或 `:memory:`），并创建父目录。当前文件带 `PRAGMA user_version = 7`，在写入第一行 `runs` 时设置；打开一个已有表但 `user_version` 不同的文件会抛出 `RecordsSchemaError`，其信息给出能读取它的代码版本：v6 文件由提交 ff9497ee 读取（表相同，只是 v7 的 `period` 在 v6 中叫 `quarter`），v5 及更早的文件由归档的代码读取。查询层做同样的检查。

| 方法 | 作用 |
|---|---|
| `begin_run(run)` | 把新文件标为 v7，创建 `runs` 及其索引（`experiment, task`；`config_hash`），插入该行，返回 `run_no`。 |
| `finish_run(run_no, status, error_message, resources)` | 设置 `status`、`error_message`、`finished_at`、`duration_sec` 与资源峰值。 |
| `save(record, replace=False)` | 把一条记录插入其类的表；`replace=True` 覆盖自然键相同的行（`INSERT OR REPLACE`）。 |
| `save_many(records, replace=False)` | 在一个事务中插入一个或多个类的记录；失败时什么也不保留。 |

表在首次使用时由其类（`get_table_ddl()`）创建，同时创建行视图，结果类还创建汇总视图。写方法失败时抛出；如何处理失败的写入，策略在 `Run` 里。

---

## 数据库文件布局

```
results/
└── db/
    ├── main.db                          ← shared by train_main.sh + eval_main.sh
    │   ├── runs                         ← one row per script run (config, provenance, status)
    │   ├── run_epochs, train_summary    ← masked-LM training and MP refines
    │   ├── train_w2v_summary, train_rs_summary, train_rs_components
    │   ├── eval_rv_results, eval_rc_results, eval_mp_results
    │   ├── eval_sample                  ← the sample funnel of every evaluated period
    │   ├── eval_ridge_splits            ← holdout splits (only under scoring="holdout")
    │   └── v_<table>, v_<result table>_summary
    ├── pred.db                          ← eval_pred.sh
    ├── coverage.db                      ← eval_coverage.sh (eval_grouped_{rv,rc}_results / _tests, eval_grouped_ridge_splits)
    ├── state_ownership.db               ← eval_state_ownership.sh (probe, grouped and pairwise tables)
    └── ...
```

不同实验流水线使用各自的文件，文件内每次运行又带 `experiment` 标签，因此多个文件可以一起查询。Shell 脚本通过 `DB` 与 `EXP` 两个变量统一管理：

```bash
DB="results/db/main.db"
EXP="main"
cfg="configs/main/eval/RV/d64/AssetBERT_d64.json"   # the last marker: the same string is stored in runs.config_file
job -m bert -m eval -m rv -m "$cfg" "RV eval AssetBERT d64" \
    uv run python -m asset_embeddings.scripts.eval.rv -c "$cfg" -r "$DB" -e "$EXP"
```

---

## 运行生命周期与状态 { #run-lifecycle-and-status }

### `Run` 对象

`asset_embeddings/records/run.py` 是脚本接触 `runs` 表的唯一入口。一个 `Run` 承载该次运行的身份、生命周期以及所有写入：

```python
class Run:
    def __init__(self, store: Optional[RecordStore], *, script: str,
                 config_file: Optional[str] = None, experiment: Optional[str] = None): ...

    @classmethod
    def from_args(cls, args, *, script) -> "Run":
        """--result_file opens the store, --config is config_file, --experiment is the family.
        The store is None without --result_file and on a non-main rank of a distributed launch."""

    run_id: str                    # UUID4, always present, even without a store
    run_no: Optional[int]          # assigned by set_config when a store exists
    task: Optional[str]            # None until set_config
    active: bool                   # store is not None
    error_file: Path               # logs/error/<experiment>/<task or script stem>_<run_id[:8]>_<stamp>.log

    def __enter__(self) -> "Run": ...                # collects the provenance; no database access
    def set_config(self, task, config): ...          # runs row, status 'running'; no-op without a store
    def __exit__(self, exc_type, exc, tb): ...       # completed / error + traceback, then the error file; returns False

    def save(self, record, replace=False): ...
    def save_many(self, records): ...
    def log_epoch(self, epoch, train_loss=None, val_loss=None, test_loss=None): ...
```

`Run` 从入口点起就存在，只知道命令行给它的信息。`from_args` 打开存储；打不开的 `--result_file`（路径问题、其他 schema 版本的文件）在进入 `with` 块之前就失败，因此 `from_args` 自己像对待运行的任何失败一样报告它（一次带 traceback 的日志、一个错误文件）并抛出。`__enter__` 收集来源信息（git commit 与 dirty 标志、主机名、包版本），不碰数据库。`set_config(task, config)` 构造 `RunRecord`（`config_hash`、`config_content`、`started_at`、来源信息；`experiment` 给定或由配置路径推导）并调用 `begin_run`；在此之前的写入被忽略并发一条 WARNING。调用方没有设置 `run_id` 时，`save` 与 `save_many` 会填上，因此代码构造记录时不写它。`log_epoch` 写一行 `run_epochs`。`__exit__` 记一行本次运行的结果与资源峰值，以 `completed`、或以 `error` 加 traceback 调用 `finish_run`，然后写错误文件；它总是返回 `False`，因此块内的异常照常传播。`KeyboardInterrupt` 同样记为 `error`，被中断的运行不会一直停留在 `running`。`module_name(globals())` 在 `python -m` 下也能给出点分模块名。

### 统一的 `main()` 形态

每个配置驱动的脚本（`train.bert` / `train.w2v` / `train.rs`，`eval.mp_bert` / `eval.mp_smlm`，`eval.rv` / `eval.rc`，`eval.rv_grouped` / `eval.rc_grouped`，`eval.label_probe` / `eval.pairwise_similarity`）都在 `get_parser()` 里构建解析器（六个共用参数来自 `utils.add_run_args`），运行 `main(get_parser().parse_args())`，并以同一方式记录：

```python
def main(args):
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        trainer = AssetBERTTrainer()
        trainer.attach(run)
        trainer.load_configs(config_file=args.config)
        if args.override:
            trainer.load_overrides(utils.parse_overrides(args.override))
        run.set_config(trainer.task_name, trainer.config.to_dict())
        trainer.prepare()
        trainer.train()
```

`run.set_config(task, config)` 是插入 `runs` 行的时刻：配置已加载、任务已知（BERT / W2V 的任务取决于配置：pretrain 还是 finetune）。训练器与评估器都有 `task_name` 与 `attach(run)`（设置 `self.run`）；类内只有 `self.run` 这一个记录句柄。评估器的每一期返回其记录（结果、切分行、漏斗步），由 `BaseEvaluator.save` 用一次 `save_many` 写入；掩码语言模型的训练循环调用 `self.run.log_epoch(epoch, train_loss, val_loss=..., test_loss=...)`，训练器在循环结束后写入其 `train_summary` 行。不经 CLI 构造的类持有一个没有存储的占位 `Run`，可以在测试或 notebook 里驱动，所有写入都是空操作。分布式启动时只有本地主进程拿到存储（`is_local_main_process()`），各行只写一次。

### 状态在哪里

`status` 与 `error_message` 只存在于 `runs` 行。结果表与明细表里从不含错误行：失败的那一期没有任何行，失败前完成的各期保留各自的行。

```sql
-- Failed runs of one experiment family, with their traceback
SELECT run_no, run_id, task, config_file, error_message FROM runs
WHERE experiment = 'main' AND status = 'error';

-- Runs still marked running (crashed hard, or in progress)
SELECT run_no, run_id, task, started_at, hostname FROM runs WHERE status = 'running';
```

`scripts/tools/runs_query.py` 在 shell 里回答同样的问题（`status`、`configs --status error`、`missing --root configs/...`），其 `missing` 模式的输出直接喂给 runner 的过滤器，失败或未完成的配置就是这样重跑的。

```
with Run.from_args(...) as run:            ← store opened, provenance collected
    run.set_config(task, config)           ← runs row, status='running'
    period 1 → save_many(results, splits, funnel) ← in DB
    period 2 → save_many(results, splits, funnel) ← in DB
    period 3 → ERROR                      ← __exit__: status='error', traceback; re-raised
    period 4 → never runs
```

### 记录层的失败策略

记录层绝不能让一次运行白算，也绝不能掩盖计算本身的失败。三个阶段，一个地方（`Run`）：

1. **开始：快速失败。** 打开存储时（`Run.from_args`：文件打不开、schema 版本不是 6）或插入行时（`set_config`）的任何失败，在任何计算开始之前就传播出去；runner 记为 FAIL，不消耗 GPU 时间。两者都留下本次运行的错误文件。
2. **运行中：尽力而为，绝不中止。** `save`、`save_many` 与 `log_epoch` 捕获 `sqlite3.Error` 与 `RecordsSchemaError`。第一次失败发一条 WARNING，并向本次运行的错误文件追加一段（运行的头字段、表、记录、traceback），runner 的失败邮件会附上它；后续失败只计数并以 DEBUG 记录。计算继续。
3. **结束：两条规则。** `__exit__` 从不吞掉块内的异常，失败的 `finish_run` 也从不取而代之：任何离开块的异常先被记一次带 traceback 的日志，`finish_run` 的错误（若有）以同样方式记录，随后该异常自己的块收尾错误文件，然后原异常照常传播。成功路径上 `finish_run` 失败同样只记录，脚本正常退出。若有任何写入失败，退出时一条 WARNING 汇总次数与错误文件。这样的运行在数据库里停留为 `running`；`runs_query status` 能看到它，`missing` 会重新选出它的配置，这就是恢复路径。

### `runs` 行存在之前的失败

`load_configs` 与 `load_overrides` 在 `with Run` 块内、但在 `run.set_config(...)` 之前执行，因此配置路径写错或覆盖键打错时**没有 `runs` 行**：一次连配置都没加载的运行没有可记录的事实。`Run.__exit__` 仍然写错误文件，标签用脚本名的 stem 而不是任务（`logs/error/<experiment>/rv_<run_id[:8]>_<stamp>.log`，`run_no : -`），并把 traceback 记进 job 日志。`runs_query missing` 仍会列出该配置，因为它没有 completed 行。配置的跨字段约束属于加载的一部分，因此同样在 `runs` 行存在之前失败：未知键、与 `features` 不一致的 `embedding_path`、长度与 `periods` 不符的路径列表，都在 `load_configs` 内被拒绝。评估过程中抛出的错误（文件缺失、列缺失）发生在 `set_config` 之后，与其他异常一样记为 `error`。

---

## 查询层

`asset_embeddings/records/query.py` 是分析脚本的读取入口。它读取一个或多个文件（`source_db` 列标记来源文件），返回 pandas DataFrame，并且每张表都经其视图读取，因此每行都带 runs 字段，结果行还带配置列；过滤条件可以是其中任意一个，并下推到 SQL。

| 函数 | 返回 |
|---|---|
| `load_runs(dbs, **filters)` | `runs` 行。 |
| `load_results(dbs, task_or_table, **filters)` | 某任务（`RV`、`RC`、`RC_temporal`、`MP`、`GROUPED_RV`、`GROUPED_RC`、`PROBE`、`PAIRWISE`）、某结果表或某结果类的结果行。传入任务名只读该任务的运行；传入表或类则读写入该表的所有任务。 |
| `load_summary(dbs, task_or_table, **filters)` | 加上 `std_value` 的汇总视图。 |
| `load_details(dbs, table, **filters)` | 按名字或类指定的任意记录表，按键排序。 |
| `load_sample`, `load_splits`, `load_grouped_splits`, `load_epochs` | `eval_sample`、`eval_ridge_splits`、`eval_grouped_ridge_splits`、`run_epochs`。 |
| `load_config(db, run_id)` | 某次运行的配置快照，作为 dict。 |

过滤条件：标量是等值比较，`None` 是 `IS NULL`，元组或集合是成员检验（`IN`），布尔值按 0 / 1 比较。对列表值的配置列（`features`），列表按整体比较（`features=["embedding", "chars"]`），列表的元组是成员检验；该列以 Python 列表返回。未知列抛出 `KeyError` 并列出可用列。

```python
from asset_embeddings.records import load_results, load_summary, load_sample, load_splits, load_epochs, load_config

# 1. Every completed RV score of AssetBERT with the characteristics as features, across two files
rv = load_results(["results/db/main.db", "results/db/static.db"], "RV",
                  model_type="AssetBERT", features=["embedding", "chars"], run_status="completed")
rv.groupby(["source_db", "n_features"]).value.mean()

# 2. Mean and standard deviation over the periods, one row per run
summary = load_summary("results/db/main.db", "RV", align_chars=True)

# 3. Where the sample of one run shrank, period by period
funnel = load_sample("results/db/main.db", run_id=rv.iloc[0]["run_id"])

# 4. Per-epoch validation loss of every finetune run, as a curve per run
epochs = load_epochs("results/db/main.db", task="finetune_BERT", run_status="completed")
curves = epochs.pivot(index="epoch", columns="run_id", values="val_loss")

# 5. Holdout splits of one period, and the config of a run
splits = load_splits("results/db/holdout_main.db", period="2024Q1")
cfg = load_config("results/db/main.db", rv.iloc[0]["run_id"])
```

关系型工作（与 `runs` 的连接、配置列、过滤）在 SQL 里逐文件完成，参数化并按视图的列校验；pandas 只拼接各文件的 DataFrame。文件逐个读取而不是 `ATTACH` 到一起，因为 SQLite 默认最多附加十个数据库，而 `results/db` 下的文件更多。

---

## 扩展系统 { #extending-the-system }

### 新增结果表

1. 在 `asset_embeddings/records/eval.py`（或由 `__init__.py` 导入的新模块）中声明该类：

```python
class NewTaskResult(Result):
    """New task score, one row per period."""

    __table__ = "eval_newtask_results"
    __key__ = ("period",)
    __config_columns__ = {"model_type": "$.model_type", "seed": "$.seed"}

    period: str = RecordField(required=True, doc="Period label")
    n_scored: int = RecordField(doc="Rows in the fit")
```

2. 若它是一个新的量，把其指标加入 `METRICS`；导出该类，并在 `RESULT_TASKS`（`records/query.py`）中映射其任务。
3. 在评估一期的地方构造各行，并在 `PeriodOutcome` 中返回；`BaseEvaluator.save` 负责写入。经漏斗记录该期的样本（`pf.sample.add(step, n_rows)`；新的步名加入 `SAMPLE_STEPS`）。

### 新增训练明细

声明一个带 `__table__` 的 `Detail`，每次运行一行时 `__key__` 为空，用 `self.run.save(record, replace=True)` 写入（随训练推进刷新），逐轮损失用 `self.run.log_epoch(...)` 记录。运行状态从不是该行的字段。

### 说明

- **命名**：结果表为 `eval_<task>_results`；其他表按内容命名（`eval_sample`、`eval_ridge_splits`、`train_summary`）。
- **没有自由列**：记录只接受声明过的字段；加一列就加一个 `RecordField`，暴露一个配置字段就加一个配置列。
- **布尔值**以 0 / 1 存储。
