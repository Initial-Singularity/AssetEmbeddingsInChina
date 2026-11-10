# 评估器与训练器

本页介绍评估与训练代码的构造：一次评估由哪几层组成，基准要填入哪些钩子，一期如何计算，以及 AssetBERT 与 managed-portfolio（MP）refine 共用的掩码语言模型训练循环。配置字段见[评估](../how-to/evaluate.md)与[训练](../how-to/train.md)；运行写入的内容见[记录系统](records-system.md)。

## 目录

- [概述](#overview)
- [评估生命周期](#the-evaluation-lifecycle)
- [特征层](#the-feature-layer)
- [数据层](#the-data-layer)
- [估计层](#the-estimation-layer)
- [Ridge 基准](#ridge-benchmarks)
- [分组评估](#grouped-evaluation)
- [标签探针](#label-probe)
- [成对相似度](#pairwise-similarity)
- [掩码语言模型训练循环](#the-masked-lm-training-loop)
- [AssetBERT 与 MP refine](#assetbert-and-the-mp-refines)
- [新增基准](#adding-a-benchmark)

---

## 概述 { #overview }

```
asset_embeddings/
├── evaluators/
│   ├── base.py         # BaseEvaluator: config, run, period loop, saving, summary, the firm universe
│   ├── feature.py      # FeatureEvaluator: characteristics, feature assembly and selection, industry drift check
│   ├── benchmark.py    # RidgeBenchmarkEvaluator: target, design, ridge fit, result rows
│   ├── rv.py / rc.py   # RVBenchmarkEvaluator, RCBenchmarkEvaluator: the targets
│   ├── grouped.py      # GroupedRidgeEvaluator and GroupedRV / GroupedRC: cells, bootstrap, permutation tests
│   ├── label_probe.py  # LabelProbeEvaluator: cross-validated logit AUC of a firm label
│   ├── pairwise_similarity.py  # PairwiseSimilarityEvaluator: pair similarity on same-value indicators
│   ├── mp.py           # the MP score of a masked-LM
│   ├── data/           # files, assembly, sources, features, grouping: one period's frame
│   └── estimation/     # ridge, classification, pairwise cells, Fama-MacBeth, bootstrap and permutation statistics
└── trainers/
    ├── base.py         # BaseTrainer: config, run, artifacts folder
    ├── mlm.py          # BaseMLMTrainer: split, loaders, model, epoch loop, best epoch, early stopping
    ├── bert.py         # AssetBERTTrainer: pretraining and finetuning
    ├── mp.py           # MPTrainer, MPBertTrainer, MPSimplifiedMLMTrainer: the MP refines
    ├── w2v.py / rs.py  # Word2Vec and the recommender-system PCA fits
```

`asset_embeddings/scripts/` 下的入口保持精简：它们解析六个共用参数，打开一个 [`Run`](records-system.md#run-lifecycle-and-status)，再依次调用 `attach(run)`、`load_configs`、`load_overrides`、`prepare` 与 `eval`（或 `train`）驱动一个评估器或训练器。基准计算的一切都在其类中，它写入的每一行都经过该运行。

一次评估由以下几层构成：

| 层 | 包 | 内容 |
|---|---|---|
| 生命周期 | `evaluators.base` | 逐期循环、每期完成即保存其记录、汇总、每期的公司全集 |
| 特征 | `evaluators.feature` | 公司特征、每期的特征列及其选择、行业哑变量漂移检查 |
| 数据 | `evaluators.data` | 读入并连接一期的文件得到数据框，以及它的样本漏斗 |
| 估计 | `evaluators.estimation` | ridge 惩罚与打分，横截面或留出；探针的分类器；成对单元与 Fama-MacBeth 汇总；bootstrap 与置换检验统计量 |

评估器类与配置的字段组一一对应。`BaseEvaluator` 只需要数据字段（`EvalDataConfig`：路径、期、列名、种子）。`FeatureEvaluator` 再加上特征字段（`FeatureConfig`：`features`、`align_chars`、`char_benchmark_path`、`normalization`）：

```
BaseEvaluator
├── FeatureEvaluator
│   ├── RidgeBenchmarkEvaluator      RV, RC; GroupedRV, GroupedRC through GroupedRidgeEvaluator
│   └── LabelProbeEvaluator
└── PairwiseSimilarityEvaluator
```

成对回归在原始嵌入上进行，因此没有特征字段，直接位于生命周期层之上。一个基准类把各层组合起来：它声明自己的配置类、任务名与列，并用其估计器从组装好的数据框计算一期。

## 评估生命周期 { #the-evaluation-lifecycle }

[`BaseEvaluator`][asset_embeddings.evaluators.base.BaseEvaluator] 对 ridge 与特征一无所知。`prepare` 构建期组装器（`make_assembler`），由它解析配置中的逐期文件（`periods` 的每个条目对应一个文件，按列表顺序；文件从不按名称与期匹配），并向子类索取其额外数据源。随后 `eval` 运行循环：

1. `iter_periods` 组装每期的主数据框，并按顺序把额外数据源连接上去。
2. `evaluate_period`（抽象方法）计算该期并返回一个 `PeriodOutcome`：该期的取值及其记录。
3. `save` 在一个事务中把这些记录写入该运行，因此一期一完成，它的行就已存在。
4. 最后一期之后，`summarize` 返回一个最终结果（默认为均值；分组评估在此记录其置换检验，成对相似度在此记录其 Fama-MacBeth 行），以同样方式保存。
5. `after_periods` 运行跨期检查（默认没有；特征层在此检查行业哑变量）。

子类填入的钩子：

| 钩子 | 作用 |
|---|---|
| `config_class` | 评估器加载的配置类 |
| `task_name` | `runs.task` 标签（`RV`、`RC`、`GROUPED_RV`、`GROUPED_RC`、`PROBE`、`PAIRWISE`） |
| `benchmark_columns` | 目标变量在股票代码之外需要的基准列 |
| `make_assembler` | 每期主数据框的组装器（默认为 `PeriodAssembler`） |
| `sources` | 连接到主数据框上的逐期文件（标签文件、分组文件） |
| `evaluate_period` | 一期的取值与记录 |
| `summarize` | 最后一期之后写入的记录 |
| `after_periods` | 汇总保存之后的跨期检查 |

不设置任何全局随机状态：一次评估的每次随机抽取（placebo 列、交叉验证的折、bootstrap 与置换）都从 `config.seed` 取得自己的生成器。

## 特征层 { #the-feature-layer }

[`FeatureEvaluator`][asset_embeddings.evaluators.feature.FeatureEvaluator] 承载基于特征列的评估所共有的部分。子类声明其公司特征（`char_cols`，抽象），供 `"chars"` 特征与 `align_chars` 使用。该层构建 `FeatureAssembler`（通过 `make_assembler`），在 `prepare` 中一次性构建特征选择 `FeatureSpec`，存为 `self.features`；在 `after_periods` 中，带 `"industry"` 特征时，若各期的行业哑变量列不同则给出警告（`check_industry_drift`）：一期的 R² 基本不受影响，但系数的跨期面板会混入不同的列集合。

## 数据层 { #the-data-layer }

`evaluators.data` 把一期的文件变成一个 `PeriodFrame`：期标签、数据框（一个 `Token` 列、目标列与特征列）、它的 `SampleFunnel`，以及分组评估的分组归属。

- `PeriodAssembler` 依据配置的数据字段构建每期的公司全集。它读入基准文件的目标列；该期有嵌入文件时，按股票代码与之做 inner join；没有时，基准文件中的公司就是全集。随后丢弃任何含缺失值的行。子类通过三个扩展点加入列，按以下顺序调用：`extra_benchmark_columns`（在目标列之外从基准文件读入的列）、`join_extra`（在嵌入之后连接的其他文件）与 `transform`（在丢弃之前加入的列）。
- `FeatureAssembler` 用 `config.features` 的特征列填充这些扩展点：公司特征（带 `"chars"` 或 `align_chars` 时）与 `IndustryCode`（带 `"industry"` 时）取自基准文件，设置了 `char_benchmark_path` 时则取自该文件；然后是 placebo 列与行业哑变量，它逐期保留哑变量列以供漂移检查。读入的每一列都参与丢弃，`align_chars` 正是借此把公司限制为具备全部特征的公司。对特征配置而言，嵌入文件恰在 `"embedding"` 是特征时存在，因此基线在基准文件的公司上运行。
- placebo 列 `Placebo_1..4` 在 [-1, 1] 上均匀分布，每期从 `np.random.default_rng([seed, crc32(period)])` 抽取，抽取发生在丢弃之前的连接后数据框上，按文件顺序每家公司一行：一期的抽取结果无论单独评估还是在其他期之后评估都相同。
- `PeriodSource` 的子类把更多逐期文件连接到完成的数据框上，通过 `read_firm_file` 读入（股票代码作为六位 `Token`，同一股票列出两次则拒绝）。`LabelFileSource` 以 inner join 连接公司标签，因此无标签的公司离开样本（漏斗步骤 `label_file` 与 `label_joined`）。`GroupFileSource` 在分组文件自身的全集上把公司分组（取值的分位数或固定分箱，或在类别分组下每个不同取值一组），并加入一个 `group` 列，保留数据框的每一行。
- `FeatureSpec` 按 `config.features` 的顺序选出特征列，并逐期缩放（`standard`、`minmax` 或 `none`）。
- 每次读入、连接与丢弃都向该期的 `SampleFunnel` 追加一步，记录为 `eval_sample` 行（见[样本漏斗](records-system.md#the-sample-funnel-eval_sample)）。

## 估计层 { #the-estimation-layer }

`evaluators.estimation.ridge` 包含两个接口相同的估计器，由 `config.scoring` 选择：

| 估计器 | `scoring` | 拟合于 | 打分于 | 取值 |
|---|---|---|---|---|
| `CrossSectionRidge` | `cross_section` | 该期的全部公司 | 同一批公司 | R²（`r2`） |
| `HoldoutRidge` | `holdout` | 每次切分的训练公司 | 留出公司 | 各次切分 Koijen 测试 R² 的中位数（或均值，`holdout_split_aggregate`）（`r2_holdout`） |

每个估计器把 `select_penalty`（固定的 `ridge_alpha`，或在固定网格上做 10 折交叉验证；留出评分下每次切分一个惩罚，在该切分的训练公司上选择）与 `score`（在给定惩罚下拟合并打分）分开；`fit` 两者都做。这种分离让调用方只选一次惩罚，就能对许多重抽样打分。留出切分按股票代码与切分种子的哈希把公司分到留出折，因此同一家公司在每个模型、每个全集中都被留出，bootstrap 重复抽到的同一公司的各个副本也同在一折。无法拟合或打分的切分（训练公司少于 10、没有留出公司，或留出公司上的目标为常数）抛出 `UnscorableSplit`。

拟合结果以一个 `RidgeFit` 值返回：指标、取值、惩罚（留出评分下为各切分的中位数）、公司数以及逐切分分数，后者成为 `eval_ridge_splits` 行。

留出评分每期要做数百次小规模 ridge 拟合，因此不逐次调用 sklearn。`cv_penalty` 按 10 折 `KFold` 下 `RidgeCV` 的方式为每次切分选惩罚：每折的中心化 Gram 矩阵只构造一次，网格上的每个惩罚都由它求解；`ridge_predict` 按 `Ridge` 的 Cholesky 求解器拟合该切分。两者都与 sklearn 逐位一致（在 lock 文件中的 sklearn 与 scipy 版本下），求解被拒时交给 sklearn。每家公司在全部切分上的哈希折每个估计器只算一次，惩罚搜索、打分与每个 bootstrap 重抽样都复用它。

`evaluators.estimation.resampling` 包含百分位 bootstrap 区间（取值为 None 的重抽样被剔除）与两组均值的双侧置换检验，各自使用带种子的独立生成器。

`evaluators.estimation.classification` 包含探针的 `cross_validated_auc`：分层 k 折逻辑回归，以测试折 ROC AUC 的均值打分，一期无法打分时抛出 `UnscorableProbe`。`evaluators.estimation.pairwise` 由逐模式的单元统计量求解成对 OLS（`cell_statistics`、`ols_from_cells`、`term_names`），`evaluators.estimation.panel` 包含 `fama_macbeth`：逐期系数的时间序列均值，及其标准误、t 统计量与 p 值。

## Ridge 基准 { #ridge-benchmarks }

[`RidgeBenchmarkEvaluator`][asset_embeddings.evaluators.benchmark.RidgeBenchmarkEvaluator] 为每期只做一次 ridge 拟合的基准实现 `evaluate_period`：

1. `target(frame)`（抽象方法）返回构建设计矩阵所用的数据框、目标值与诊断信息。丢弃行的目标变量返回缩减后的数据框，因此被丢弃的行不参与特征缩放；它追加的步骤只进入漏斗一次。
2. `design(frame)` 构建特征矩阵：选中的列，经过缩放。
3. 估计器在设计矩阵上拟合目标变量；目标变量在各公司间为常数时，按计算结果写入，并在日志中给出警告（sklearn 将其打分为 1.0）。
4. 该期的记录包括结果行、每次留出切分一条切分行，以及漏斗。

两个基准只在目标变量与列上不同：

| 基准 | 目标变量 | 公司特征 |
|---|---|---|
| RV（`RVBenchmarkEvaluator`） | 对数账面权益对对数市场权益做横截面 OLS 的残差；在 `nonpositive_equity: "drop"` 下，账面或市场权益非正的公司先离开该期 | beta、资产增长、盈利能力、股利对资产比 |
| RC（`RCBenchmarkEvaluator`） | 下一季度收益 `qret_next` | beta、规模、账面市值比、资产增长、盈利能力、动量 |

## 分组评估 { #grouped-evaluation }

[`GroupedRidgeEvaluator`][asset_embeddings.evaluators.grouped.GroupedRidgeEvaluator] 是一个协作式子类：`GroupedRVEvaluator(GroupedRidgeEvaluator, RVBenchmarkEvaluator)` 与 `GroupedRCEvaluator(GroupedRidgeEvaluator, RCBenchmarkEvaluator)` 复用父类的目标变量与设计矩阵，替换逐期计算与汇总。每期：

1. 分组文件的数据源已把文件中的每家公司分到一个分组：取值的分位数或固定分箱，或在 `grouping: "categorical"` 下每个不同取值一组（§7.1 的国有产权分组）。文件中少于 10 家公司、或连接后少于 10 行的分组被跳过并给出警告。
2. 惩罚在整期上选一次（`alpha_scope: "cross_section"`），或在每个（分组, 期）单元上各选一次（`"cell"`），所用网格以 `cv_alpha_min` 为下限。
3. 单元打分时，目标变量在单元内做缩尾（`winsor_pct`）；bootstrap 在选定惩罚下对单元公司的 `n_bootstrap` 个重抽样重新打分，得到百分位区间。
4. 最后一期之后，`summarize` 用置换检验比较各分组的各期得分（相邻分组，然后在多于两组时首组对末组），并记录每个检验。在类别分组下，分组是被打过分的取值，按取值排序。

在留出评分下运行同一段代码，只是换成 `HoldoutRidge`：单元的取值是各切分留出 R² 的汇总值，每次切分记入 `eval_grouped_ridge_splits`，区间是对单元公司做 bootstrap 得到的该汇总值的区间；切分无法拟合的单元被跳过，无法打分的重抽样不计入区间，并在日志中计数。

## 标签探针 { #label-probe }

[`LabelProbeEvaluator`][asset_embeddings.evaluators.label_probe.LabelProbeEvaluator] 是一个特征评估，使用 RV 的公司特征，并以标签文件为数据源。每期它检查标签是否为二值，在整个横截面上缩放选中的特征，并用 `cross_validated_auc` 打分。该期的记录包括结果行、每个已打分折一行以及漏斗；无法打分的期只写入其漏斗，并给出警告。汇总记录已打分各期的平均 AUC 到日志，不写入任何行。配置字段与记录见[评估：标签探针](../how-to/evaluate.md#label-probe)。

## 成对相似度 { #pairwise-similarity }

[`PairwiseSimilarityEvaluator`][asset_embeddings.evaluators.pairwise_similarity.PairwiseSimilarityEvaluator] 位于生命周期层：它的数据框是 `PeriodAssembler` 给出的公司全集（原始嵌入连接上基准文件的 `match_on` 列，即它的 `benchmark_columns`）再加上标签。每期它通过单元统计量把每对公司的余弦相似度回归到同值指示变量上，并记录拟合、每项一条系数行以及漏斗；`summarize` 用 `fama_macbeth` 把每项的系数汇总为结果行。配置字段与记录见[评估：成对相似度](../how-to/evaluate.md#pairwise-similarity)。

## 掩码语言模型训练循环 { #the-masked-lm-training-loop }

[`BaseMLMTrainer`][asset_embeddings.trainers.mlm.BaseMLMTrainer] 包含 AssetBERT 预训练、微调与 MP refine 共用的部分。它的配置是一个容器，含 `model`、`tokenizer`、`dataset`、`dataloader`、`accelerator`、`optimizer` 与 `train` 块；`train` 块是循环配置（`AssetBERTTrainConfig` 或 `MPTrainConfig`），它声明留出切分。

`prepare` 为运行设定种子，构建 tokenizer 与数据集，并用 `split_dataset` 切分：按配置比例得到训练、验证与测试部分，由训练种子随机抽取，或取为连续切片。每个部分各配一个 `MaskingCollator`（`split_masking`）：验证与测试 loader 用以训练种子与 `i` 为种子的生成器遮蔽数据集索引 `i` 处的样本，因此每次评估留出损失都看到相同的遮蔽输入，各 epoch 可以在同等条件下比较；训练 loader 则每个 epoch 重新抽取遮蔽（给定 `mask_indices` 时，每个部分都遮蔽这些位置）。`dataset.num_repeats` 只重放训练部分。随后它构建 loader、模型与优化器，并用 `accelerate` 包装。

`fit` 运行各个 epoch。每个 epoch 依次是：一次训练遍历；在选择所需的每个留出切分上做一次打分遍历（有验证切分时打分验证切分；`best_metric` 或已启用的 `early_stop_metric` 为 `test_loss` 时打分测试切分）；一次报告；一条 `run_epochs` 行；每 `save_every_epochs` 个 epoch 一次定期保存；选择损失达到新的最小值时调用 `on_best`；以及早停检查。循环结束后，`on_fit_end` 接收循环的 `FitSummary`。指标所读切分没有任何行时，在第一个 epoch 之前就被拒绝。

| 钩子 | 作用 |
|---|---|
| `model_config_class`、`make_model_preparer` | 模型及其构建方式 |
| `train_config_class` | 循环配置，并随之决定可用的切分 |
| `extra_configs` | 更多配置块（MP 的 `run` 块及其期） |
| `on_periodic_save`、`on_best`、`on_fit_end` | 在定期 epoch、新的最优 epoch 以及循环结束后保存什么 |

## AssetBERT 与 MP refine { #assetbert-and-the-mp-refines }

`AssetBERTTrainer`（`train.bert`）在定期 epoch 保存 `accelerate` 状态与静态嵌入，在每个新的最优 epoch 保存模型及其静态嵌入；循环结束后写入 `train_summary` 行，执行 `keep_best_only`，并可选地计算上下文嵌入。

`MPTrainer`（`eval.mp_bert`、`eval.mp_smlm`）是一次 refine 加一次打分。它先在测试切分上为加载的模型打分（基线），运行循环，每个新的最优 epoch 的 `accelerate` 状态覆盖保存在前一个之上，然后重新加载最优状态，再次在测试切分上打分。分数来自 `evaluators.mp`，是遮蔽 token 上 `1 + log p / log V` 的均值（随机猜测为 0，完全确定为 1）。`MPBertTrainer` refine 一个 AssetBERT 检查点，`MPSimplifiedMLMTrainer` 在固定嵌入上拟合无注意力的 SimplifiedMLM 头；两者都写入 `eval_mp_results` 行与 `train_summary` 行。

Word2Vec 与推荐系统训练器没有这种 epoch 循环，直接派生自 `BaseTrainer`。

## 新增基准 { #adding-a-benchmark }

新的 ridge 基准继承 `RidgeBenchmarkEvaluator`，并提供 `config_class`、`task_name`、`benchmark_columns`、`char_cols`、`result_class` 与 `target`；`asset_embeddings.records` 中新增的结果类为它提供表与视图（见[扩展系统](records-system.md#extending-the-system)），`asset_embeddings/scripts/eval/` 下一个精简脚本驱动它。需要另一个逐期文件的基准通过 `sources` 加入一个 `PeriodSource`。不是 ridge 的基准按其配置选择所在的层：基于特征列的（如标签探针）继承 `FeatureEvaluator`，没有特征字段的（如成对相似度）继承 `BaseEvaluator`；两者都自行实现 `evaluate_period`，并复用数据层。
