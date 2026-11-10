# 评估器 API

`asset_embeddings.scripts.eval` 各 CLI 背后的基准测试类。RV 与 RC 脚本构造一个，挂上 [`Run`](records.md)，加载配置，再调用 `prepare()` 与 `eval()`。分组脚本以同样的方式驱动一个分组评估器；它逐个（分组, 期）单元打分，并记录各分组之间的置换检验。标签探针与成对相似度脚本以同样的方式驱动 `LabelProbeEvaluator` 与 `PairwiseSimilarityEvaluator`。命令行与配置见[评估](../../how-to/evaluate.md)。managed-portfolio 分数（`mp_token_scores`、`score_loader`）在本页；负责 refine 模型并写入 MP 行的 MP 训练器见[训练器](trainers.md)页。

## Ridge 基准

::: asset_embeddings.evaluators.base.BaseEvaluator

::: asset_embeddings.evaluators.base.PeriodOutcome

::: asset_embeddings.evaluators.feature.FeatureEvaluator

::: asset_embeddings.evaluators.benchmark.RidgeBenchmarkEvaluator

::: asset_embeddings.evaluators.rv.RVBenchmarkEvaluator

::: asset_embeddings.evaluators.rv.log_equity_target

::: asset_embeddings.evaluators.rc.RCBenchmarkEvaluator

::: asset_embeddings.evaluators.rc_temporal.TemporalRCEvaluator

## 分组变体

::: asset_embeddings.evaluators.grouped.GroupedRidgeEvaluator

::: asset_embeddings.evaluators.grouped.GroupedRVEvaluator

::: asset_embeddings.evaluators.grouped.GroupedRCEvaluator

## 标签探针与成对相似度

::: asset_embeddings.evaluators.label_probe.LabelProbeEvaluator

::: asset_embeddings.evaluators.pairwise_similarity.PairwiseSimilarityEvaluator

## Managed portfolios

::: asset_embeddings.evaluators.mp.mp_token_scores

::: asset_embeddings.evaluators.mp.score_loader

## 数据层

::: asset_embeddings.evaluators.data.files.resolve_period_files

::: asset_embeddings.evaluators.data.frame.PeriodFrame

::: asset_embeddings.evaluators.data.frame.SampleFunnel

::: asset_embeddings.evaluators.data.assembly.PeriodAssembler

::: asset_embeddings.evaluators.data.assembly.FeatureAssembler

::: asset_embeddings.evaluators.data.sources.read_firm_file

::: asset_embeddings.evaluators.data.sources.LabelFileSource

::: asset_embeddings.evaluators.data.sources.GroupFileSource

::: asset_embeddings.evaluators.data.grouping.assign_groups

::: asset_embeddings.evaluators.data.features.FeatureSpec

## 估计层

::: asset_embeddings.evaluators.estimation.ridge.CrossSectionRidge

::: asset_embeddings.evaluators.estimation.ridge.HoldoutRidge

::: asset_embeddings.evaluators.estimation.ridge.PenaltySelector

::: asset_embeddings.evaluators.estimation.ridge.RidgeFit

::: asset_embeddings.evaluators.estimation.ridge.SplitScore

::: asset_embeddings.evaluators.estimation.ridge.UnscorableSplit

::: asset_embeddings.evaluators.estimation.ridge.make_estimator

::: asset_embeddings.evaluators.estimation.ridge.holdout_fold

::: asset_embeddings.evaluators.estimation.ridge.holdout_test_mask

::: asset_embeddings.evaluators.estimation.alignment.procrustes_rotation

::: asset_embeddings.evaluators.estimation.resampling.bootstrap_interval

::: asset_embeddings.evaluators.estimation.resampling.permutation_test

::: asset_embeddings.evaluators.estimation.classification.cross_validated_auc

::: asset_embeddings.evaluators.estimation.classification.ProbeFit

::: asset_embeddings.evaluators.estimation.classification.UnscorableProbe

::: asset_embeddings.evaluators.estimation.pairwise.cell_statistics

::: asset_embeddings.evaluators.estimation.pairwise.ols_from_cells

::: asset_embeddings.evaluators.estimation.pairwise.term_names

::: asset_embeddings.evaluators.estimation.panel.fama_macbeth
