# Evaluators API

The benchmark classes behind the `asset_embeddings.scripts.eval` CLIs. The RV and RC scripts construct one, attach the [`Run`](records.md), load the config and call `prepare()` then `eval()`. The grouped scripts drive a grouped evaluator the same way; it scores one (group, period) cell at a time and records the permutation tests between groups. The label probe and the pairwise-similarity scripts drive `LabelProbeEvaluator` and `PairwiseSimilarityEvaluator` the same way. See [Evaluate](../../how-to/evaluate.md) for the command lines and configs. The managed-portfolio score (`mp_token_scores`, `score_loader`) is documented here; the MP trainers that refine the models and write the MP rows are on the [Trainers](trainers.md) page.

## Ridge benchmarks

::: asset_embeddings.evaluators.base.BaseEvaluator

::: asset_embeddings.evaluators.base.PeriodOutcome

::: asset_embeddings.evaluators.feature.FeatureEvaluator

::: asset_embeddings.evaluators.benchmark.RidgeBenchmarkEvaluator

::: asset_embeddings.evaluators.rv.RVBenchmarkEvaluator

::: asset_embeddings.evaluators.rv.log_equity_target

::: asset_embeddings.evaluators.rc.RCBenchmarkEvaluator

::: asset_embeddings.evaluators.rc_temporal.TemporalRCEvaluator

## Grouped variants

::: asset_embeddings.evaluators.grouped.GroupedRidgeEvaluator

::: asset_embeddings.evaluators.grouped.GroupedRVEvaluator

::: asset_embeddings.evaluators.grouped.GroupedRCEvaluator

## Label probe and pairwise similarity

::: asset_embeddings.evaluators.label_probe.LabelProbeEvaluator

::: asset_embeddings.evaluators.pairwise_similarity.PairwiseSimilarityEvaluator

## Managed portfolios

::: asset_embeddings.evaluators.mp.mp_token_scores

::: asset_embeddings.evaluators.mp.score_loader

## Data layer

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

## Estimation layer

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
