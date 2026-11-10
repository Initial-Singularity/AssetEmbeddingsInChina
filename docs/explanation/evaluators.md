# Evaluators and trainers

This page describes how the evaluation and training code is built: the layers of an evaluation, the hooks a benchmark fills in, how one period is computed, and the masked-LM training loop that AssetBERT and the managed-portfolio (MP) refines share. For the config fields see [Evaluate](../how-to/evaluate.md) and [Train](../how-to/train.md); for what the runs write see [Records system](records-system.md).

## Contents

- [Overview](#overview)
- [The evaluation lifecycle](#the-evaluation-lifecycle)
- [The feature layer](#the-feature-layer)
- [The data layer](#the-data-layer)
- [The estimation layer](#the-estimation-layer)
- [Ridge benchmarks](#ridge-benchmarks)
- [Grouped evaluation](#grouped-evaluation)
- [Label probe](#label-probe)
- [Pairwise similarity](#pairwise-similarity)
- [The masked-LM training loop](#the-masked-lm-training-loop)
- [AssetBERT and the MP refines](#assetbert-and-the-mp-refines)
- [Adding a benchmark](#adding-a-benchmark)

---

## Overview

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

The entry points under `asset_embeddings/scripts/` stay thin: they parse the six shared flags, open one [`Run`](records-system.md#run-lifecycle-and-status), and drive an evaluator or trainer through `attach(run)`, `load_configs`, `load_overrides`, `prepare` and `eval` (or `train`). Everything a benchmark computes lives in its class, and every row it writes goes through the run.

An evaluation is built from these layers:

| Layer | Package | Holds |
|---|---|---|
| Lifecycle | `evaluators.base` | the period loop, saving each period's records as it completes, the summary, each period's firm universe |
| Features | `evaluators.feature` | the characteristics, the feature columns of each period and their selection, the industry drift check |
| Data | `evaluators.data` | reading and joining one period's files into a frame, and its sample funnel |
| Estimation | `evaluators.estimation` | the ridge penalty and score, cross section or holdout; the probe's classifier; the pairwise cells and the Fama-MacBeth aggregate; bootstrap and permutation statistics |

The evaluator classes follow the config's field groups. `BaseEvaluator` needs only the data fields (`EvalDataConfig`: paths, periods, column names, seed). `FeatureEvaluator` adds the feature fields (`FeatureConfig`: `features`, `align_chars`, `char_benchmark_path`, `normalization`):

```
BaseEvaluator
├── FeatureEvaluator
│   ├── RidgeBenchmarkEvaluator      RV, RC; GroupedRV, GroupedRC through GroupedRidgeEvaluator
│   └── LabelProbeEvaluator
└── PairwiseSimilarityEvaluator
```

The pairwise regression runs on the raw embedding, so it has no feature fields and sits on the lifecycle layer directly. A benchmark class combines the layers: it declares its config class, task name and columns, and computes a period from the assembled frame with its estimator.

## The evaluation lifecycle

[`BaseEvaluator`][asset_embeddings.evaluators.base.BaseEvaluator] knows nothing about ridges or features. `prepare` builds the period assembler (`make_assembler`), which resolves the per-period files of the config (one file per entry of `periods`, in list order; files are never matched to periods by name), and asks the subclass for its extra sources. `eval` then runs the loop:

1. `iter_periods` assembles each period's primary frame and joins the extra sources onto it, in order.
2. `evaluate_period` (abstract) computes the period and returns a `PeriodOutcome`: the period's value and its records.
3. `save` writes the records to the run in one transaction, so a period's rows exist as soon as it completes.
4. After the last period, `summarize` returns a final outcome (the mean value by default; the grouped evaluation records its permutation tests here, the pairwise similarity its Fama-MacBeth rows), which is saved the same way.
5. `after_periods` runs the checks across periods (none by default; the feature layer checks the industry dummies).

The hooks a subclass fills in:

| Hook | Purpose |
|---|---|
| `config_class` | the config class the evaluator loads |
| `task_name` | the `runs.task` label (`RV`, `RC`, `GROUPED_RV`, `GROUPED_RC`, `PROBE`, `PAIRWISE`) |
| `benchmark_columns` | the benchmark columns the target needs beyond the stock id |
| `make_assembler` | the assembler of each period's primary frame (`PeriodAssembler` by default) |
| `sources` | per-period files joined onto the primary frame (the label file, the group file) |
| `evaluate_period` | one period's value and records |
| `summarize` | the records written after the last period |
| `after_periods` | checks across the periods, after the summary is saved |

No global random state is seeded: every random draw of an evaluation (the placebo columns, the cross-validation folds, the bootstrap and the permutations) takes its own generator from `config.seed`.

## The feature layer

[`FeatureEvaluator`][asset_embeddings.evaluators.feature.FeatureEvaluator] holds what the evaluations over feature columns share. A subclass declares its firm characteristics (`char_cols`, abstract), used by the `"chars"` feature and by `align_chars`. The layer builds the `FeatureAssembler` (through `make_assembler`), builds the feature selection `FeatureSpec` once in `prepare` as `self.features`, and in `after_periods`, with the `"industry"` feature, warns when the industry dummy columns differ between periods (`check_industry_drift`): a period's R² is essentially unaffected, but a cross-period panel of the coefficients would mix different column sets.

## The data layer

`evaluators.data` turns a period's files into a `PeriodFrame`: the period label, the frame (a `Token` column, the target columns and the feature columns), its `SampleFunnel` and, for the grouped evaluation, the group assignment.

- `PeriodAssembler` builds each period's firm universe from the data fields of the config. It reads the benchmark's target columns and, when the period has an embedding file, inner-joins it on the stock id; without one the benchmark's firms are the universe. It then drops every row with a missing value. A subclass adds columns through three extension points, called in this order: `extra_benchmark_columns` (read from the benchmark beside the targets), `join_extra` (further files joined after the embedding) and `transform` (columns added before the drop).
- `FeatureAssembler` fills them with the feature columns of `config.features`: the characteristics (with `"chars"` or `align_chars`) and `IndustryCode` (with `"industry"`) from the benchmark file, or from `char_benchmark_path` when that is set; then the placebo columns and the industry dummies, whose columns it keeps per period for the drift check. Every column read enters the drop, which is how `align_chars` restricts the firms to those with every characteristic. For a feature config the embedding file is there exactly when `"embedding"` is a feature, so the baselines run on the benchmark's firms.
- The placebo columns `Placebo_1..4` are uniform on [-1, 1], drawn per period from `np.random.default_rng([seed, crc32(period)])` on the joined frame before the drop, one row per firm in file order: a period's draw is the same whether it is evaluated alone or after other periods.
- `PeriodSource` subclasses join further per-period files onto the finished frame, read through `read_firm_file` (the stock id as a six-digit `Token`, a stock listed twice refused). `LabelFileSource` inner-joins a firm label, so unlabeled firms leave (funnel steps `label_file` and `label_joined`). `GroupFileSource` assigns firms to groups on the group file's own universe (quantiles or fixed bins of the value, or one group per distinct value under the categorical grouping) and adds a `group` column, keeping every row of the frame.
- `FeatureSpec` selects the feature columns in the order of `config.features` and scales them per period (`standard`, `minmax` or `none`).
- Every read, join and drop appends a step to the period's `SampleFunnel`, recorded as the `eval_sample` rows (see [the sample funnel](records-system.md#the-sample-funnel-eval_sample)).

## The estimation layer

`evaluators.estimation.ridge` holds two estimators with one interface, chosen by `config.scoring`:

| Estimator | `scoring` | Fits on | Scores on | Value |
|---|---|---|---|---|
| `CrossSectionRidge` | `cross_section` | every firm of the period | the same firms | the R² (`r2`) |
| `HoldoutRidge` | `holdout` | the training firms of each split | the held-out firms | the median (or mean, `holdout_split_aggregate`) Koijen test R² over the splits (`r2_holdout`) |

Each estimator separates `select_penalty` (the fixed `ridge_alpha`, or a 10-fold cross-validation over a fixed grid; under holdout one penalty per split, selected on that split's training firms) from `score` (fit and score with a given penalty); `fit` does both. The separation lets a caller select once and score many resamples. A holdout split assigns a firm to the held-out fold by a hash of its stock id and the split seed, so the same firm is held out in every model and every universe, and the copies of a firm drawn twice by a bootstrap share its fold. A split that cannot be fitted or scored (fewer than 10 training firms, no held-out firm, or a constant target on the held-out firms) raises `UnscorableSplit`.

The fit comes back as a `RidgeFit` value: the metric, the value, the penalty (the median over the splits under holdout), the number of firms and the per-split scores, which become the `eval_ridge_splits` rows.

The holdout runs hundreds of small ridge fits per period, so it does not call sklearn per fit. `cv_penalty` selects each split's penalty as `RidgeCV` with the 10-fold `KFold` would, forming each fold's centered Gram matrix once and solving every grid penalty from it, and `ridge_predict` fits the split the way `Ridge`'s Cholesky solver does; both reproduce sklearn to the last bit (under the locked sklearn and scipy versions) and hand a rejected solve to sklearn. Each firm's hash folds over all splits are computed once per estimator and reused by the penalty search, the scoring and every bootstrap resample.

`evaluators.estimation.resampling` holds the percentile bootstrap interval (a resample whose value is None is left out) and the two-sided permutation test of two means, each with its own seeded generator.

`evaluators.estimation.classification` holds the probe's `cross_validated_auc`: a stratified k-fold logistic regression scored by the mean test-fold ROC AUC, raising `UnscorableProbe` when a period cannot be scored. `evaluators.estimation.pairwise` solves the pairwise OLS from per-pattern cell statistics (`cell_statistics`, `ols_from_cells`, `term_names`), and `evaluators.estimation.panel` holds `fama_macbeth`, the time-series mean of per-period coefficients with its standard error, t statistic and p-value.

## Ridge benchmarks

[`RidgeBenchmarkEvaluator`][asset_embeddings.evaluators.benchmark.RidgeBenchmarkEvaluator] implements `evaluate_period` for a benchmark whose period is one ridge fit:

1. `target(frame)` (abstract) returns the frame the design is built from, the target values and diagnostics. A target that drops rows returns the reduced frame, so dropped rows do not enter the feature scaling; the steps it adds go to the funnel once.
2. `design(frame)` builds the feature matrix: the selected columns, scaled.
3. The estimator fits the target on the design; a target that is constant across firms is written as computed with a warning in the log (sklearn scores it 1.0).
4. The period's records are the result row, one split row per holdout split and the funnel.

The two benchmarks differ only in their target and columns:

| Benchmark | Target | Characteristics |
|---|---|---|
| RV (`RVBenchmarkEvaluator`) | the residual of the cross-sectional OLS of log book equity on log market equity; under `nonpositive_equity: "drop"` firms with non-positive book or market equity leave the period first | beta, asset growth, profitability, dividend-to-asset |
| RC (`RCBenchmarkEvaluator`) | the next-quarter return `qret_next` | beta, size, book-to-market, asset growth, profitability, momentum |

## Grouped evaluation

[`GroupedRidgeEvaluator`][asset_embeddings.evaluators.grouped.GroupedRidgeEvaluator] is a cooperating subclass: `GroupedRVEvaluator(GroupedRidgeEvaluator, RVBenchmarkEvaluator)` and `GroupedRCEvaluator(GroupedRidgeEvaluator, RCBenchmarkEvaluator)` reuse the parent's target and design and replace the period and the summary. Per period:

1. The group file's source has assigned every firm of the file to a group: quantiles or fixed bins of the value, or under `grouping: "categorical"` one group per distinct value (the state-ownership groups of §7.1). A group with fewer than 10 firms in the file, or fewer than 10 rows after the join, is skipped with a warning.
2. The penalty is selected once on the whole period (`alpha_scope: "cross_section"`) or once per (group, period) cell (`"cell"`), on the grid floored at `cv_alpha_min`.
3. The cell is scored with the target winsorized within the cell (`winsor_pct`), and a bootstrap rescores `n_bootstrap` resamples of the cell's firms at the selected penalty for the percentile interval.
4. After the last period, `summarize` compares the groups' per-period scores by permutation tests (adjacent groups, then the first against the last when there are more than two) and records each test. Under the categorical grouping the groups are the values that were scored, ordered by value.

Under holdout scoring the same code runs with `HoldoutRidge`: the cell's value is the held-out R² aggregated over the splits, each split is recorded in `eval_grouped_ridge_splits`, the interval is a bootstrap over the cell's firms of that aggregate, a cell whose splits cannot be fitted is skipped, and a resample that cannot be scored is left out of the interval and counted in the log.

## Label probe

[`LabelProbeEvaluator`][asset_embeddings.evaluators.label_probe.LabelProbeEvaluator] is a feature evaluation with RV's characteristics and the label file as its source. Per period it checks that the label is binary, scales the selected features on the whole cross section and scores them with `cross_validated_auc`. The period's records are the result row, one row per scored fold and the funnel; a period that cannot be scored writes only its funnel, with a warning. The summary logs the mean AUC over the scored periods and writes no rows. Config fields and records: [Evaluate: Label probe](../how-to/evaluate.md#label-probe).

## Pairwise similarity

[`PairwiseSimilarityEvaluator`][asset_embeddings.evaluators.pairwise_similarity.PairwiseSimilarityEvaluator] sits on the lifecycle layer: its frame is the firm universe of `PeriodAssembler` (the raw embedding joined with the benchmark's `match_on` columns, which are its `benchmark_columns`) plus the label. Per period it regresses the cosine similarity of every firm pair on the same-value indicators through the cell statistics and records the fit, one coefficient row per term and the funnel; `summarize` aggregates each term's coefficients by `fama_macbeth` into the result rows. Config fields and records: [Evaluate: Pairwise similarity](../how-to/evaluate.md#pairwise-similarity).

## The masked-LM training loop

[`BaseMLMTrainer`][asset_embeddings.trainers.mlm.BaseMLMTrainer] holds what AssetBERT pretraining, finetuning and the MP refines share. Its config is one container of `model`, `tokenizer`, `dataset`, `dataloader`, `accelerator`, `optimizer` and `train` blocks; the `train` block is the loop config (`AssetBERTTrainConfig` or `MPTrainConfig`), which declares the held-out splits.

`prepare` seeds the run, builds the tokenizer and the dataset, and splits it with `split_dataset`: training, validation and test parts of the configured shares, drawn at random from the training seed or taken as consecutive slices. Each part gets a `MaskingCollator` (`split_masking`): the validation and test loaders mask the sample at dataset index `i` from a generator seeded by the training seed and `i`, so every evaluation of a held-out loss sees the same masked inputs and epochs compare on equal terms, while the training loader draws fresh masks every epoch (with `mask_indices`, every part masks those positions). `dataset.num_repeats` replays the training part only. It then builds the loaders, the model and the optimizer and wraps them in `accelerate`.

`fit` runs the epochs. Each epoch is a training pass, a scoring pass on each held-out split the selection needs (the validation split when it exists, the test split when `best_metric` or the enabled `early_stop_metric` is `test_loss`), a report, a `run_epochs` row, a periodic save every `save_every_epochs` epochs, `on_best` when the selection loss reaches a new minimum, and the early-stopping check. After the loop, `on_fit_end` receives the loop's `FitSummary`. A metric whose split holds no rows is refused before the first epoch.

| Hook | Purpose |
|---|---|
| `model_config_class`, `make_model_preparer` | the model and how it is built |
| `train_config_class` | the loop config, and with it the available splits |
| `extra_configs` | further config blocks (the MP `run` block with its period) |
| `on_periodic_save`, `on_best`, `on_fit_end` | what is saved at a periodic epoch, at a new best epoch and after the loop |

## AssetBERT and the MP refines

`AssetBERTTrainer` (`train.bert`) saves the `accelerate` state and the static embedding at the periodic epochs, the model and its static embedding at each new best epoch, and after the loop writes the `train_summary` row, applies `keep_best_only` and optionally computes the contextual embeddings.

`MPTrainer` (`eval.mp_bert`, `eval.mp_smlm`) is a refine followed by a score. It scores the loaded model on the test split first (the baseline), runs the loop with the `accelerate` state of each new best epoch saved over the previous one, reloads the best state and scores the test split again. The score, from `evaluators.mp`, is the mean over the masked tokens of `1 + log p / log V` (0 at chance, 1 at certainty). `MPBertTrainer` refines an AssetBERT checkpoint and `MPSimplifiedMLMTrainer` fits the attention-free SimplifiedMLM head on a fixed embedding; both write the `eval_mp_results` row and the `train_summary` row.

The Word2Vec and recommender-system trainers have no epoch loop of this kind and derive from `BaseTrainer` directly.

## Adding a benchmark

A new ridge benchmark subclasses `RidgeBenchmarkEvaluator` and provides `config_class`, `task_name`, `benchmark_columns`, `char_cols`, `result_class` and `target`; a new result class in `asset_embeddings.records` gives it its table and views (see [Extending the system](records-system.md#extending-the-system)), and a thin script under `asset_embeddings/scripts/eval/` drives it. A benchmark that needs another per-period file adds a `PeriodSource` through `sources`. A benchmark that is not a ridge picks its layer by its config: one over feature columns (like the label probe) subclasses `FeatureEvaluator`, one without feature fields (like the pairwise similarity) subclasses `BaseEvaluator`; either implements `evaluate_period` itself, reusing the data layer.
