# Records system

This document describes the AssetEmbeddingsInChina experiment records system (schema v7): the `Record` classes and the tables they define, the sample funnel, the per-table views, the `RecordStore` persistence layer, the `Run` object every script records through and the query layer.

## Contents

- [Overview](#overview)
- [Record classes](#record-classes)
- [Tables](#tables)
- [Views](#views)
- [RecordStore persistence](#recordstore-persistence)
- [Database file layout](#database-file-layout)
- [Run lifecycle and status](#run-lifecycle-and-status)
- [Query layer](#query-layer)
- [Extending the system](#extending-the-system)

---

## Overview

The records system lives under `asset_embeddings/records/`. Every train and eval script writes one `runs` row per invocation and, linked to it by `run_id`, the rows that describe what the run observed or estimated.

### Design principles

- **One fact, one home.** What a run was asked to do lives in `runs.config_content`; rows hold what was observed or estimated. No row repeats the model type, the feature set or the penalty setting: the views expose declared config fields next to the rows instead.
- **Observations, not derivations.** Rates, shares, means over periods and aggregates of the holdout splits are computed by queries and views.
- **A result row is what a paper table reads.** `metric` / `value` at the table's natural grain, the penalty actually used, and the size of the sample the fit ran on (`n_scored`) and its feature count (`n_features`).
- **Variable length is rows, not columns.** The sample funnel, the holdout splits and the principal components are rows.
- **One naming scheme.** Counts are `n_<noun>`; estimates are named for what they are; tables are named for their content.
- **One link key and a natural key.** Every table links to `runs` by `run_id` and declares a natural key that is `UNIQUE` within the run; `run_no` is a column of `runs` only, a display sequence.
- **Tables created on demand, with constraints.** A file holds only the tables of the tasks that ran; each is created from its class with `NOT NULL`, `UNIQUE`, `CHECK` and (declared, not enforced) `FOREIGN KEY` clauses, together with its views.
- **One `Run` object per script.** Identity, lifecycle (the `runs` row, closed as `completed` / `error`) and every write go through `asset_embeddings.records.Run`; scripts never touch `RecordStore` directly.

### File layout

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

### Relationship to the CLI

Every evaluation and training script takes a SQLite database path via `--result_file` (`-r`). Without it nothing is recorded. With it, the script opens a `Run` on that file and writes as it goes. The optional `--experiment` (`-e`) names the experiment family stored in `runs.experiment`; without it the family is derived from the config path (`configs/main/eval/RV/...` gives `main`, `configs/<family>/<split>/finetune/...` gives `<family>/<split>`).

```bash
uv run python -m asset_embeddings.scripts.eval.rv -c configs/main/eval/RV/d16/AssetBERT_d16.json -r results/db/main.db -e main
uv run python -m asset_embeddings.scripts.train.bert -c configs/main/finetune/AssetBERT/d16/AssetBERT_d16_2023Q2.json -r results/db/main.db -e main
```

The runner scripts under `scripts/run/` set `DB` and `EXP` once and pass `-r "$DB" -e "$EXP"` to every call.

---

## Record classes

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

`RecordField` declares a column: `type_hint` (inferred from the annotation), `default`, `required` (checked at construction and emitted as `NOT NULL`), `nullable` (overrides that), `primary_key` and `doc`. Values are type-checked on assignment (an `int` is accepted for a `float` column; a `bool` is not a number). SQLite types follow `SQLITE_TYPE_AFFINITY` (`int` `INTEGER`, `float` `REAL`, `str` `TEXT`, `bool` `INTEGER` stored as 0 / 1, `bytes` `BLOB`); an `Optional[X]` takes the type of `X`.

A record takes only its declared fields: an unknown keyword raises `TypeError`, so no column appears that the class does not declare. `run_id` is left unset by the code that builds a record; `Run.save` fills it. Every concrete class registers under its table name (`record_classes()`), which is how the store and the query layer find a class from a table.

A `Detail` class declares:

- `__table__`: its table.
- `__key__`: the key columns besides `run_id`; the DDL emits `UNIQUE (run_id, <key>)`, which is also the table's index. A one-per-run row (`TrainingSummary`, `W2VTraining`, `RSFit`) has an empty key: `UNIQUE (run_id)`.
- `__config_columns__`: name to JSON path in `runs.config_content` (nested paths such as `$.model.embedding_model` for the container configs of the MP runs). The names may not collide with the row's fields.

A `Result` adds `metric` (a key of `METRICS`) and `value`, and `__summary_keys__`: the key columns besides the period that its summary view groups by.

`METRICS`, the vocabulary of `metric`:

| metric | What `value` is |
|---|---|
| `r2` | Cross-sectional ridge R² on every firm of the period (RV explanatory, RC predictive) |
| `r2_holdout` | Koijen test R², `1 - Var(resid) / Var(y)`, on held-out firms, aggregated over the firm splits (`holdout_split_aggregate`; the mean in runs written before that field) |
| `mp_score` | Managed-portfolio score: normalized log-likelihood of the masked holdings on the test split |
| `auc` | Label probe: mean test-fold ROC AUC of the cross-validated logit |
| `fm_coef` | Pairwise similarity: Fama-MacBeth mean of a term's per-period coefficients |
| `r2_temporal` | Temporal RC: Koijen R² of the slope(s) fitted on earlier periods (`slope_window`; the previous period only in runs written before that field), applied unchanged to this period |
| `r2_temporal_aligned` | Temporal RC as `r2_temporal`, this period's embedding Procrustes-aligned onto each earlier period's |

---

## Tables

Every table below has `run_id` (the link to `runs`) as its first column; the key column list is `UNIQUE` together with `run_id`.

### `runs`

One row per script invocation, the only home of config and status.

| Column | Type | Description |
|---|---|---|
| `run_no` | `int` | Display sequence of the file (SQLite rowid alias), assigned on insert. |
| `run_id` | `str` | UUID4: the link key of every other table, in logs and in `runs_query`. |
| `experiment` | `str` | Experiment family, e.g. `main`, `timesplit_robust`. |
| `task` | `str` | `RV`, `RC`, `MP`, `GROUPED_RV`, `GROUPED_RC`, `PROBE`, `PAIRWISE`, `pretrain_BERT`, `finetune_BERT`, `pretrain_W2V`, `finetune_W2V`, `fit_RS`. |
| `script` | `str` | Module name, e.g. `asset_embeddings.scripts.eval.rv`. |
| `config_file` | `str` | Config path as given on the CLI. |
| `config_hash` | `str` | sha1 of the canonical JSON of the config. |
| `config_content` | `str` | The config snapshot; the views read the config columns from it. |
| `started_at` / `finished_at` | `str` | ISO 8601. |
| `duration_sec` | `float` | Wall time. |
| `peak_ram_mb` / `peak_vram_mb` | `float` | Peak resident memory of the process tree; peak CUDA memory (NULL without CUDA). |
| `status` | `str` | `running`, `completed` or `error`. |
| `error_message` | `str` | Traceback when `status` is `error`. |
| `git_commit` / `git_dirty` | `str` / `bool` | Checkout at run time. |
| `hostname`, `package_version` | `str` | Provenance. |
| `schema_version` | `int` | 7 (`CHECK`). |

### Evaluation results

| Table (class) | Key | Columns besides `metric`, `value` |
|---|---|---|
| `eval_rv_results` (`RVResult`) | `period` | `ridge_alpha`, `n_scored`, `n_features` |
| `eval_rc_results` (`RCResult`) | `period` | `ridge_alpha`, `n_scored`, `n_features` |
| `eval_grouped_rv_results` / `eval_grouped_rc_results` (`GroupedRVResult` / `GroupedRCResult`) | `period`, `group_label` | `ci_lower`, `ci_upper`, `ridge_alpha`, `n_scored`, `n_features`, `n_group_firms`, `group_value_mean`, `group_value_median`, `group_value_min`, `group_value_max` |
| `eval_mp_results` (`MPResult`) | `period` | `value_std`, `baseline_value`, `baseline_value_std`, `n_features` |
| `eval_probe_results` (`ProbeResult`) | `period` | `n_scored`, `n_positive`, `n_features`, `n_folds_used` |
| `eval_pairwise_results` (`PairwiseResult`) | `term` | `se`, `t`, `p`, `n_periods` |

- `ridge_alpha` is the penalty used; under holdout scoring the median of the per-split penalties, under temporal RC the mean penalty of the slopes applied (the previous period's under `slope_window: "previous"`).
- `eval_rc_results` holds two tasks: `RC` (metrics `r2`, `r2_holdout`) and `RC_temporal` (`r2_temporal`, `r2_temporal_aligned`; one row per period after the first, `n_scored` the firms of the scored period). A task name passed to the query layer reads only that task's runs.
- `n_scored` is the number of firms in the fit: the rows of the last funnel step (grouped: the rows of the cell). Under holdout scoring it counts every firm of the period, which the splits divide into training and held-out firms (`n_train`, `n_test` in `eval_ridge_splits`).
- `n_features` counts every feature column: embedding, placebo, characteristics and industry dummies.
- `group_label` is `G1` … `Gn` under quantile grouping (`G1` the lowest value), the fixed-bin label (e.g. `1`, `2-3`, `4-10`, `>10`, or the configured `bin_labels`), or under categorical grouping the value itself (an integer-valued number without its decimal, e.g. `0`, `1`); the column is not called `group` because that is an SQL keyword. `n_group_firms` counts the firms of the group file in the group; `group_value_*` summarize their grouping value (`coverage_count`) and are NULL under categorical grouping.
- A probe period that cannot be scored (a single label class, fewer labeled firms than folds, no scorable fold) has no `eval_probe_results` row; its funnel is recorded. `n_scored` counts the labeled firms, `n_positive` those with label 1.
- `eval_pairwise_results` is written after the last period, one row per term (`const`, a `match_on` column, or an interaction `a:b`); its summary key is `term`, so `v_eval_pairwise_results_summary` has one row per run and term. `t` and `p` are NULL when the standard error is 0 or undefined (a single period).
- `baseline_value` is the MP score of the loaded model before refining; the MP run's training summary is a `train_summary` row.

Config columns of the views, per class:

| Class | Config columns |
|---|---|
| `RVResult` | `model_type`, `features`, `align_chars`, `scoring`, `fixed_alpha` (`$.ridge_alpha`: NULL when the penalty is cross-validated), `normalization`, `seed`, `nonpositive_equity` |
| `RCResult` | the RV columns without `nonpositive_equity` |
| `GroupedRVResult` / `GroupedRCResult` | the RV / RC columns plus `grouping`, `n_quantiles`, `alpha_scope`, `winsor_pct`, `cv_alpha_min` |
| `MPResult` | `model_type` (`$.model.embedding_model`), `head` (`$.model.model_type`: `BERT` or `SimplifiedMLM`), `freeze_embedding`, `seed` (`$.train.seed`) |
| `ProbeResult` | `model_type`, `features`, `align_chars`, `normalization`, `seed`, `label_col`, `n_folds` |
| `PairwiseResult`, `PairwiseFit`, `PairwiseCoef` | `model_type`, `label_col`, `match_on`, `interaction`, `seed` |

`features` and `match_on` are JSON lists in `config_content`; the query layer decodes them to Python lists.

### The sample funnel (`eval_sample`)

`SampleStep`, key `period`, `step_no`; columns `step`, `n_rows`. One row per place a period's rows are read or change, numbered in the order the steps ran. File steps give the rows of the file; the other steps give the rows left after the step, so a drop is the difference to the step before. Every evaluation writes it, the baselines included.

| Step | Written by | `n_rows` |
|---|---|---|
| `embedding_file` | the assembler, with the `"embedding"` feature | rows of the embedding file |
| `benchmark_file` | the assembler | rows of the benchmark file |
| `joined` | the assembler, with the `"embedding"` feature | rows after the embedding and benchmark inner join |
| `char_file` | the assembler, when `char_benchmark_path` is read | rows of the characteristics file |
| `char_joined` | the assembler, when `char_benchmark_path` is read | rows after the characteristics join |
| `na_dropped` | the assembler | rows after dropping any row with a missing value |
| `nonpositive_equity_dropped` | RV | rows after the non-positive equity drop (unchanged under `nonpositive_equity: "keep"`) |
| `label_file` | the label probe and the pairwise similarity | labeled rows of the period's label file |
| `label_joined` | the label probe and the pairwise similarity | rows after the label inner join (unlabeled firms leave) |
| `group_file` | grouped evaluations | rows of the period's group file |

The vocabulary is `SAMPLE_STEPS`, enforced by a `CHECK`; a new step name goes into `SAMPLE_STEPS`.

### Holdout splits (`eval_ridge_splits`)

`RidgeSplit`, key `period`, `split`; columns `value` (Koijen test R² of the split), `value_sklearn` (`1 - SS_res / SS_tot`), `alpha`, `n_train`, `n_test`. One row per firm split of a holdout fit; the cross section writes none. The aggregate of `value` over a period's splits (`holdout_split_aggregate`: the median by default, the mean in runs written before that field) is the result's `value`.

### Grouped holdout splits (`eval_grouped_ridge_splits`)

`GroupedRidgeSplit`, key `period`, `group_label`, `split`; the columns of `eval_ridge_splits`. One row per firm split of a scored grouped cell under holdout scoring, for grouped RV and RC alike (the run's task tells them apart); the cross section and skipped cells write none. The aggregate of a cell's splits is that cell's `value`.

### Grouped tests (`eval_grouped_rv_tests`, `eval_grouped_rc_tests`)

`GroupedRVTest` / `GroupedRCTest`, key `group_a`, `group_b`; columns `statistic` (absolute difference of the groups' mean per-period scores), `p_value`, `n_a`, `n_b`. Adjacent groups, then the first against the last when there are more than two.

### Probe folds (`eval_probe_folds`)

`ProbeFold`, key `period`, `fold`; columns `auc` (test-fold ROC AUC), `n_train`, `n_test`, `n_test_positive`. One row per scored fold of the label probe; the mean of `auc` over a period's folds is the result's `value`.

### Pairwise fits and coefficients (`eval_pairwise_fits`, `eval_pairwise_coefs`)

`PairwiseFit`, key `period`; columns `r2` (of the pair-level regression), `n_pairs`, `mean_sim` (mean cosine similarity over the pairs), `n_firms`. `PairwiseCoef`, key `period`, `term`; column `coef`. The Fama-MacBeth aggregate of a term's `coef` over the periods is its `eval_pairwise_results` row; a premium relative to the average similarity divides that value by the mean of `mean_sim` (see [Evaluate: Pairwise similarity](../how-to/evaluate.md#pairwise-similarity)).

### Training

| Table (class) | Key | Columns |
|---|---|---|
| `run_epochs` (`Epoch`) | `epoch` | `train_loss`, `val_loss` (NULL without a validation split), `test_loss` (NULL unless a selection metric reads the test split) |
| `train_summary` (`TrainingSummary`) | none: one row per run | `epochs_run`, `best_epoch`, `best_loss`, `final_train_loss`, `final_val_loss`, `final_test_loss`, `early_stop_epoch`, `n_parameters`, `tensorboard_dir`, `checkpoint_path`, `embedding_path` |
| `train_w2v_summary` (`W2VTraining`) | none | `epochs_run`, `n_vocab`, `n_sentences`, `embedding_path` |
| `train_rs_summary` (`RSFit`) | none | `n_investors`, `n_assets`, `embedding_path` |
| `train_rs_components` (`RSComponent`) | `component` | `explained_variance_ratio` |

`train_summary` has one shape for AssetBERT pretraining and finetuning and for both MP refines: `checkpoint_path` is the best HF model (AssetBERT) or the best `accelerate` state (MP), `embedding_path` the best static embedding (AssetBERT only). The selection metric itself is `train.best_metric` of the config. The W2V and RS rows are written when the model is built and refreshed as training proceeds (W2V: after every epoch), so a failed run keeps what it reached.

---

## Views

Every `Detail` table gets a row view `v_<table>`: its rows with `run_no`, `experiment`, `task`, `config_file`, `git_commit`, `run_status` from `runs` and the class's config columns, read with `json_extract` from `runs.config_content`. A list-valued config field (`features`) arrives as its JSON text, e.g. `["embedding","chars"]`.

Every result table also gets a summary view `v_<table>_summary`, one row per `run_id`, summary key and `metric`: `mean_value`, `var_value` (the sample variance over the periods; NULL for one period), `n_periods`, plus the runs fields and config columns. The variance is computed from the deviations from the group mean, so it is never negative and stays accurate when the per-period values are close to each other. SQLite builds do not all ship `sqrt`, so the view carries the variance; `load_summary` adds `std_value`. Views hold no data: opening a file with `RecordStore` rebuilds any view whose stored definition differs from the current code, and the query API, which never writes, reads such a file through temporary views of the current definitions, so a config column added since the file was written reads as NULL.

```sql
SELECT model_type, features, mean_value, var_value, n_periods
FROM v_eval_rv_results_summary WHERE run_status = 'completed' AND align_chars = 1;

SELECT group_label, mean_value FROM v_eval_grouped_rv_results_summary WHERE run_id = ?;
```

---

## RecordStore persistence

`RecordStore(db_path)` opens a file (or `:memory:`), creating parent directories. A current file carries `PRAGMA user_version = 7`, set when its first `runs` row is written; opening a file that has tables but another `user_version` raises `RecordsSchemaError`, whose message names the checkout that reads it: commit ff9497ee for a v6 file (the same tables with `quarter` where v7 has `period`), the archived code for v5 and older. The query layer applies the same check.

| Method | What it does |
|---|---|
| `begin_run(run)` | Stamps a new file as v7, creates `runs` and its indexes (`experiment, task`; `config_hash`), inserts the row, returns `run_no`. |
| `finish_run(run_no, status, error_message, resources)` | Sets `status`, `error_message`, `finished_at`, `duration_sec` and the resource peaks. |
| `save(record, replace=False)` | Inserts one record into its class's table; `replace=True` overwrites the row with the same natural key (`INSERT OR REPLACE`). |
| `save_many(records, replace=False)` | Inserts records of one or several classes in one transaction; nothing is kept on failure. |

A table is created from its class (`get_table_ddl()`) on first use, with its row view and, for a result class, its summary view. Write methods raise; the policy for a failed write lives in `Run`.

---

## Database file layout

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

Different experiment pipelines use separate files, and every run inside a file carries its `experiment` label, so several files can be queried together. Shell scripts manage both through the `DB` and `EXP` variables:

```bash
DB="results/db/main.db"
EXP="main"
cfg="configs/main/eval/RV/d64/AssetBERT_d64.json"   # the last marker: the same string is stored in runs.config_file
job -m bert -m eval -m rv -m "$cfg" "RV eval AssetBERT d64" \
    uv run python -m asset_embeddings.scripts.eval.rv -c "$cfg" -r "$DB" -e "$EXP"
```

---

## Run lifecycle and status

### The `Run` object

`asset_embeddings/records/run.py` is the one place scripts touch the `runs` table. A `Run` carries the run's identity, its lifecycle and every write:

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

A `Run` exists from the entry point and knows only what the command line gives it. `from_args` opens the store; a `--result_file` that cannot be opened (the path, a file of another schema version) fails before the `with` block, so `from_args` reports it itself like any other failure of the run (one logged traceback, one error file) and raises. `__enter__` collects the provenance (git commit and dirty flag, hostname, package version) and touches no database. `set_config(task, config)` builds the `RunRecord` (`config_hash`, `config_content`, `started_at`, the provenance; `experiment` given or derived from the config path) and calls `begin_run`; a write before it is ignored with a WARNING. `save` and `save_many` fill `run_id` when the caller left it unset, so code builds records without it. `log_epoch` writes one `run_epochs` row. `__exit__` logs one line with the run's outcome and resource peaks, calls `finish_run` with `completed`, or with `error` and the traceback, then writes the error file; it always returns `False`, so the block's exception propagates. `KeyboardInterrupt` is recorded as `error` too, so an interrupted run never lingers as `running`. `module_name(globals())` yields the dotted module name even under `python -m`.

### The common `main()` shape

Every config-driven script (`train.bert` / `train.w2v` / `train.rs`, `eval.mp_bert` / `eval.mp_smlm`, `eval.rv` / `eval.rc`, `eval.rv_grouped` / `eval.rc_grouped`, `eval.label_probe` / `eval.pairwise_similarity`) builds its parser in `get_parser()` (the six shared flags come from `utils.add_run_args`), runs `main(get_parser().parse_args())`, and records the same way:

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

`run.set_config(task, config)` is the moment the `runs` row is inserted, once the config is loaded and the task is known (for BERT / W2V the task depends on the config: pretrain or finetune). Trainers and evaluators expose `task_name` and `attach(run)`, which sets `self.run`; inside the class `self.run` is the only records handle. An evaluator's period returns its records (results, split rows, funnel steps) and `BaseEvaluator.save` writes them in one `save_many`; the masked-LM loop calls `self.run.log_epoch(epoch, train_loss, val_loss=..., test_loss=...)` and the trainer writes its `train_summary` row after the loop. A class constructed without a CLI holds a placeholder `Run` without a store, so it can be driven from tests or notebooks and every write is a no-op. Under a distributed launch only the local main process gets a store (`is_local_main_process()`), so the rows are written once.

### Where the status lives

`status` and `error_message` exist only on the `runs` row. Result and detail tables never contain error rows: a period that failed has no rows at all, and the periods completed before the failure keep theirs.

```sql
-- Failed runs of one experiment family, with their traceback
SELECT run_no, run_id, task, config_file, error_message FROM runs
WHERE experiment = 'main' AND status = 'error';

-- Runs still marked running (crashed hard, or in progress)
SELECT run_no, run_id, task, started_at, hostname FROM runs WHERE status = 'running';
```

`scripts/tools/runs_query.py` answers the same questions from the shell (`status`, `configs --status error`, `missing --root configs/...`) and its `missing` mode feeds the runner filters, which is how a failed or unfinished config is re-run.

```
with Run.from_args(...) as run:            ← store opened, provenance collected
    run.set_config(task, config)           ← runs row, status='running'
    period 1 → save_many(results, splits, funnel) ← in DB
    period 2 → save_many(results, splits, funnel) ← in DB
    period 3 → ERROR                      ← __exit__: status='error', traceback; re-raised
    period 4 → never runs
```

### Failure policy of the records layer

The records layer must never cost a run its computation, and never hide a failure of the computation. Three phases, one place (`Run`):

1. **Start: fail fast.** Anything that fails when the store opens (`Run.from_args`: the file cannot be opened, the schema version is not 6) or when the row is inserted (`set_config`) propagates before any computation starts; the runner records FAIL and no GPU time is spent. Both leave the run's error file.
2. **During the run: best effort, never abort.** `save`, `save_many` and `log_epoch` catch `sqlite3.Error` and `RecordsSchemaError`. The first failure is a WARNING plus a block appended to the run's error file (the run's header fields, the tables, the records, the traceback), which the runner's failure e-mail attaches; later failures are counted and logged at DEBUG. The computation continues.
3. **End: two rules.** `__exit__` never suppresses the block's exception, and a failing `finish_run` never replaces it: any exception leaving the block is logged once with its traceback, the `finish_run` error (if any) is recorded the same way, the exception's own block closes the error file, then the original exception propagates. On the success path a failing `finish_run` is logged the same way and the script exits normally. If any write failed, one WARNING at exit summarizes the count and the error file. Such a run stays `running` in the database; `runs_query status` shows it and `missing` re-selects its config, which is the recovery path.

### Failures before the `runs` row exists

`load_configs` and `load_overrides` run inside the `with Run` block but before `run.set_config(...)`, so a bad config path or an override typo leaves **no `runs` row**: there is nothing to record about a run whose config never loaded. `Run.__exit__` still writes the error file, labeled with the script's stem instead of the task (`logs/error/<experiment>/rv_<run_id[:8]>_<stamp>.log`, `run_no : -`), and logs the traceback into the job log. `runs_query missing` still lists the config, since it has no completed row. Config cross-constraints are part of loading, so they also fail before the `runs` row exists: an unknown key, an `embedding_path` that disagrees with `features`, or a path list whose length differs from `periods` is rejected inside `load_configs`. Errors raised while evaluating (a missing file, a missing column) happen after `set_config` and are recorded as `error` like any other exception.

---

## Query layer

`asset_embeddings/records/query.py` is the read path for analysis scripts. It reads one or several files (a `source_db` column names the file), returns pandas DataFrames, and reads every table through its view, so each row carries the runs fields and, for a result, the config columns; filters may name any of them and are pushed down to SQL.

| Function | Returns |
|---|---|
| `load_runs(dbs, **filters)` | `runs` rows. |
| `load_results(dbs, task_or_table, **filters)` | Result rows of a task (`RV`, `RC`, `RC_temporal`, `MP`, `GROUPED_RV`, `GROUPED_RC`, `PROBE`, `PAIRWISE`), a result table or a result class. A task name reads only that task's runs; a table or class reads every task that writes the table. |
| `load_summary(dbs, task_or_table, **filters)` | The summary view with `std_value` added. |
| `load_details(dbs, table, **filters)` | Any record table by name or class, in key order. |
| `load_sample`, `load_splits`, `load_grouped_splits`, `load_epochs` | `eval_sample`, `eval_ridge_splits`, `eval_grouped_ridge_splits`, `run_epochs`. |
| `load_config(db, run_id)` | The config snapshot of one run as a dict. |

Filters: a scalar is an equality, `None` is `IS NULL`, a tuple or set is a membership test (`IN`), a boolean compares as 0 / 1. On a list-valued config column (`features`) a list is compared as a whole (`features=["embedding", "chars"]`) and a tuple of lists is a membership test; the column comes back as a Python list. An unknown column raises `KeyError` listing the available ones.

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

Relational work (the join to `runs`, the config columns, the filters) happens in SQL, per file, parameterized and validated against the view's columns; pandas only concatenates the per-file frames. Files are read one at a time rather than `ATTACH`-ed together because SQLite caps attached databases at ten by default and `results/db` holds more.

---

## Extending the system

### Adding a result table

1. Declare the class in `asset_embeddings/records/eval.py` (or a new module imported by `__init__.py`):

```python
class NewTaskResult(Result):
    """New task score, one row per period."""

    __table__ = "eval_newtask_results"
    __key__ = ("period",)
    __config_columns__ = {"model_type": "$.model_type", "seed": "$.seed"}

    period: str = RecordField(required=True, doc="Period label")
    n_scored: int = RecordField(doc="Rows in the fit")
```

2. Add its metric to `METRICS` when it is a new quantity, export the class, and map its task in `RESULT_TASKS` (`records/query.py`).
3. Build the rows where the period is evaluated and return them in the `PeriodOutcome`; `BaseEvaluator.save` writes them. Record the period's sample through the funnel (`pf.sample.add(step, n_rows)`; a new step name goes into `SAMPLE_STEPS`).

### Adding a training detail

Declare a `Detail` with `__table__` and an empty `__key__` for a one-per-run row, write it with `self.run.save(record, replace=True)` (refreshed as training proceeds) and per-epoch losses with `self.run.log_epoch(...)`. The run's status is never a field of the row.

### Notes

- **Naming**: a result table is `eval_<task>_results`; other tables are named for their content (`eval_sample`, `eval_ridge_splits`, `train_summary`).
- **No free columns**: a record takes only its declared fields; add a `RecordField` to add a column, and a config column to expose a config field.
- **Booleans** are stored as 0 / 1.
