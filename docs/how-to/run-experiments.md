# Run experiments

> Part of [AssetEmbeddingsInChina](../index.md). See also [Installation](install.md), [Training](train.md), [Evaluation](evaluate.md), [Generate configs](generate-configs.md).

Every experiment family in the repository is driven by a hand-written bash runner under `scripts/run/`. A runner is a flat list of `job` calls, one per config file, sourced on top of `scripts/run/run_helper.sh`, and it is launched through `scripts/run/run_notify.sh`, which adds the filter flags and an e-mail on completion. This page documents that contract: the file layout, the `job` function and its markers, the filters, the `DB`/`EXP` variables, the wrapper, where logs and error files land, and how to build re-run lists from the records database. Every command below is taken from the runner files.

## Layout and naming

Runners are named `scripts/run/{train,eval}_<exp>.sh`, one responsibility per file: `train_main.sh` trains the models of the `main` family, `eval_main.sh` evaluates them. An experiment family therefore consists of a `templates/<exp>.json` preset, which [`fast_generator config -f`](generate-configs.md) expands into `configs/<exp>/…`, and one or two runners that walk those configs (`train_<exp>.sh` and `eval_<exp>.sh`). The header comment of each runner states what it runs, the marker vocabulary and a usage line, so `head -25 scripts/run/eval_main.sh` is the quickest orientation.

```
scripts/run/
├── run_helper.sh          # job, print_summary, filters, error e-mails (sourced, never run)
├── run_notify.sh          # wrapper: filters, throttling, completion e-mail
├── train_main.sh          # train_<exp>.sh
├── eval_main.sh           # eval_<exp>.sh
└── …                      # train_timesplit_robust.sh, eval_coverage.sh, …
```

A runner starts with the `source` line, then sets its variables, then lists its jobs and ends with `print_summary`:

```bash
source scripts/run/run_helper.sh

DB="results/db/main.db"
EXP="main"  # runs.experiment; the --experiment flag of every script below
DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
```

`run_helper.sh` requires bash 4.3 or newer (associative arrays); runners are always started from the repository root, since every path in them is repo-relative.

## The `job` function and its markers

`job` is the only thing a runner calls. Its signature is `job [-m tag]... "description" command args...`: it runs the command, prints `[PASS]`, `[FAIL]` or `[SKIP]` with the description, counts the outcome, and on failure collects the new error files and mails them (see below). It returns 0 after a failure, so the loop continues with the next config; the only exception is `-M`/`-x` (below), where the failure that reaches the limit prints the summary and exits the runner with code 2. From `train_main.sh`:

```bash
for dim in $DIMS; do
    cfg="configs/main/pretrained/AssetBERT/AssetBERT_d${dim}_pretrained.json"
    job -m bert -m pretrain -m "d${dim}" -m "$cfg" "BERT pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "$cfg" \
            --log "checkpoints/pretrained/AssetBERT/d${dim}/train.log" \
                  "logs/main/train/pretrained/AssetBERT/d${dim}.log" \
            -r "$DB" -e "$EXP"
done
```

Each `-m <tag>` attaches a marker to the job; markers are lowercased, so `-m "${quarter,,}"` and `-m 2023Q2` mean the same thing. The convention is that the last `-m` carries the config path exactly as it is passed to `--config`, repo-relative with forward slashes. That string is what the scripts store in `runs.config_file`, so the records database and the filters speak the same key. The other markers name the model (`bert`, `w2v`, `rs`, `rs_binary`, …), the stage (`pretrain`, `finetune`, `eval`, `rv`, `rc`, `mp`), the dimension (`d64`) and the quarter (`2023q2`); each runner's header lists its vocabulary. A job without any `-m` is unconditional and runs under every filter.

The `--log` flag receives two paths from the train runners: one under the checkpoint directory of the run (`checkpoints/…/train.log`, so the log travels with the artifact) and one under `logs/`, mirroring the `configs/` layout. Eval runners pass one path under `logs/` for RV/RC and two (a `refine.log` under `checkpoints/refine/…` plus the `logs/` copy) for MP.

## Filters

Filters are given to `run_notify.sh`, which exports them as environment variables that `run_helper.sh` parses once when it is sourced. Four inputs exist:

| Flag | Environment variable | Meaning |
|---|---|---|
| `-k, --filter <a,b>` | `AE_FILTER_INCLUDE` | include group: all comma-separated terms must be present (AND); repeat the flag for OR |
| `-e, --exclude <a,b>` | `AE_FILTER_EXCLUDE` | exclude terms: any match skips the job; checked before the include groups |
| `--filter-file <path>` | `AE_FILTER_INCLUDE_FILE` | one AND-group per line, lines OR-ed; blank lines and `#` comments ignored |
| `--exclude-file <path>` | `AE_FILTER_EXCLUDE_FILE` | one term per line |

A term is compared against the job's markers. A term without `*` or `?` is an exact set lookup, so a file with thousands of config paths costs nothing; a term containing `*` or `?` is a bash glob matched against every tag, which is how a whole directory of configs is selected through the config-path marker. Quote globs so the shell does not expand them.

```bash
# Only BERT pretraining
bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_main.sh

# BERT or W2V finetune
bash scripts/run/run_notify.sh -k bert,finetune -k w2v,finetune scripts/run/train_main.sh

# Everything except the RS variants
bash scripts/run/run_notify.sh --exclude rs_binary,rs_ranks,rs_level0,rs_levelmin scripts/run/train_main.sh

# One directory of configs (glob over the config-path tag)
bash scripts/run/run_notify.sh -k 'configs/main/finetune/AssetBERT/*' scripts/run/train_main.sh

# Long lists from a file (see the runs_query section)
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh
```

Use files for long lists: a single argument or environment string is limited to 128 KiB on Linux.

## `DB` and `EXP`

Each train and eval runner sets two variables at the top (or per window or disclosure level, in the `sliding_window` and `disclosure_regime` runners, which cover several labels) and passes them to every CLI as `-r "$DB" -e "$EXP"`. `DB` is the SQLite records database (`--result_file`), one per experiment family under `results/db/`; `EXP` is the family name stored in `runs.experiment` (`--experiment`), which also names the subdirectory of `logs/error/` a failed run writes to. `train_main.sh` hard-codes both; `eval_main.sh` reads them from the environment with a default, so one eval runner can be pointed at another database or experiment label without editing it:

```bash
DB="${DB:-results/db/main.db}"  # override with DB=... in the environment
EXP="${EXP:-main}"  # runs.experiment; the --experiment flag of every script below (override with EXP=...)
```

```bash
DB=results/db/main_rerun.db EXP=main_rerun bash scripts/run/run_notify.sh -k rv scripts/run/eval_main.sh
```

The six flags every train/eval CLI accepts are `-c/--config`, `-l/--log`, `-r/--result_file`, `-e/--experiment`, `-o/--override` and `-v/--verbose` (see the [CLI reference](../reference/cli.md#train-eval-shared)). Most runners use the first four; `-o` patches config fields on the command line and appears where a runner varies one setting without a separate config (`eval_coverage.sh` selects its grouping with `-o grouping=quantile`).

## `run_notify.sh`

`run_notify.sh [OPTIONS] <script.sh> [SCRIPT_ARGS...]` runs one runner under `bash`, reads the counts that `print_summary` leaves behind, and sends one e-mail through `scripts/tools/smtp.py` (`uv run python -m scripts.tools.smtp -t … -s … -b … -a …`) when the runner exits. The recipient is `NOTIFY_EMAIL` from the environment or `.env` (no e-mail is sent without it); the SMTP host, port and credentials come from `.env` as well. Options come before the runner path; anything after it is passed to the runner.

| Option | Default | Environment variable | Meaning |
|---|---|---|---|
| `-k, --filter`, `-e, --exclude`, `--filter-file`, `--exclude-file` | none (every job runs) | `AE_FILTER_*` | Select jobs by their markers; see [Filters](#filters). Terms are lowercased. |
| `-x` | off | — | Stop at the first failed job; the same as `-M 1`. |
| `-M, --max-errors <N>` | `0` (no limit) | `AE_MAX_ERRORS` | Stop the batch after N failed jobs; the runner then exits with code 2. |
| `-T, --throttle <seconds>` | `0` (each failure mailed at once) | `AE_ERROR_THROTTLE` | Buffer the per-failure e-mails and send them in batches, at most one every N seconds; a batch with several error files attaches them as one zip. |
| `-v, --verbose` | normal output | `AE_VERBOSE=2` | Print the full command line of each job. |
| `-q, --quiet` | normal output | `AE_VERBOSE=0` | Print only the PASS / FAIL / SKIP lines. |
| `-a, --attach <file>` | none | — | Attach a file to the completion e-mail, e.g. the result database; repeatable. |
| `-h, --help` | | | Print the usage text and exit. |

An option on the command line wins over its environment variable. The subject of the completion e-mail is tagged by the runner's exit code:

| Exit code | Tag | Meaning |
|---|---|---|
| 0 | `[SUCCESS]` | every job passed |
| 1 | `[PARTIAL]` | ran to completion, some jobs failed |
| 2 | `[FAILED]` | aborted early by `-x` or `-M` |

```bash
# Full run, result database attached to the e-mail
bash scripts/run/run_notify.sh -a results/db/main.db scripts/run/train_main.sh

# Stop on first failure
bash scripts/run/run_notify.sh -x scripts/run/train_main.sh

# Verbose, throttled error e-mails, abort after 5 failures
bash scripts/run/run_notify.sh -v -T 300 -M 5 scripts/run/train_main.sh
```

One letter collides: `-e` on `run_notify.sh` is `--exclude`, while `-e` on every train/eval CLI is `--experiment`. The two never meet on one command line, because the wrapper's flags come before the runner path and the CLI's flags are inside the runner, but a runner should not be run bare with `-e` expecting a filter, and `-e "$EXP"` inside a `job` line is never an exclude. When in doubt write `--exclude` for the wrapper and `--experiment` for the CLI.

The runner may also be executed directly with `bash scripts/run/train_main.sh`; then no filter is applied and no completion e-mail is sent, but `run_helper.sh` still mails each failure on its own, since it reads `NOTIFY_EMAIL` from `.env` itself.

## Logs and error files

Three kinds of files come out of a batch. The job log is what `--log` names: train and MP jobs write `logs/<family>/<stage>/<model>/d<dim>[/<quarter>].log` (`logs/main/train/finetune/AssetBERT/d64/2023Q2.log`) plus the copy next to the checkpoint, and RV/RC jobs write one log per model shared across dimensions, `logs/<family>/eval/<RV|RC>/<model>[_addChars].log` (`logs/main/eval/RV/AssetBERT.log`). The job log holds everything at DEBUG level; the console shows INFO unless the CLI is run with `-v`.

Error files are the entry point for debugging. A failed train or eval run writes exactly one file, `logs/error/<experiment>/<task>_<run_id[:8]>_<stamp>.log`, with the config, the job log path, the command and the traceback; the `<experiment>` directory is the `EXP` the runner passed. Data scripts, which have no `Run`, write `logs/error/<task>_<stamp>.log` at the top level. `job` collects every error file newer than the moment it started and attaches each one to the failure e-mail; with `-T` the failures are buffered and each batch e-mail zips its error files when there are several. `run_notify.sh` does the same for the whole batch at the end, zipping when there are several. The mechanism and the file format are in [Logging system](../explanation/logging-system.md).

The third file is the records database itself, `results/db/<family>.db`, which carries one `runs` row per job with its status (`completed`, `error`, `running`), the config, the provenance and the error message; see [Records system](../explanation/records-system.md).

## Status and re-run lists with `runs_query`

`scripts/tools/runs_query.py` reads one or more records databases and prints text that the filters accept. The runners tag every job with its config path and the scripts store the same string in `runs.config_file`, so a list printed by the tool selects exactly the runs the database reports.

```bash
# counts per (experiment, task): completed / error / running and the latest start
uv run python -m scripts.tools.runs_query -d results/db/main.db status

# resume a grid: every *.json under configs/main without a completed run
uv run python -m scripts.tools.runs_query -d results/db/main.db missing --root configs/main --compress > /tmp/todo.txt
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh

# re-run failures only: config paths whose latest run has status error
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status error > /tmp/failed.txt
bash scripts/run/run_notify.sh --filter-file /tmp/failed.txt scripts/run/eval_main.sh

# skip what is already done
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status completed > /tmp/done.txt
bash scripts/run/run_notify.sh --exclude-file /tmp/done.txt scripts/run/eval_main.sh
```

`-d/--db` is repeatable, `--experiment PREFIX` and `--task TASK` narrow the query, and `--compress` collapses a directory to `dir/*` when every config under it on disk is in the set, so the file stays short. The first line of the output is a `#` comment with the query's provenance, which the filter engine ignores. The tool proposes a list; which jobs actually run is decided by the command you write around it. The full option table is in the [CLI reference](../reference/cli.md#runs-query).
