# CLI reference

Every entry point runs as a module:

```bash
uv run python -m asset_embeddings.scripts.<group>.<name> [args...]
```

Common flags of the training and evaluation scripts: `-c`/`--config` (JSON config; training & evaluation
are config-driven), `-l`/`--log` (log file path(s)), `-r`/`--result_file` (SQLite records database),
`-e`/`--experiment` (experiment family stored in `runs`), `-o`/`--override KEY=VALUE ...` (patch config fields
on the command line; on every training and evaluation script) and `-v`/`--verbose` (console DEBUG). Every boolean option is a flag: give it to switch it on, or `--no-<name>` for the ones that default to on. Pass `-h` to any script for its full help.

The **training** (`train.rs` / `train.w2v` / `train.bert`)
and **evaluation** (`eval.rv` / `eval.rc` / `eval.rc_temporal` / `eval.mp_bert` / `eval.mp_smlm` /
`eval.rv_grouped` / `eval.rc_grouped` / `eval.label_probe` / `eval.pairwise_similarity`) scripts are config-driven and
share one [common option set](#train-eval-shared); see [Train](../how-to/train.md) and
[Evaluate](../how-to/evaluate.md) for the per-task config-field references. The batch runners under
`scripts/run/` and the `runs_query` tool are at the [end of this page](#runners); the config generator `scripts/tools/fast_generator.py` is covered in [Generate configs](../how-to/generate-configs.md).

---

## Training & evaluation (shared options) {#train-eval-shared}

Training (`train.rs`, `train.w2v`, `train.bert`) and evaluation (`eval.rv`, `eval.rc`, `eval.rc_temporal`, `eval.mp_bert`, `eval.mp_smlm`, `eval.rv_grouped`, `eval.rc_grouped`, `eval.label_probe`, `eval.pairwise_similarity`) are config-driven: every model and run parameter lives in the JSON config. All of them expose the **same** six flags — there are no per-script options.

| Parameter | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | Path to the JSON config file. **Required.** |
| `--log`, `-l` | `str` (0+) | Log-file path(s). Default: no file logging. |
| `--result_file`, `-r` | `str` | SQLite path for structured results (per-period scores, diagnostics). Default: `None`. |
| `--experiment`, `-e` | `str` | Experiment family stored in the `runs` table (default: derived from the config path, e.g. `configs/main/...` gives `main`). The runners pass `-e "$EXP"`. |
| `--override`, `-o` | `str` (1+) | Patch config fields on the command line as `key=value`, dot notation for nested keys (e.g. `train.max_epochs=10 optimizer.learning_rate=0.001`). |
| `--verbose`, `-v` | flag | Console DEBUG output. Default: `INFO`. |

Note the short-flag collision with the runner wrapper: on these Python scripts `-e` is `--experiment`, whereas on `scripts/run/run_notify.sh` `-e` is `--exclude` (a marker filter); the runners themselves pass the experiment through as `-e "$EXP"` to the scripts.

For the per-task **config-field** references, see [Train](../how-to/train.md) and [Evaluate](../how-to/evaluate.md).

## Runners {#runners}

Every experiment family has single-responsibility runners `scripts/run/{train,eval}_<exp>.sh` that `source run_helper.sh` and wrap each command as `job -m <tag> ... "<description>" cmd`, the last `-m` being the config path; `DB` / `EXP` in the environment become `-r "$DB" -e "$EXP"` on every script call.
`scripts/run/run_notify.sh <runner>` runs a runner with an email report and filters its jobs by marker: `-k` / `--exclude` (comma-separated AND terms, repeatable for OR; a term with `*` or `?` is a glob over the tags) and `--filter-file` / `--exclude-file` for long lists.
The full contract, the marker conventions and the resume workflow are in [Run experiments](../how-to/run-experiments.md).

## `scripts.tools.runs_query` — filter inputs from the records database {#runs-query}

Reads one or more records databases and prints line-oriented text for `run_notify.sh --filter-file` / `--exclude-file` (or, for short lists, for `-k` / `--exclude` through command substitution). The runners tag every job with its config path, the same string the scripts store in `runs.config_file`, so these lists select exactly the runs the database reports.

| Mode | Prints |
|---|---|
| `status` (default) | one row per `(experiment, task)`: `completed= error= running=` counts and the latest start |
| `configs --status completed\|error\|running` | config paths whose **latest** run has that status, one per line |
| `missing --root DIR` | every `*.json` under `DIR` on disk without a completed run |

Global options, given before the mode: `-d/--db` (repeatable, required), `--experiment PREFIX`, `--task TASK`. Mode options, given after it: `configs --status` (default `completed`) and, on `configs` and `missing`, `--compress` (collapse a directory to `dir/*` when every config under it is in the set; checked against the filesystem, never the database alone) and `--root` (required for `missing`; default `configs` for `configs`). The first output line is a `#` comment with the query's provenance; the filter engine ignores it. The tool proposes; which jobs run is still decided by the command you write.

```bash
# resume a grid: everything without a completed run
uv run python -m scripts.tools.runs_query -d results/db/main.db missing --root configs/main --compress > /tmp/todo.txt
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh

# re-run failures only
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status error > /tmp/failed.txt
bash scripts/run/run_notify.sh --filter-file /tmp/failed.txt scripts/run/eval_main.sh
```
