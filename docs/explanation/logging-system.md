# Logging system

This document describes how the AssetEmbeddingsInChina scripts log: one process-wide setup call, standard module loggers, one job log file per config run, and one error file per failed run.

## Contents

- [Overview](#overview)
- [configure_logging](#configure_logging)
- [Module loggers](#module-loggers)
- [The job log file](#the-job-log-file)
- [Error files](#error-files)
- [Console output](#console-output)
- [Color system](#color-system)
- [Notebooks](#notebooks)
- [Distributed launches](#distributed-launches)
- [FAQ](#faq)

---

## Overview

| File | Component | Description |
|------|------|------|
| `asset_embeddings/logger.py` | `configure_logging` | The one place handlers are installed (root logger) |
| `asset_embeddings/logger.py` | `error_dir`, `write_error_block`, `failure_report`, `job_log_paths` | Error files |
| `asset_embeddings/logger.py` | `ColorParser`, `ColoredConsoleFormatter`, `ColoredHTMLFormatter` | Color for the terminal and for notebooks |
| `asset_embeddings/logger.py` | `TqdmLoggingHandler`, `SysLoggingHandler`, `NotebookLoggingHandler` | The three console channels |
| `asset_embeddings/configs/object_configs.py` | `LoggerConfig` | The parameter object of `configure_logging` |
| `asset_embeddings/records/run.py` | `Run` | Writes the error file of a failed run (see [Records system](records-system.md)) |

### Principles

- **Libraries name, the entry point configures.** Every module has `logger = logging.getLogger(__name__)` and never touches a handler. `main()` calls `configure_logging(...)` once; everything else follows from the standard logger hierarchy. No logger object is passed around.
- **The job log is the complete record of the process.** Handlers sit on the root logger, so the package's DEBUG output, the WARNINGs of third-party libraries (transformers, accelerate, gensim, torch) and captured Python warnings all land in the same file and on the same screen.
- **File DEBUG, screen INFO.** The file keeps everything; the console shows INFO and above (DEBUG with `-v`).
- **One file per config run, appended, never rotated.** A re-run of the same config adds a new block to the same file; each block starts with a `=====` header line.
- **One error file per failed run.** Written once, by `Run.__exit__` (or `failure_report` in scripts without a `Run`), with the run identity, the config path, the job log path, the command line and the traceback.
- **tqdm-safe, colored console.** Progress bars are never broken by log lines; colors are on the console only.

---

## configure_logging

```python
from asset_embeddings.logger import configure_logging

def main(args):
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    ...
```

```python
def configure_logging(
    config: Optional[LoggerConfig] = None,
    *,
    log_files: Union[str, Sequence[str], None] = None,
    console_level: Optional[str] = None,
) -> None
```

Keyword arguments override the corresponding `LoggerConfig` fields; the scripts pass only those two. What it installs, in order:

1. **Root logger.** The handlers go on the root; root level `WARNING`; `asset_embeddings` logger level `DEBUG`, and so is `__main__`, the name under which an entry-point module logs when it has something to say (`python -m asset_embeddings.scripts.eval.rv` and `python asset_embeddings/scripts/eval/rv.py` both execute the module as `__main__`; the train and eval entry scripts only drive a class from the package). Effective levels are therefore: records of `asset_embeddings.*` and of the entry-point module exist from DEBUG, records of every other logger (and of `py.warnings`) from WARNING. The command line is logged once at DEBUG, so `-v` shows it on the console.
2. **Console handler** per `console_stream`: `tqdm` (default, writes through `tqdm.write`), `std` (plain stdout) or `html` (notebook), at `console_level`.
3. **File handlers**, one `logging.FileHandler(path, mode="a", encoding="utf-8")` per path in `log_files`, at `file_level` (DEBUG by default). Parent directories are created. After the handlers are attached, one header line is written directly to each file:
   `===== 2026-09-22T03:15:44 uv run python -m asset_embeddings.scripts.eval.rv -c … -r … -e main =====`
4. **Short names.** A filter on every installed handler sets `record.short`: names under `asset_embeddings.` lose that prefix and a following `scripts.` (`asset_embeddings.scripts.train.bert` becomes `train.bert`, `asset_embeddings.preparers` becomes `preparers`, `asset_embeddings.records.store` becomes `records.store`); other names keep their first component (`transformers.modeling_utils` becomes `transformers`); `__main__` becomes the script's real short name (`train.w2v`), resolved from the module spec or the file's place in the package; `py.warnings` becomes `warnings`. The default format is `%(asctime)s [%(short)s][%(levelname)s]: %(message)s`.
5. **Python warnings.** `logging.captureWarnings(True)`: `warnings.warn(...)` becomes a WARNING record of the `py.warnings` logger, once per location (Python's default filter). No module in the package silences warnings globally; a place that must be quiet uses a local `warnings.catch_warnings()`.
6. **transformers.** HF keeps a private stderr handler on its own logger and does not propagate; when `transformers` is already imported, its default handler is removed and propagation enabled so its WARNINGs reach the root handlers.
7. **Idempotent.** The handlers this function installed are tagged; a second call removes exactly those and installs afresh (a notebook re-running its setup cell, or the test suite, never accumulates handlers).

### LoggerConfig

`LoggerConfig` is a `Config` like any other in the project, but it never appears in a JSON config: it is the parameter object of `configure_logging`.

| Field | Type | Default | Description |
|------|------|--------|------|
| `log_file` | `str \| list[str] \| None` | `None` | The job log file(s); `None` means console only. |
| `log_format` | `str` | `%(asctime)s [%(short)s][%(levelname)s]: %(message)s` | Format string; `%(short)s` is the shortened logger name, `%(name)s` still works. |
| `date_format` | `str` | `%Y-%m-%d %H:%M:%S` | Timestamp format. |
| `console_level` | `DEBUG` … `CRITICAL` | `INFO` | Minimum level on the console. |
| `console_stream` | `tqdm` / `std` / `html` | `tqdm` | Console channel. |
| `file_level` | `DEBUG` … `CRITICAL` | `DEBUG` | Minimum level in the file(s). |
| `enable_colors` | `bool` | `True` | Color on the console. |
| `color_target` | `message` / `levelname` / `format` | `format` | What the console formatter colors. |

---

## Module loggers

Every module that logs:

```python
import logging

logger = logging.getLogger(__name__)


class Dataset_Preparer(Preparer):          # asset_embeddings/preparers.py
    def prepare(self, tokenizer):
        self.logger.info("Preparing Dataset...")   # classes keep `self.logger`; it is the module logger
        ...
```

The train and eval entry scripts hold no classes: the trainers live in `asset_embeddings.trainers`, the evaluators in `asset_embeddings.evaluators`, and each `main()` only constructs one and drives it, so everything they log is a real node of the package tree (`trainers.bert`, `evaluators.base`, `preparers`, `records.run`). Within the tree, the component that does the work reports it: a dataset logs the files and records it loads, a preparer what it builds, and a trainer only the phases it owns (`Preparing Trainer...` / `Trainer Prepared.`), its own decisions (the split, the best epoch) and where it saved; it does not announce what a preparer is about to do. The trainers get `self.logger` from `BaseTrainer`, which binds the subclass's module logger. A module that does run as the entry point (a data script, a notebook cell) logs as `__main__`; `configure_logging` gives that logger the package's level and the short-name filter prints it under the script's real name, so nothing is lost there either. `self.logger` stays as an attribute because several hundred call sites use it and because a test can still install a quiet logger on one instance (`evaluator.logger = logging.getLogger("test.quiet")`). Nothing injects loggers: `Preparer` has no `set_logger`, `Run` and `RecordStore` take no `logger` argument, and the trainers and evaluators have no `set_logger` method. A message logged anywhere in the package, or by a library the process imports, reaches the same handlers.

Typical lines of one run, as they appear in the file:

```
===== 2026-09-22T03:15:44 uv run python -m asset_embeddings.scripts.train.bert -c configs/main/finetune/AssetBERT/d16/AssetBERT_d16_2023Q2.json -l logs/main/train/finetune/AssetBERT/d16/2023Q2.log -r results/db/main.db -e main =====
2026-09-22 03:15:44 [trainers.bert][INFO]: Loading configuration from `configs\main\finetune\AssetBERT\d16\AssetBERT_d16_2023Q2.json`
2026-09-22 03:15:45 [records.run][INFO]: run 3f9a1c2e-… (#17) configs/main/finetune/AssetBERT/d16/AssetBERT_d16_2023Q2.json
2026-09-22 03:15:45 [datasets.corpus][INFO]: Corpus: 6,876 portfolios, 680,469 holdings, 4,538 stocks from 1 file(s)
2026-09-22 03:15:45 [preparers][INFO]: Loading model from checkpoint: `checkpoints\pretrained\AssetBERT\d16\AssetBERT_d16_pretrained_best\model.safetensors`
2026-09-22 03:15:46 [transformers][WARNING]: Some weights of BertForMaskedLM were not initialized …
2026-09-22 03:15:46 [warnings][WARNING]: …\sklearn\metrics\_regression.py:1266: UndefinedMetricWarning: …
2026-09-22 03:15:47 [trainers.bert][DEBUG]: VRAM 702 MB allocated, 3894 MB reserved, peak 1210 MB; RAM 2105 MB
2026-09-22 03:16:02 [trainers.bert][INFO]: Epoch:1, Train Loss:5.931200 (std 0.412300), Val Loss:5.812000 (std 0.398100)
2026-09-22 03:20:41 [records.run][INFO]: run 3f9a1c2e-… completed after 297 s: peak RAM 2105 MB, peak VRAM 1210 MB
```

---

## The job log file

Every training and evaluation script takes `--log` (`-l`) with one or more paths. The train runners under `scripts/run/`, and the eval runners for MP, pass two: one under the checkpoint directory of the run (`checkpoints/.../train.log`, or `refine.log` for MP, which travels with the artifact) and one under `logs/<experiment family>/<stage>/<model>/d<dim>[/<quarter>].log`, mirroring the `configs/` layout; both receive the same content. The RV and RC eval runners pass one path per model, shared across dimensions: `logs/<experiment family>/eval/<RV|RC>/<model>[_addChars].log`.

The file is opened in append mode and never rotated (a run log is kilobytes to a few megabytes; rotation is a long-running-service idea). A re-run of the same config appends a new block; `grep '^=====' <file>` lists the runs the file holds, and the `[records.run]` line right after the header carries the `run_id` and `run_no` of the database row, so a block can be matched to `runs` and to its error file.

The runner does not capture the process's stdout/stderr: anything a C library prints directly (CUDA, bitsandbytes) is visible only in the terminal.

---

## Error files

Debugging starts from `logs/error/`. The writer lives in `asset_embeddings/logger.py`:

```python
def error_dir() -> Path                       # $AE_ERROR_DIR or logs/error
def job_log_paths() -> List[str]              # the files configure_logging installed
def write_error_block(path, fields, exc_info) # `key : value` lines, a rule, the traceback; appends
@contextmanager
def failure_report(task, *, config_file=None) # for scripts without a Run
```

**Scripts with a `Run`** (the ten train/eval scripts): `Run.__exit__` writes `logs/error/<experiment>/<task>_<run_id[:8]>_<stamp>.log` for any exception that leaves the `with Run` block, after logging the traceback once through `records.run`. Records write failures during the run, and a failing `finish_run` at exit, append to the same file first (table and record dump each); the traceback that ended the run is always the last block, so a run has at most one error file and it reads top to bottom in the order things went wrong. A failure before `run.set_config(...)` — a bad config path, an override typo — has no `runs` row yet; the file is still written, with the script stem as label (`rv_3f9a1c2e_…log`) and `run_no : -`. A `--result_file` that cannot be opened (a v4 file awaiting migration, an unwritable path) fails in `Run.from_args`, before the `with` block; `from_args` reports it the same way, so no failure of a run goes without its file.

```
run_id     : 3f9a1c2e-…
run_no     : 17
database   : results/db/main.db
task       : RV
experiment : main
config     : configs/main/eval/RV/d16/AssetBERT_d16.json
job log    : logs/main/eval/RV/AssetBERT.log
command    : uv run python -m asset_embeddings.scripts.eval.rv -c … -r … -e main
------------------------------------------------------------
Traceback (most recent call last):
  …
ValueError: char_benchmark_path resolved to 3 file(s) but benchmark_path resolved to 2 file(s). …
```

**Scripts without a `Run`** (one-off tools) wrap their body:

```python
from asset_embeddings.logger import configure_logging, failure_report

configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
with failure_report(task="my_tool", config_file=getattr(args, "config", None)):
    ...
```

On an exception this writes `logs/error/my_tool_<stamp>.log` (task, config, job log, command, traceback), logs it once, and re-raises.

The runners (`run_helper.sh`, `run_notify.sh`) collect every error file newer than the job's start marker with a recursive `find` and attach them to the failure e-mail. Setting `AE_ERROR_DIR` to another directory (a temporary one in tests, for example) keeps error files out of the repository's `logs/error/`.

---

## Console output

Three channels, chosen by `console_stream`:

| Channel | Handler | Use |
|---|---|---|
| `tqdm` (default) | `TqdmLoggingHandler` — `tqdm.write` | Scripts with progress bars: a log line is printed above the bar without breaking it |
| `std` | `SysLoggingHandler` — `sys.stdout` | Plain terminals, piped output |
| `html` | `NotebookLoggingHandler` — `IPython.display.HTML` | Notebooks (demos) |

The console formatter is `ColoredConsoleFormatter` (ANSI) or `ColoredHTMLFormatter` (CSS); the file formatter is a plain `logging.Formatter` with the same format string, so files never contain escape codes.

---

## Color system

`ColorParser` turns strings such as `"red bold"` or `"cyan dim"` into ANSI codes (terminal) or CSS (HTML).

| Level | Console | HTML |
|---|---|---|
| DEBUG | blue | purple |
| INFO | cyan | navy |
| WARNING | yellow | orange |
| ERROR | red | orangered |
| CRITICAL | red bold | darkred bold |

`color_target` selects what is colored: `message` (the message text only), `levelname` (the level only) or `format` (each component of the format string: dim timestamp, level in its level color, and a green logger name when the format uses `%(name)s`; the default format prints the short name `%(short)s`, which stays uncolored). Supported ANSI colors: black, red, green, yellow, blue, magenta, cyan, white, their `bg_` variants, and the styles bold / bright / dim / normal; HTML accepts any CSS color name or hex code.

---

## Notebooks

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.logger import configure_logging

configure_logging(LoggerConfig(console_stream="html", log_file=None, console_level="INFO"))
```

Every `asset_embeddings.*` INFO line then renders as colored HTML in the cell output; re-running the cell replaces the handlers rather than adding to them.

---

## Distributed launches

`configure_logging` reads `LOCAL_RANK`. On a non-main rank the console shows WARNING and above only and no file handler is installed, so the job log is written once, by the main process; the same rule (`is_local_main_process`) decides which rank owns the `runs` row and writes the error file of a failed run; a non-main rank logs the traceback and writes no file.

---

## FAQ

### How do I disable colors?

`configure_logging(LoggerConfig(enable_colors=False, ...))`. Files never carry colors.

### How do I see DEBUG on the screen?

`-v` / `--verbose` on any script sets `console_level="DEBUG"`; the file is at DEBUG regardless.

### A library floods the log with warnings

Python emits each warning once per location by default, so the volume is bounded. If a specific warning is known and harmless, silence it locally around the call (`with warnings.catch_warnings(): warnings.simplefilter("ignore", SomeWarning)`), not at import time.

### Where is the traceback of a failed run?

Three places, by design: the job log (once, from `records.run`), the error file under `logs/error/<experiment>/`, and `runs.error_message` in the database. The error file is the entry point; its header names the other two.
