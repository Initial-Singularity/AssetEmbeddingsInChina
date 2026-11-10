# CLI 参考

每个入口都以模块形式运行：

```bash
uv run python -m asset_embeddings.scripts.<group>.<name> [args...]
```

训练与评估脚本的通用参数：`-c`/`--config`（JSON 配置；训练与评估是配置驱动的）、`-l`/`--log`（日志文件路径）、`-r`/`--result_file`（SQLite 记录库）、`-e`/`--experiment`（存入 `runs` 的实验族）、`-o`/`--override KEY=VALUE ...`（在命令行上修补配置字段；每个训练与评估脚本都有）与 `-v`/`--verbose`（控制台 DEBUG）。所有布尔参数都是开关：给出即打开；默认打开的参数用 `--no-<名称>` 关闭。给任意脚本传 `-h` 可看其完整帮助。

**训练**（`train.rs` / `train.w2v` / `train.bert`）与**评估**（`eval.rv` / `eval.rc` / `eval.rc_temporal` / `eval.mp_bert` / `eval.mp_smlm` / `eval.rv_grouped` / `eval.rc_grouped` / `eval.label_probe` / `eval.pairwise_similarity`）脚本是配置驱动的，并共享一套[通用选项集](#train-eval-shared)；逐任务的配置字段参考见 [训练](../how-to/train.md) 与 [评估](../how-to/evaluate.md)。`scripts/run/` 下的批量 runner 以及 `runs_query` 工具在[本页末尾](#runners)；配置生成器 `scripts/tools/fast_generator.py` 见 [生成配置](../how-to/generate-configs.md)。

---

## 训练与评估（共享选项） {#train-eval-shared}

训练（`train.rs`、`train.w2v`、`train.bert`）与评估（`eval.rv`、`eval.rc`、`eval.rc_temporal`、`eval.mp_bert`、`eval.mp_smlm`、`eval.rv_grouped`、`eval.rc_grouped`、`eval.label_probe`、`eval.pairwise_similarity`）是配置驱动的：每个模型与运行参数都在 JSON 配置中。它们都暴露**同一套**六个参数——没有逐脚本的专属选项。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置文件路径。**必填。** |
| `--log`, `-l` | `str`（0+） | 日志文件路径。默认：不写文件日志。 |
| `--result_file`, `-r` | `str` | 存放结构化结果（逐期分数、诊断）的 SQLite 路径。默认：`None`。 |
| `--experiment`, `-e` | `str` | 存入 `runs` 表的实验族（默认由配置路径推导，如 `configs/main/...` 得到 `main`）。runner 会传 `-e "$EXP"`。 |
| `--override`, `-o` | `str`（1+） | 在命令行上以 `key=value` 修补配置字段，嵌套键用点号（如 `train.max_epochs=10 optimizer.learning_rate=0.001`）。 |
| `--verbose`, `-v` | flag | 控制台 DEBUG 输出。默认：`INFO`。 |

注意与 runner 包装脚本的短参数冲突：在这些 Python 脚本上 `-e` 是 `--experiment`，而在 `scripts/run/run_notify.sh` 上 `-e` 是 `--exclude`（标记过滤器）；runner 自身会以 `-e "$EXP"` 把实验族传给脚本。

逐任务的**配置字段**参考见 [训练](../how-to/train.md) 与 [评估](../how-to/evaluate.md)。

## Runner {#runners}

每个实验族都有单一职责的 runner `scripts/run/{train,eval}_<exp>.sh`，它们 `source run_helper.sh` 并把每条命令包成 `job -m <tag> ... "<description>" cmd`，最后一个 `-m` 是配置路径；环境中的 `DB` / `EXP` 成为每次脚本调用上的 `-r "$DB" -e "$EXP"`。
`scripts/run/run_notify.sh <runner>` 带邮件报告地运行一个 runner，并按标记过滤其 job：`-k` / `--exclude`（逗号分隔的 AND 项，可重复以表示 OR；含 `*` 或 `?` 的项是对标记的 glob）以及用于长列表的 `--filter-file` / `--exclude-file`。
完整约定、标记惯例与续跑流程见 [运行实验](../how-to/run-experiments.md)。

## `scripts.tools.runs_query` — 从记录库生成过滤输入 {#runs-query}

读取一个或多个记录库，打印按行组织的文本，供 `run_notify.sh --filter-file` / `--exclude-file` 使用（短列表也可经命令替换交给 `-k` / `--exclude`）。runner 给每个 job 打上其配置路径标记，与脚本存入 `runs.config_file` 的字符串相同，因此这些列表能精确选中记录库报告的那些运行。

| 模式 | 输出 |
|---|---|
| `status`（默认） | 每个 `(experiment, task)` 一行：`completed= error= running=` 计数与最近一次开始时间 |
| `configs --status completed\|error\|running` | **最近一次**运行为该状态的配置路径，每行一个 |
| `missing --root DIR` | 磁盘上 `DIR` 下没有 completed 运行的每个 `*.json` |

全局选项写在模式之前：`-d/--db`（可重复，必填）、`--experiment PREFIX`、`--task TASK`。模式选项写在模式之后：`configs --status`（默认 `completed`），以及 `configs` 与 `missing` 都接受的 `--compress`（当某目录下所有配置都在集合内时折叠为 `dir/*`；以文件系统为准，绝不只看数据库）和 `--root`（`missing` 必填；`configs` 默认 `configs`）。输出首行是带查询来源的 `#` 注释，过滤引擎会忽略它。工具只提议，跑什么仍由你写的命令决定。

```bash
# 续跑一个网格：所有没有 completed 运行的配置
uv run python -m scripts.tools.runs_query -d results/db/main.db missing --root configs/main --compress > /tmp/todo.txt
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh

# 只重跑失败的
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status error > /tmp/failed.txt
bash scripts/run/run_notify.sh --filter-file /tmp/failed.txt scripts/run/eval_main.sh
```
