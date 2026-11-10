# 运行实验

> 隶属于 [AssetEmbeddingsInChina](../index.md)。另见 [安装](install.md)、[训练](train.md)、[评估](evaluate.md)、[生成配置](generate-configs.md)。

仓库中的每个实验族都由 `scripts/run/` 下一个手写的 bash runner 驱动。runner 是一份扁平的 `job` 调用列表，每个配置文件对应一次调用，建立在 `source` 进来的 `scripts/run/run_helper.sh` 之上；它通过 `scripts/run/run_notify.sh` 启动，后者附加过滤参数与完成邮件。本页记录这一约定：文件布局、`job` 函数及其标记、过滤器、`DB`/`EXP` 变量、包装脚本、日志与错误文件的位置，以及如何从记录数据库构造重跑列表。下面每条命令都取自 runner 文件。

## 布局与命名

runner 命名为 `scripts/run/{train,eval}_<exp>.sh`，一个文件一项职责：`train_main.sh` 训练 `main` 族的模型，`eval_main.sh` 评估它们。因此一个实验族由一份 `templates/<exp>.json` 预设（[`fast_generator config -f`](generate-configs.md) 把它展开为 `configs/<exp>/…`）和一到两个遍历这些配置的 runner 组成（`train_<exp>.sh` 与 `eval_<exp>.sh`）。每个 runner 的头部注释说明它运行什么、标记词表和一行用法，所以 `head -25 scripts/run/eval_main.sh` 是最快的入门方式。

```
scripts/run/
├── run_helper.sh          # job、print_summary、过滤器、错误邮件（只 source，不直接运行）
├── run_notify.sh          # 包装脚本：过滤、节流、完成邮件
├── train_main.sh          # train_<exp>.sh
├── eval_main.sh           # eval_<exp>.sh
└── …                      # train_timesplit_robust.sh、eval_coverage.sh、…
```

runner 以 `source` 行开头，随后设置变量，再列出各个 job，最后以 `print_summary` 结束：

```bash
source scripts/run/run_helper.sh

DB="results/db/main.db"
EXP="main"  # runs.experiment; the --experiment flag of every script below
DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
```

`run_helper.sh` 需要 bash 4.3 或更新版本（关联数组）；runner 总是从仓库根目录启动，因为其中每个路径都是相对仓库根的。

## `job` 函数及其标记

`job` 是 runner 唯一调用的东西。签名为 `job [-m tag]... "description" command args...`：它运行命令，连同描述打印 `[PASS]`、`[FAIL]` 或 `[SKIP]`，统计结果，失败时收集新产生的错误文件并发邮件（见下文）。失败后它返回 0，所以循环继续处理下一个配置；唯一的例外是 `-M`/`-x`（见下文）：达到上限的那次失败会打印汇总，并以退出码 2 结束 runner。取自 `train_main.sh`：

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

每个 `-m <tag>` 给 job 附加一个标记；标记会转成小写，所以 `-m "${quarter,,}"` 与 `-m 2023Q2` 含义相同。约定是最后一个 `-m` 携带配置路径，与传给 `--config` 的字符串完全一致，相对仓库根、使用正斜杠。这个字符串正是脚本存入 `runs.config_file` 的内容，因此记录数据库与过滤器使用同一把键。其余标记命名模型（`bert`、`w2v`、`rs`、`rs_binary`、…）、阶段（`pretrain`、`finetune`、`eval`、`rv`、`rc`、`mp`）、维度（`d64`）和季度（`2023q2`）；每个 runner 的头部列出自己的词表。不带任何 `-m` 的 job 是无条件的，在任何过滤下都会运行。

训练 runner 给 `--log` 传两个路径：一个在本次运行的 checkpoint 目录下（`checkpoints/…/train.log`，日志随产物一起走），一个在 `logs/` 下，镜像 `configs/` 的布局。评估 runner 对 RV/RC 传一个 `logs/` 下的路径，对 MP 传两个（`checkpoints/refine/…` 下的 `refine.log` 加 `logs/` 下的副本）。

## 过滤器

过滤器交给 `run_notify.sh`，它把它们导出为环境变量，`run_helper.sh` 在被 source 时解析一次。共有四种输入：

| 参数 | 环境变量 | 含义 |
|---|---|---|
| `-k, --filter <a,b>` | `AE_FILTER_INCLUDE` | 包含组：逗号分隔的各项必须全部出现（AND）；重复该参数表示 OR |
| `-e, --exclude <a,b>` | `AE_FILTER_EXCLUDE` | 排除项：任一匹配即跳过该 job；在包含组之前检查 |
| `--filter-file <path>` | `AE_FILTER_INCLUDE_FILE` | 每行一个 AND 组，各行之间 OR；空行与 `#` 注释忽略 |
| `--exclude-file <path>` | `AE_FILTER_EXCLUDE_FILE` | 每行一个排除项 |

一个项与 job 的标记比较。不含 `*` 或 `?` 的项是精确的集合查找，所以一份含数千个配置路径的文件几乎没有开销；含 `*` 或 `?` 的项是对每个标记做 bash glob 匹配，借助配置路径标记即可选中整个配置目录。给 glob 加引号，避免 shell 先行展开。

```bash
# 只跑 BERT 预训练
bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_main.sh

# BERT 或 W2V 微调
bash scripts/run/run_notify.sh -k bert,finetune -k w2v,finetune scripts/run/train_main.sh

# 除 RS 各变体以外的全部
bash scripts/run/run_notify.sh --exclude rs_binary,rs_ranks,rs_level0,rs_levelmin scripts/run/train_main.sh

# 一个配置目录（对配置路径标记做 glob）
bash scripts/run/run_notify.sh -k 'configs/main/finetune/AssetBERT/*' scripts/run/train_main.sh

# 长列表来自文件（见 runs_query 一节）
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh
```

长列表请用文件：Linux 上单个参数或环境变量字符串的上限是 128 KiB。

## `DB` 与 `EXP`

每个训练与评估 runner 在顶部设置两个变量（`sliding_window` 与 `disclosure_regime` 的 runner 覆盖多个标签，则按窗口或披露级别分别设置），并以 `-r "$DB" -e "$EXP"` 传给每个 CLI。`DB` 是 SQLite 记录数据库（`--result_file`），每个实验族在 `results/db/` 下各有一份；`EXP` 是存入 `runs.experiment` 的族名（`--experiment`），也是失败运行写入 `logs/error/` 时所用的子目录名。`train_main.sh` 把两者写死；`eval_main.sh` 从环境读取并带默认值，因此不改文件就能把一个评估 runner 指向另一个数据库或实验标签：

```bash
DB="${DB:-results/db/main.db}"  # override with DB=... in the environment
EXP="${EXP:-main}"  # runs.experiment; the --experiment flag of every script below (override with EXP=...)
```

```bash
DB=results/db/main_rerun.db EXP=main_rerun bash scripts/run/run_notify.sh -k rv scripts/run/eval_main.sh
```

每个训练/评估 CLI 接受的六个参数是 `-c/--config`、`-l/--log`、`-r/--result_file`、`-e/--experiment`、`-o/--override` 和 `-v/--verbose`（见 [CLI 参考](../reference/cli.md#train-eval-shared)）。多数 runner 只用前四个；`-o` 在命令行上补丁配置字段，用于 runner 需要在不另建配置的情况下改变某一项设置的场合（`eval_coverage.sh` 就以 `-o grouping=quantile` 选择分组方式）。

## `run_notify.sh`

`run_notify.sh [OPTIONS] <script.sh> [SCRIPT_ARGS...]` 用 `bash` 运行一个 runner，读取 `print_summary` 留下的计数，并在 runner 退出时通过 `scripts/tools/smtp.py`（`uv run python -m scripts.tools.smtp -t … -s … -b … -a …`）发送一封邮件。收件人是环境变量或 `.env` 中的 `NOTIFY_EMAIL`（未设置则不发邮件）；SMTP 主机、端口与凭据同样来自 `.env`。选项写在 runner 路径之前；路径之后的参数原样传给 runner。

| 选项 | 默认 | 环境变量 | 含义 |
|---|---|---|---|
| `-k, --filter`、`-e, --exclude`、`--filter-file`、`--exclude-file` | 无（运行所有 job） | `AE_FILTER_*` | 按标记选择 job，见[过滤器](#过滤器)。过滤项会转为小写。 |
| `-x` | 关 | — | 第一个 job 失败即停；等价于 `-M 1`。 |
| `-M, --max-errors <N>` | `0`（不限） | `AE_MAX_ERRORS` | N 个 job 失败后中止整批；runner 以退出码 2 结束。 |
| `-T, --throttle <seconds>` | `0`（每次失败立即发信） | `AE_ERROR_THROTTLE` | 缓冲逐次失败的邮件并分批发送，间隔不短于 N 秒；一批中有多个错误文件时打包为一个 zip 附件。 |
| `-v, --verbose` | 常规输出 | `AE_VERBOSE=2` | 打印每个 job 的完整命令行。 |
| `-q, --quiet` | 常规输出 | `AE_VERBOSE=0` | 只打印 PASS / FAIL / SKIP 单行。 |
| `-a, --attach <file>` | 无 | — | 给完成邮件附加文件（如结果数据库）；可重复。 |
| `-h, --help` | | | 打印用法并退出。 |

命令行选项优先于对应的环境变量。完成邮件的主题按 runner 的退出码打标签：

| 退出码 | 标签 | 含义 |
|---|---|---|
| 0 | `[SUCCESS]` | 所有 job 通过 |
| 1 | `[PARTIAL]` | 跑完全部，部分 job 失败 |
| 2 | `[FAILED]` | 被 `-x` 或 `-M` 提前中止 |

```bash
# 完整运行，结果数据库随邮件附上
bash scripts/run/run_notify.sh -a results/db/main.db scripts/run/train_main.sh

# 首次失败即停
bash scripts/run/run_notify.sh -x scripts/run/train_main.sh

# 详细输出，错误邮件节流，5 次失败后中止
bash scripts/run/run_notify.sh -v -T 300 -M 5 scripts/run/train_main.sh
```

有一个字母相撞：`run_notify.sh` 的 `-e` 是 `--exclude`，而每个训练/评估 CLI 的 `-e` 是 `--experiment`。二者不会出现在同一条命令行上，因为包装脚本的参数在 runner 路径之前、CLI 的参数在 runner 内部；但不要裸跑 runner 时用 `-e` 指望它是过滤器，`job` 行里的 `-e "$EXP"` 也从来不是排除。拿不准时，包装脚本写 `--exclude`，CLI 写 `--experiment`。

runner 也可以直接 `bash scripts/run/train_main.sh` 运行；此时不施加过滤、不发完成邮件，但 `run_helper.sh` 仍会对每次失败单独发邮件，因为它自己从 `.env` 读取 `NOTIFY_EMAIL`。

## 日志与错误文件

一个批次产出三类文件。job 日志由 `--log` 指定：训练与 MP 的 job 写入 `logs/<family>/<stage>/<model>/d<dim>[/<quarter>].log`（`logs/main/train/finetune/AssetBERT/d64/2023Q2.log`），并在 checkpoint 旁写一份副本；RV/RC 的 job 每个模型写一个日志，各维度共用，即 `logs/<family>/eval/<RV|RC>/<model>[_addChars].log`（`logs/main/eval/RV/AssetBERT.log`）。job 日志记录 DEBUG 级别的全部内容；控制台显示 INFO，除非 CLI 以 `-v` 运行。

错误文件是排错的入口。一次失败的训练或评估运行恰好写一个文件 `logs/error/<experiment>/<task>_<run_id[:8]>_<stamp>.log`，含配置、job 日志路径、命令与 traceback；`<experiment>` 目录就是 runner 传入的 `EXP`。没有 `Run` 的数据脚本写到顶层 `logs/error/<task>_<stamp>.log`。`job` 收集自己启动后新出现的每个错误文件，逐个附到失败邮件；使用 `-T` 时失败会被缓冲，每封批量邮件在错误文件有多个时打包成 zip。`run_notify.sh` 在结束时对整个批次做同样的事，多个时同样打包 zip。机制与文件格式见 [日志系统](../explanation/logging-system.md)。

第三类文件是记录数据库本身 `results/db/<family>.db`，每个 job 对应一行 `runs`，含状态（`completed`、`error`、`running`）、配置、来源信息与错误消息；见 [记录系统](../explanation/records-system.md)。

## 用 `runs_query` 查看状态与构造重跑列表

`scripts/tools/runs_query.py` 读取一个或多个记录数据库，输出过滤器可接受的文本。runner 给每个 job 打上其配置路径的标记，脚本把同一字符串存入 `runs.config_file`，所以该工具打印的列表恰好选中数据库所报告的那些运行。

```bash
# 按 (experiment, task) 计数：completed / error / running 与最近一次开始时间
uv run python -m scripts.tools.runs_query -d results/db/main.db status

# 续跑一个网格：configs/main 下每个没有完成运行的 *.json
uv run python -m scripts.tools.runs_query -d results/db/main.db missing --root configs/main --compress > /tmp/todo.txt
bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_main.sh

# 只重跑失败项：最近一次运行状态为 error 的配置路径
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status error > /tmp/failed.txt
bash scripts/run/run_notify.sh --filter-file /tmp/failed.txt scripts/run/eval_main.sh

# 跳过已完成的
uv run python -m scripts.tools.runs_query -d results/db/main.db configs --status completed > /tmp/done.txt
bash scripts/run/run_notify.sh --exclude-file /tmp/done.txt scripts/run/eval_main.sh
```

`-d/--db` 可重复，`--experiment PREFIX` 与 `--task TASK` 收窄查询，`--compress` 在某目录下磁盘上的每个配置都在集合中时把它折叠为 `dir/*`，让文件保持简短。输出的第一行是带查询来源的 `#` 注释，过滤引擎会忽略它。该工具只是提议一份列表；真正运行哪些 job，由你围绕它写的命令决定。完整选项表见 [CLI 参考](../reference/cli.md#runs-query)。
