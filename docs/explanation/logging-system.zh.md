# 日志系统

本文档介绍 AssetEmbeddingsInChina 脚本的日志方式：一次进程级的配置调用、标准的模块 logger、每个配置运行一个 job 日志文件、每次失败运行一个错误文件。

## 目录

- [概述](#概述)
- [configure_logging](#configure_logging)
- [模块 logger](#模块-logger)
- [job 日志文件](#job-日志文件)
- [错误文件](#错误文件)
- [控制台输出](#控制台输出)
- [颜色系统](#颜色系统)
- [Notebook](#notebook)
- [分布式启动](#分布式启动)
- [FAQ](#faq)

---

## 概述

| 文件 | 组件 | 说明 |
|------|------|------|
| `asset_embeddings/logger.py` | `configure_logging` | 唯一安装 handler 的地方（根 logger） |
| `asset_embeddings/logger.py` | `error_dir`、`write_error_block`、`failure_report`、`job_log_paths` | 错误文件 |
| `asset_embeddings/logger.py` | `ColorParser`、`ColoredConsoleFormatter`、`ColoredHTMLFormatter` | 终端与 notebook 的着色 |
| `asset_embeddings/logger.py` | `TqdmLoggingHandler`、`SysLoggingHandler`、`NotebookLoggingHandler` | 三种控制台通道 |
| `asset_embeddings/configs/object_configs.py` | `LoggerConfig` | `configure_logging` 的参数对象 |
| `asset_embeddings/records/run.py` | `Run` | 写失败运行的错误文件（见 [记录系统](records-system.md)） |

### 原则

- **库只命名，入口只配置。** 每个模块有 `logger = logging.getLogger(__name__)`，从不碰 handler。`main()` 调用一次 `configure_logging(...)`，其余全靠标准库的 logger 层级。没有 logger 对象在类之间传递。
- **job 日志是进程的完整记录。** handler 挂在根 logger 上，因此包内的 DEBUG 输出、第三方库（transformers、accelerate、gensim、torch）的 WARNING、被捕获的 Python 警告都进同一个文件、同一块屏幕。
- **文件 DEBUG，屏幕 INFO。** 文件什么都留；控制台显示 INFO 以上（`-v` 时 DEBUG）。
- **每个配置运行一个文件，追加，不轮转。** 同一配置重跑在同一文件里追加新的一段；每段以一行 `=====` 头开始。
- **每次失败运行一个错误文件。** 由 `Run.__exit__` 写一次（没有 `Run` 的脚本由 `failure_report` 写），带运行身份、配置路径、job 日志路径、命令行与 traceback。
- **tqdm 安全、带颜色的控制台。** 日志行不会打断进度条；颜色只在控制台。

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

关键字参数覆盖 `LoggerConfig` 里对应的字段；脚本只传这两个。它按顺序安装：

1. **根 logger。** handler 挂在根上；根级别 `WARNING`；`asset_embeddings` logger 级别 `DEBUG`，`__main__` 也是——入口模块自己要记日志时用的就是这个名字（`python -m asset_embeddings.scripts.eval.rv` 与 `python asset_embeddings/scripts/eval/rv.py` 都把模块作为 `__main__` 执行；训练与评估的入口脚本只驱动包里的类）。有效级别因此是：`asset_embeddings.*` 与入口模块的记录从 DEBUG 起产生，其他所有 logger（以及 `py.warnings`）的记录从 WARNING 起产生。命令行以 DEBUG 记一次，`-v` 时能在控制台看到。
2. **控制台 handler**，按 `console_stream`：`tqdm`（默认，经 `tqdm.write` 写）、`std`（普通 stdout）或 `html`（notebook），级别为 `console_level`。
3. **文件 handler**，`log_files` 里每个路径一个 `logging.FileHandler(path, mode="a", encoding="utf-8")`，级别为 `file_level`（默认 DEBUG）。父目录自动创建。handler 挂好之后，直接向每个文件写一行头：
   `===== 2026-09-22T03:15:44 uv run python -m asset_embeddings.scripts.eval.rv -c … -r … -e main =====`
4. **短名。** 每个安装的 handler 上有一个过滤器设置 `record.short`：`asset_embeddings.` 下的名字去掉这个前缀及随后的 `scripts.`（`asset_embeddings.scripts.train.bert` 变为 `train.bert`，`asset_embeddings.preparers` 变为 `preparers`，`asset_embeddings.records.store` 变为 `records.store`）；其他名字保留第一节（`transformers.modeling_utils` 变为 `transformers`）；`__main__` 显示为脚本的真实短名（`train.w2v`），由模块 spec 或文件在包内的位置解析；`py.warnings` 变为 `warnings`。默认格式为 `%(asctime)s [%(short)s][%(levelname)s]: %(message)s`。
5. **Python 警告。** `logging.captureWarnings(True)`：`warnings.warn(...)` 变成 `py.warnings` logger 的一条 WARNING，每个位置一次（Python 默认过滤器）。包内没有模块全局静音警告；确需安静的地方用局部的 `warnings.catch_warnings()`。
6. **transformers。** HF 在自己的 logger 上挂了私有的 stderr handler 且不向上传播；若 `transformers` 已导入，则移除其默认 handler 并开启传播，让它的 WARNING 到达根 handler。
7. **幂等。** 本函数安装的 handler 带标记；再次调用只移除这些、重新安装（notebook 重跑设置单元或测试套件都不会累积 handler）。

### LoggerConfig

`LoggerConfig` 和项目里其他 `Config` 一样，但从不出现在 JSON 配置里：它是 `configure_logging` 的参数对象。

| 字段 | 类型 | 默认 | 说明 |
|------|------|--------|------|
| `log_file` | `str \| list[str] \| None` | `None` | job 日志文件；`None` 表示只有控制台。 |
| `log_format` | `str` | `%(asctime)s [%(short)s][%(levelname)s]: %(message)s` | 格式串；`%(short)s` 是缩短的 logger 名，`%(name)s` 仍可用。 |
| `date_format` | `str` | `%Y-%m-%d %H:%M:%S` | 时间戳格式。 |
| `console_level` | `DEBUG` … `CRITICAL` | `INFO` | 控制台最低级别。 |
| `console_stream` | `tqdm` / `std` / `html` | `tqdm` | 控制台通道。 |
| `file_level` | `DEBUG` … `CRITICAL` | `DEBUG` | 文件最低级别。 |
| `enable_colors` | `bool` | `True` | 控制台着色。 |
| `color_target` | `message` / `levelname` / `format` | `format` | 控制台格式器着色的对象。 |

---

## 模块 logger

每个会写日志的模块：

```python
import logging

logger = logging.getLogger(__name__)


class Dataset_Preparer(Preparer):          # asset_embeddings/preparers.py
    def prepare(self, tokenizer):
        self.logger.info("Preparing Dataset...")   # 类保留 `self.logger`，它就是模块 logger
        ...
```

训练与评估的入口脚本里没有类：训练器在 `asset_embeddings.trainers`，评估器在 `asset_embeddings.evaluators`，每个 `main()` 只构造一个并驱动它，因此它们记的每条日志都是包树上真实的节点（`trainers.bert`、`evaluators.base`、`preparers`、`records.run`）。树内由干活的组件报告它干的活：数据集记它加载的文件与记录数，preparer 记它构建的东西，训练器只记它自己的阶段（`Preparing Trainer...` / `Trainer Prepared.`）、它自己的决定（切分、最佳轮次）和存到了哪里，不去预告 preparer 要做什么。训练器的 `self.logger` 来自 `BaseTrainer`，绑定的是子类模块的 logger。真正作为入口运行的模块（数据脚本、notebook 单元）以 `__main__` 记日志；`configure_logging` 给这个 logger 与包相同的级别，短名过滤器把它打印成脚本的真实名字，所以那里也不会丢。`self.logger` 作为属性保留，因为几百处调用点在用它，也因为测试仍可以给某个实例装一个安静的 logger（`evaluator.logger = logging.getLogger("test.quiet")`）。没有任何注入：`Preparer` 没有 `set_logger`，`Run` 与 `RecordStore` 不接 `logger` 参数，训练器与评估器没有 `set_logger` 方法。包内任何地方、或进程导入的任何库记的一条消息，都到达同一组 handler。

一次运行在文件里的典型几行：

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

## job 日志文件

每个训练与评估脚本接受 `--log`（`-l`），一个或多个路径。`scripts/run/` 下的训练 runner 与 MP 评估 runner 传两个：一个在该运行的 checkpoint 目录下（`checkpoints/.../train.log`，MP 为 `refine.log`，跟着产物走），一个在 `logs/<实验族>/<阶段>/<模型>/d<dim>[/<季度>].log`，镜像 `configs/` 的布局；两份内容相同。RV 与 RC 评估 runner 每个模型只传一个路径，各维度共用：`logs/<实验族>/eval/<RV|RC>/<模型>[_addChars].log`。

文件以追加模式打开，从不轮转（一次运行的日志是 KB 到几 MB；轮转是常驻服务的概念）。同一配置重跑会追加新的一段；`grep '^=====' <file>` 列出文件里的各次运行，头行之后紧跟的 `[records.run]` 那行带着数据库行的 `run_id` 与 `run_no`，因此一段日志可以对上 `runs` 表和它的错误文件。

runner 不捕获进程的 stdout/stderr：C 库直接打印的内容（CUDA、bitsandbytes）只在终端可见。

---

## 错误文件

排错从 `logs/error/` 开始。写入函数在 `asset_embeddings/logger.py`：

```python
def error_dir() -> Path                       # $AE_ERROR_DIR 或 logs/error
def job_log_paths() -> List[str]              # configure_logging 安装的文件
def write_error_block(path, fields, exc_info) # `键 : 值` 行、一条分隔线、traceback；追加
@contextmanager
def failure_report(task, *, config_file=None) # 给没有 Run 的脚本用
```

**有 `Run` 的脚本**（十个训练/评估脚本）：任何异常离开 `with Run` 块时，`Run.__exit__` 先经 `records.run` 记一次带 traceback 的 ERROR，再写 `logs/error/<experiment>/<task>_<run_id前8位>_<时间戳>.log`。运行期间 records 的写入失败、以及退出时失败的 `finish_run`，先追加到同一文件（各带表名与记录内容）；终止运行的 traceback 总是最后一段，所以一次运行至多一个错误文件，自上而下就是出错的顺序。在 `run.set_config(...)` 之前的失败——配置路径写错、覆盖键打错——还没有 `runs` 行；文件照样写，标签用脚本名的 stem（`rv_3f9a1c2e_…log`），`run_no : -`。打不开的 `--result_file`（等待迁移的 v4 文件、不可写的路径）在 `Run.from_args` 里、进入 `with` 块之前就失败；`from_args` 以同样方式报告它，因此运行的任何失败都不会没有文件。

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

**没有 `Run` 的脚本**（一次性工具）把主体包起来：

```python
from asset_embeddings.logger import configure_logging, failure_report

configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
with failure_report(task="my_tool", config_file=getattr(args, "config", None)):
    ...
```

出现异常时写 `logs/error/my_tool_<时间戳>.log`（task、config、job 日志、命令行、traceback），记一次日志，重新抛出。

runner（`run_helper.sh`、`run_notify.sh`）用递归 `find` 收集比 job 开始标记新的所有错误文件，附在失败邮件里。把 `AE_ERROR_DIR` 指向别的目录（例如测试中的临时目录），错误文件就不会写进仓库的 `logs/error/`。

---

## 控制台输出

三种通道，由 `console_stream` 选择：

| 通道 | Handler | 用途 |
|---|---|---|
| `tqdm`（默认） | `TqdmLoggingHandler`——`tqdm.write` | 带进度条的脚本：日志行打印在进度条上方，不打断它 |
| `std` | `SysLoggingHandler`——`sys.stdout` | 普通终端、管道输出 |
| `html` | `NotebookLoggingHandler`——`IPython.display.HTML` | Notebook（演示） |

控制台格式器是 `ColoredConsoleFormatter`（ANSI）或 `ColoredHTMLFormatter`（CSS）；文件格式器是同一格式串的普通 `logging.Formatter`，文件里永远没有转义码。

---

## 颜色系统

`ColorParser` 把 `"red bold"`、`"cyan dim"` 这样的字符串转成 ANSI 码（终端）或 CSS（HTML）。

| 级别 | 控制台 | HTML |
|---|---|---|
| DEBUG | blue | purple |
| INFO | cyan | navy |
| WARNING | yellow | orange |
| ERROR | red | orangered |
| CRITICAL | red bold | darkred bold |

`color_target` 决定着色对象：`message`（只有消息文本）、`levelname`（只有级别）或 `format`（格式串的每个组件：变暗的时间戳、按级别着色的级别名；格式串使用 `%(name)s` 时 logger 名为绿色，默认格式打印的是短名 `%(short)s`，不着色）。支持的 ANSI 颜色：black、red、green、yellow、blue、magenta、cyan、white 及其 `bg_` 变体，样式 bold / bright / dim / normal；HTML 接受任何 CSS 颜色名或十六进制码。

---

## Notebook

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.logger import configure_logging

configure_logging(LoggerConfig(console_stream="html", log_file=None, console_level="INFO"))
```

此后 `asset_embeddings.*` 的每条 INFO 都以着色 HTML 渲染在单元输出里；重跑该单元替换 handler 而不是叠加。

---

## 分布式启动

`configure_logging` 读取 `LOCAL_RANK`。非主进程的控制台只显示 WARNING 以上，且不安装文件 handler，因此 job 日志只由主进程写一次；决定哪个 rank 拥有 `runs` 行、并在失败时写错误文件的是同一条规则（`is_local_main_process`）；非主进程只记 traceback，不写文件。

---

## FAQ

### 如何关闭颜色？

`configure_logging(LoggerConfig(enable_colors=False, ...))`。文件里从来没有颜色。

### 如何在屏幕上看 DEBUG？

任何脚本的 `-v` / `--verbose` 把 `console_level` 设为 `DEBUG`；文件无论如何都是 DEBUG。

### 某个库的警告刷屏

Python 默认每个位置只发一次警告，量是有界的。若某条警告已知且无害，在调用处局部静音（`with warnings.catch_warnings(): warnings.simplefilter("ignore", SomeWarning)`），不要在 import 时全局静音。

### 失败运行的 traceback 在哪？

按设计有三处：job 日志（一次，来自 `records.run`）、`logs/error/<experiment>/` 下的错误文件、数据库里的 `runs.error_message`。错误文件是入口；它的头部指向另外两处。
