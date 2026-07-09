# 日志系统

本文档介绍 AssetEmbeddings 的日志系统，包括颜色支持、各种输出格式以及 `Logger_Preparer`。

## 目录

- [概述](#概述)
- [LoggerConfig 配置](#loggerconfig-配置)
- [Logger_Preparer](#logger_preparer)
- [颜色系统](#颜色系统)
- [Handler 类型](#handler-类型)
- [着色策略](#着色策略)
- [错误处理装饰器](#错误处理装饰器)
- [示例](#示例)

---

## 概述

日志系统由以下组件构成：

| 文件 | 组件 | 说明 |
|------|------|------|
| `asset_embeddings/logger.py` | ColorParser | 颜色字符串解析 |
| `asset_embeddings/logger.py` | ColoredConsoleFormatter | 彩色控制台格式化 |
| `asset_embeddings/logger.py` | ColoredHTMLFormatter | HTML / Notebook 格式化 |
| `asset_embeddings/logger.py` | Handler 类 | 各种日志输出机制 |
| `asset_embeddings/preparers.py` | Logger_Preparer | 配置驱动的 logger 创建 |
| `asset_embeddings/utils/log_utils.py` | log_exceptions_inclass | 异常处理装饰器 |

### 特性

- **彩色输出**——同时支持 ANSI 终端与 HTML / Notebook 配色方案。
- **多种着色策略**——为消息、级别名或完整格式着色。
- **tqdm 兼容**——日志输出不打断进度条。
- **文件轮转**——基于大小的轮转文件日志。
- **配置驱动创建**——通过 `LoggerConfig` 声明式配置。

---

## LoggerConfig 配置

`LoggerConfig` 定义 logger 的全部配置项：

```python
from asset_embeddings.configs import LoggerConfig

config = LoggerConfig(
    log_name="MyApp",
    log_file="logs/app.log",
    log_format="%(asctime)s [%(levelname)s] %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    console_stream="tqdm",       # tqdm/std/html
    console_level="INFO",
    file_level="DEBUG",
    enable_colors=True,
    color_target="format",       # message/levelname/format
    max_bytes=10*1024*1024,      # 10MB
    backup_count=5
)
```

### 配置字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_name` | str | `"app"` | logger 名称 |
| `log_file` | str/list/None | `"app.log"` | 日志文件路径（可为列表；`None` 禁用文件日志） |
| `log_format` | str | 见下 | 日志格式串 |
| `date_format` | str | `"%Y-%m-%d %H:%M:%S"` | 日期格式 |
| `console_stream` | str | `"tqdm"` | 控制台输出类型 |
| `console_level` | str | `"INFO"` | 控制台日志级别 |
| `file_level` | str | `"DEBUG"` | 文件日志级别 |
| `enable_colors` | bool | `True` | 是否启用颜色 |
| `color_target` | str | `"format"` | 着色策略 |
| `max_bytes` | int | `10_000_000` | 日志文件最大字节数（10 MB） |
| `backup_count` | int | `5` | 备份文件数量 |

### 默认日志格式

```python
"%(asctime)s [%(name)s][%(levelname)s]: %(message)s"
```

---

## Logger_Preparer

`Logger_Preparer` 是创建 logger 的推荐方式，遵循项目的 Preparer 模式。

### 基本用法

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer

logger = Logger_Preparer().set_config(
    LoggerConfig(
        log_name="Training",
        log_file="logs/train.log",
        console_level="INFO",
        file_level="DEBUG"
    )
).prepare()

logger.info("Training started")
logger.debug("Detailed debug info")  # 仅写入文件
```

### 链式配置

```python
logger = (
    Logger_Preparer()
    .set_config(LoggerConfig(
        log_name="MyApp",
        console_level="DEBUG" if args.verbose else "INFO"
    ))
    .prepare()
)
```

### 多文件日志

```python
config = LoggerConfig(
    log_name="MultiLog",
    log_file=[
        "logs/all.log",
        "logs/errors.log"
    ],
    file_level="DEBUG"
)
logger = Logger_Preparer().set_config(config).prepare()
```

---

## 颜色系统

### ColorParser

`ColorParser` 把颜色字符串转换为 ANSI 码或 CSS 样式。

#### 支持的 ANSI 颜色

| 颜色 | 码 | 背景 |
|------|------|--------|
| black | `Fore.BLACK` | `bg_black` |
| red | `Fore.RED` | `bg_red` |
| green | `Fore.GREEN` | `bg_green` |
| yellow | `Fore.YELLOW` | `bg_yellow` |
| blue | `Fore.BLUE` | `bg_blue` |
| magenta | `Fore.MAGENTA` | `bg_magenta` |
| cyan | `Fore.CYAN` | `bg_cyan` |
| white | `Fore.WHITE` | `bg_white` |

#### 支持的样式

| 样式 | 效果 |
|------|------|
| bold | 加粗 / 高亮 |
| bright | 高亮（同 bold） |
| dim | 变暗 |
| normal | 正常 |

#### 示例

```python
from asset_embeddings.logger import ColorParser

# 解析为 ANSI 码
ansi = ColorParser.parse_ansi("red bold")  # red bold

# 解析为 CSS 样式
css = ColorParser.parse_html("navy bold")  # {'color': 'navy', 'font-weight': 'bold'}

# 包裹文本
colored_text = ColorParser.colorize_ansi("ERROR", "red bold")
html_text = ColorParser.colorize_html("ERROR", "red bold")
```

### 默认级别颜色

#### 控制台（ANSI）

| 级别 | 颜色 |
|------|------|
| DEBUG | blue |
| INFO | cyan |
| WARNING | yellow |
| ERROR | red |
| CRITICAL | red bold |

#### HTML / Notebook

| 级别 | 颜色 |
|------|------|
| DEBUG | purple |
| INFO | navy |
| WARNING | orange |
| ERROR | orangered |
| CRITICAL | darkred bold |

### 组件颜色（format 模式）

| 组件 | 控制台颜色 | HTML 颜色 |
|------|------------|-----------|
| asctime | dim | gray |
| name | green | darkgreen |
| levelname | 按级别 | 按级别 |
| message | 无 | 无 |
| pathname | magenta | purple |
| filename | magenta | purple |
| funcName | cyan | teal |
| lineno | dim | gray |

---

## Handler 类型

### TqdmLoggingHandler

通过 `tqdm.write()` 输出，因此不打断进度条：

```python
# console_stream="tqdm" 时自动选用
config = LoggerConfig(console_stream="tqdm")
```

### SysLoggingHandler

标准 `sys.stdout` 输出：

```python
config = LoggerConfig(console_stream="std")
```

### NotebookLoggingHandler

通过 `IPython.display.HTML` 输出，用于 Jupyter Notebook：

```python
config = LoggerConfig(console_stream="html")
```

### 文件 handler

使用 `RotatingFileHandler`，按大小轮转：

```python
config = LoggerConfig(
    log_file="logs/app.log",
    max_bytes=10*1024*1024,  # 超过 10MB 后轮转
    backup_count=5           # 保留 5 个备份
)
```

---

## 着色策略

通过 `color_target` 选择着色策略：

### "message" —— 仅为消息着色

```
2024-01-15 10:30:00 [INFO] MyApp: <cyan>This is the message</cyan>
```

### "levelname" —— 仅为级别名着色

```
2024-01-15 10:30:00 [<cyan>INFO</cyan>] MyApp: This is the message
```

### "format" —— 为完整格式着色（推荐）

```
<dim>2024-01-15 10:30:00</dim> [<cyan>INFO</cyan>] <green>MyApp</green>: This is the message
```

---

## 错误处理装饰器

`asset_embeddings/utils/log_utils.py` 提供 `log_exceptions_inclass` 装饰器，用于自动记录异常。

### 基本用法

```python
from asset_embeddings.utils.log_utils import log_exceptions_inclass

class MyProcessor:
    def __init__(self, logger):
        self.logger = logger

    @log_exceptions_inclass("logger")
    def process(self, data):
        # 一旦抛出异常，会自动：
        # 1. 记录到 self.logger
        # 2. 写入 logs/error/ 目录
        # 3. 重新抛出异常
        return self._do_process(data)
```

### 特性

- 自动把异常堆栈记录到指定的 logger。
- 在 `logs/error/` 目录创建一份 `{ClassName}_{timestamp}.log` 错误日志。
- 异常会被重新抛出，因此不干扰正常的错误处理。

### 错误日志目录结构

```
logs/
└── error/
    ├── DataProcessor_20240115_103000_123456.log   # {ClassName}_%Y%m%d_%H%M%S_%f
    └── DataProcessor_20240115_114523_654321.log
```

---

## 示例

### 训练脚本的日志设置

```python
import argparse
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--log", "-l", type=str, default=None)
args = parser.parse_args()

logger = Logger_Preparer().set_config(
    LoggerConfig(
        log_name="BERT Training",
        log_file=args.log,
        console_stream="tqdm",      # 兼容进度条
        console_level="DEBUG" if args.verbose else "INFO",
        file_level="DEBUG",
        enable_colors=True,
        color_target="format"
    )
).prepare()

logger.info("Starting training...")
logger.debug("Config: %s", config.to_dict())
```

### Notebook 的日志设置

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer

logger = Logger_Preparer().set_config(
    LoggerConfig(
        log_name="Notebook",
        console_stream="html",      # HTML 输出
        console_level="INFO",
        enable_colors=True,
        color_target="format"
    )
).prepare()

logger.info("This will appear with HTML formatting")
```

### 数据处理脚本

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.utils.log_utils import log_exceptions_inclass

class DataProcessor:
    def __init__(self, logger):
        self.logger = logger

    @log_exceptions_inclass("logger")
    def load_data(self, path):
        self.logger.info(f"Loading data from {path}")
        # ... 加载逻辑
        self.logger.debug(f"Loaded {len(df)} records")
        return df

    @log_exceptions_inclass("logger")
    def process(self, df):
        self.logger.info("Processing data...")
        # ... 处理逻辑
        return result

if __name__ == "__main__":
    logger = Logger_Preparer().set_config(
        LoggerConfig(
            log_name="DataProcessor",
            log_file="logs/data_process.log",
            console_level="INFO",
            file_level="DEBUG"
        )
    ).prepare()

    processor = DataProcessor(logger)
    df = processor.load_data("data/raw/input.csv")
    result = processor.process(df)
```

---

## 常见问题

### 如何关闭彩色输出？

```python
config = LoggerConfig(enable_colors=False)
```

### 日志文件中的 ANSI 码如何处理？

文件日志使用标准 `Formatter`，不包含颜色码。颜色只作用于控制台 / HTML 输出。

### 如何同时以不同级别写入多个文件？

目前当 `LoggerConfig.log_file` 为列表时，所有文件使用相同的 `file_level`。若需要不同级别，请手动添加 handler：

```python
logger = Logger_Preparer().set_config(config).prepare()

# 添加一个只记录 ERROR 的 handler
error_handler = logging.FileHandler("logs/errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(config.log_format))
logger.addHandler(error_handler)
```

### tqdm 与日志冲突怎么办？

使用 `console_stream="tqdm"`；日志会通过 `tqdm.write()` 输出，不会打断进度条。
