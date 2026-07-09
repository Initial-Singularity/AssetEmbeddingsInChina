# Logging system

This document describes the AssetEmbeddings logging system, including color support, the various output formats, and `Logger_Preparer`.

## Contents

- [Overview](#overview)
- [LoggerConfig configuration](#loggerconfig-configuration)
- [Logger_Preparer](#logger_preparer)
- [Color system](#color-system)
- [Handler types](#handler-types)
- [Coloring strategies](#coloring-strategies)
- [Error-handling decorator](#error-handling-decorator)
- [Examples](#examples)

---

## Overview

The logging system is made up of the following components:

| File | Component | Description |
|------|------|------|
| `asset_embeddings/logger.py` | ColorParser | Color-string parsing |
| `asset_embeddings/logger.py` | ColoredConsoleFormatter | Colored console formatting |
| `asset_embeddings/logger.py` | ColoredHTMLFormatter | HTML / Notebook formatting |
| `asset_embeddings/logger.py` | Handler classes | The various log-output mechanisms |
| `asset_embeddings/preparers.py` | Logger_Preparer | Config-driven logger creation |
| `asset_embeddings/utils/log_utils.py` | log_exceptions_inclass | Exception-handling decorator |

### Features

- **Colored output** — supports both ANSI terminal and HTML / Notebook color schemes.
- **Multiple coloring strategies** — color the message, the level name, or the full format.
- **tqdm compatibility** — log output that doesn't break progress bars.
- **File rotation** — size-based rotating file logs.
- **Config-driven creation** — declarative configuration through `LoggerConfig`.

---

## LoggerConfig configuration

`LoggerConfig` defines all of a logger's configuration options:

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

### Configuration fields

| Field | Type | Default | Description |
|------|------|--------|------|
| `log_name` | str | `"app"` | Logger name |
| `log_file` | str/list/None | `"app.log"` | Log file path (may be a list; `None` disables file logging) |
| `log_format` | str | see below | Log format string |
| `date_format` | str | `"%Y-%m-%d %H:%M:%S"` | Date format |
| `console_stream` | str | `"tqdm"` | Console output type |
| `console_level` | str | `"INFO"` | Console log level |
| `file_level` | str | `"DEBUG"` | File log level |
| `enable_colors` | bool | `True` | Whether to enable colors |
| `color_target` | str | `"format"` | Coloring strategy |
| `max_bytes` | int | `10_000_000` | Maximum log file size in bytes (10 MB) |
| `backup_count` | int | `5` | Number of backup files |

### Default log format

```python
"%(asctime)s [%(name)s][%(levelname)s]: %(message)s"
```

---

## Logger_Preparer

`Logger_Preparer` is the recommended way to create a logger, following the project's Preparer pattern.

### Basic usage

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
logger.debug("Detailed debug info")  # written to file only
```

### Chained configuration

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

### Multi-file logging

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

## Color system

### ColorParser

`ColorParser` converts color strings into ANSI codes or CSS styles.

#### Supported ANSI colors

| Color | Code | Background |
|------|------|--------|
| black | `Fore.BLACK` | `bg_black` |
| red | `Fore.RED` | `bg_red` |
| green | `Fore.GREEN` | `bg_green` |
| yellow | `Fore.YELLOW` | `bg_yellow` |
| blue | `Fore.BLUE` | `bg_blue` |
| magenta | `Fore.MAGENTA` | `bg_magenta` |
| cyan | `Fore.CYAN` | `bg_cyan` |
| white | `Fore.WHITE` | `bg_white` |

#### Supported styles

| Style | Effect |
|------|------|
| bold | Bold / bright |
| bright | Bright (same as bold) |
| dim | Dim |
| normal | Normal |

#### Examples

```python
from asset_embeddings.logger import ColorParser

# Parse into ANSI codes
ansi = ColorParser.parse_ansi("red bold")  # red bold

# Parse into a CSS style
css = ColorParser.parse_html("navy bold")  # {'color': 'navy', 'font-weight': 'bold'}

# Wrap text
colored_text = ColorParser.colorize_ansi("ERROR", "red bold")
html_text = ColorParser.colorize_html("ERROR", "red bold")
```

### Default level colors

#### Console (ANSI)

| Level | Color |
|------|------|
| DEBUG | blue |
| INFO | cyan |
| WARNING | yellow |
| ERROR | red |
| CRITICAL | red bold |

#### HTML / Notebook

| Level | Color |
|------|------|
| DEBUG | purple |
| INFO | navy |
| WARNING | orange |
| ERROR | orangered |
| CRITICAL | darkred bold |

### Component colors (format mode)

| Component | Console color | HTML color |
|------|------------|-----------|
| asctime | dim | gray |
| name | green | darkgreen |
| levelname | by level | by level |
| message | none | none |
| pathname | magenta | purple |
| filename | magenta | purple |
| funcName | cyan | teal |
| lineno | dim | gray |

---

## Handler types

### TqdmLoggingHandler

Outputs via `tqdm.write()`, so it doesn't break progress bars:

```python
# Selected automatically: console_stream="tqdm"
config = LoggerConfig(console_stream="tqdm")
```

### SysLoggingHandler

Standard `sys.stdout` output:

```python
config = LoggerConfig(console_stream="std")
```

### NotebookLoggingHandler

Outputs via `IPython.display.HTML`, for use in Jupyter Notebook:

```python
config = LoggerConfig(console_stream="html")
```

### File handler

Uses `RotatingFileHandler`, with size-based rotation:

```python
config = LoggerConfig(
    log_file="logs/app.log",
    max_bytes=10*1024*1024,  # rotate after 10MB
    backup_count=5           # keep 5 backups
)
```

---

## Coloring strategies

Choose a coloring strategy through `color_target`:

### "message" — color the message only

```
2024-01-15 10:30:00 [INFO] MyApp: <cyan>This is the message</cyan>
```

### "levelname" — color the level name only

```
2024-01-15 10:30:00 [<cyan>INFO</cyan>] MyApp: This is the message
```

### "format" — color the full format (recommended)

```
<dim>2024-01-15 10:30:00</dim> [<cyan>INFO</cyan>] <green>MyApp</green>: This is the message
```

---

## Error-handling decorator

`asset_embeddings/utils/log_utils.py` provides the `log_exceptions_inclass` decorator for logging exceptions automatically.

### Basic usage

```python
from asset_embeddings.utils.log_utils import log_exceptions_inclass

class MyProcessor:
    def __init__(self, logger):
        self.logger = logger

    @log_exceptions_inclass("logger")
    def process(self, data):
        # If an exception is raised, it automatically:
        # 1. logs to self.logger
        # 2. writes to the logs/error/ directory
        # 3. re-raises the exception
        return self._do_process(data)
```

### Features

- Automatically logs the exception stack trace to the specified logger.
- Creates a `{ClassName}_{timestamp}.log` error log in the `logs/error/` directory.
- The exception is re-raised, so it doesn't interfere with normal error handling.

### Error-log directory structure

```
logs/
└── error/
    ├── DataProcessor_20240115_103000_123456.log   # {ClassName}_%Y%m%d_%H%M%S_%f
    └── DataProcessor_20240115_114523_654321.log
```

---

## Examples

### Logging setup for a training script

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
        console_stream="tqdm",      # compatible with progress bars
        console_level="DEBUG" if args.verbose else "INFO",
        file_level="DEBUG",
        enable_colors=True,
        color_target="format"
    )
).prepare()

logger.info("Starting training...")
logger.debug("Config: %s", config.to_dict())
```

### Logging setup for a Notebook

```python
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer

logger = Logger_Preparer().set_config(
    LoggerConfig(
        log_name="Notebook",
        console_stream="html",      # HTML output
        console_level="INFO",
        enable_colors=True,
        color_target="format"
    )
).prepare()

logger.info("This will appear with HTML formatting")
```

### Data-processing script

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
        # ... loading logic
        self.logger.debug(f"Loaded {len(df)} records")
        return df

    @log_exceptions_inclass("logger")
    def process(self, df):
        self.logger.info("Processing data...")
        # ... processing logic
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

## FAQ

### How do I disable colored output?

```python
config = LoggerConfig(enable_colors=False)
```

### How are ANSI codes handled in log files?

File logs use the standard `Formatter` and don't include color codes. Colors apply only to console / HTML output.

### How do I write to multiple files at different levels at once?

Currently, when `LoggerConfig.log_file` is a list, all files use the same `file_level`. If you need different levels, add the handler manually:

```python
logger = Logger_Preparer().set_config(config).prepare()

# Add a handler that logs ERROR only
error_handler = logging.FileHandler("logs/errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(config.log_format))
logger.addHandler(error_handler)
```

### What if tqdm and logging conflict?

Use `console_stream="tqdm"`; logs are output through `tqdm.write()` and won't break progress bars.
