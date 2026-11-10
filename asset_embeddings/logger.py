"""Process-wide logging setup and the error-file writers.

Holds `configure_logging` (the one place handlers are installed, on the root logger), the colored console
formatters and the three console channels (tqdm, stdout, notebook HTML), the short-name filter, and the
error-file helpers `error_dir`, `job_log_paths`, `write_error_block` and `failure_report`. Every entry point
calls `configure_logging` once; every other module only does `logging.getLogger(__name__)`.
"""

import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import tqdm
from colorama import Fore, Style, Back, init
from IPython.display import display, HTML

from .configs import LoggerConfig


class ColorParser:
    """Turns color strings such as `"red bold"` or `"cyan dim"` into ANSI codes or CSS styles.

    ANSI accepts the eight colors (black, red, green, yellow, blue, magenta, cyan, white), their `bg_`
    variants and the styles bold, bright, dim and normal. HTML accepts any CSS color name or hex code plus
    the same styles.
    """

    ANSI_COLORS = {
        "black": Fore.BLACK,
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
    }

    ANSI_BG_COLORS = {
        "bg_black": Back.BLACK,
        "bg_red": Back.RED,
        "bg_green": Back.GREEN,
        "bg_yellow": Back.YELLOW,
        "bg_blue": Back.BLUE,
        "bg_magenta": Back.MAGENTA,
        "bg_cyan": Back.CYAN,
        "bg_white": Back.WHITE,
    }

    ANSI_STYLES = {
        "bold": Style.BRIGHT,
        "bright": Style.BRIGHT,
        "dim": Style.DIM,
        "normal": Style.NORMAL,
    }

    @staticmethod
    def parse_ansi(color_string: str) -> str:
        """Translate a color string into the ANSI escape codes that switch it on.

        Args:
            color_string: Space-separated color and style words, e.g. `"red bold"`; case-insensitive.

        Returns:
            The concatenated escape codes, or `""` for an empty or blank string.
        """
        if not color_string or not color_string.strip():
            return ""

        parts = color_string.lower().split()
        codes = []

        for part in parts:
            if part in ColorParser.ANSI_COLORS:
                codes.append(ColorParser.ANSI_COLORS[part])
            elif part in ColorParser.ANSI_BG_COLORS:
                codes.append(ColorParser.ANSI_BG_COLORS[part])
            elif part in ColorParser.ANSI_STYLES:
                codes.append(ColorParser.ANSI_STYLES[part])
            # An unknown word is dropped, not raised: a typo in a color scheme must never break logging.

        return "".join(codes)

    @staticmethod
    def parse_html(color_string: str) -> Dict[str, str]:
        """Translate a color string into CSS properties.

        Args:
            color_string: Space-separated words, e.g. `"navy dim"` or `"#FF5733 bold"`; any word that is not
                a style is taken as a CSS color.

        Returns:
            CSS property-value pairs (`color`, `font-weight`, `opacity`), empty for an empty or blank string.
        """
        if not color_string or not color_string.strip():
            return {}

        parts = color_string.split()
        style = {}

        for part in parts:
            part_lower = part.lower()
            if part_lower == "bold" or part_lower == "bright":
                style["font-weight"] = "bold"
            elif part_lower == "dim":
                style["opacity"] = "0.7"
            elif part_lower == "normal":
                pass  # the CSS default; not a color, so it must not fall through to the color branch
            else:
                style["color"] = part

        return style

    @staticmethod
    def colorize_ansi(text: str, color_string: str) -> str:
        """Wrap text in the ANSI codes of a color string and a reset.

        Args:
            text: The text to color.
            color_string: Color and style words, e.g. `"red bold"`; empty leaves the text untouched.

        Returns:
            The text between the escape codes and `Style.RESET_ALL`.
        """
        if not color_string or not color_string.strip():
            return text

        color_code = ColorParser.parse_ansi(color_string)
        return f"{color_code}{text}{Style.RESET_ALL}"

    @staticmethod
    def colorize_html(text: str, color_string: str) -> str:
        """Wrap text in a `<span>` carrying the inline CSS of a color string.

        Args:
            text: The text to color.
            color_string: Color and style words, e.g. `"red bold"`; empty leaves the text untouched.

        Returns:
            The span, or the bare text when the string yields no style.
        """
        if not color_string or not color_string.strip():
            return text

        style_dict = ColorParser.parse_html(color_string)
        if not style_dict:
            return text

        style_str = ";".join([f"{k}:{v}" for k, v in style_dict.items()])
        return f'<span style="{style_str};">{text}</span>'


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter that colors the message, the level name or every format component with ANSI codes.

    Which of the three it colors is `color_target`; with colors disabled it is a plain `logging.Formatter`.

    Args:
        fmt: Log format string, e.g. `"%(asctime)s [%(levelname)s]: %(message)s"`.
        datefmt: Timestamp format.
        style: Format style (`%`, `{` or `$`).
        color_target: `"message"`, `"levelname"` or `"format"`.
        enable_colors: Whether to color at all.
    """

    LEVEL_COLORS = {
        "DEBUG": "blue",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red bold",
    }

    COMPONENT_COLORS = {
        "asctime": "dim",
        "name": "green",
        "levelname": "",  # replaced per record by LEVEL_COLORS
        "message": "",  # the message stays plain in format mode
        "pathname": "magenta",
        "filename": "magenta",
        "funcName": "cyan",
        "lineno": "dim",
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        color_target: str = "format",
        enable_colors: bool = True,
    ):
        super().__init__(fmt, datefmt, style)
        init(autoreset=True)
        self.color_target = color_target
        self.enable_colors = enable_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, colored according to `color_target`.

        Args:
            record: The record to format.

        Returns:
            The formatted line, with ANSI codes unless colors are disabled or the target is unknown.
        """
        if not self.enable_colors:
            return super().format(record)

        if self.color_target == "message":
            return self._format_message_colored(record)
        elif self.color_target == "levelname":
            return self._format_levelname_colored(record)
        elif self.color_target == "format":
            return self._format_full_colored(record)
        else:
            return super().format(record)

    def _format_message_colored(self, record):
        """Color only the message text."""
        base_message = super().format(record)
        plain_message = record.getMessage()

        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        if not level_color:
            return base_message

        colored_message = ColorParser.colorize_ansi(plain_message, level_color)
        return base_message.replace(plain_message, colored_message)

    def _format_levelname_colored(self, record):
        """Color only the levelname."""
        original_levelname = record.levelname

        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        if level_color:
            record.levelname = ColorParser.colorize_ansi(original_levelname, level_color)

        result = super().format(record)
        record.levelname = original_levelname
        return result

    def _format_full_colored(self, record):
        """Color each component of the format string separately."""
        original_values = {}

        component_colors = dict(self.COMPONENT_COLORS)
        component_colors["levelname"] = self.LEVEL_COLORS.get(record.levelname, "")

        # Repeats the steps of `Formatter.format` by hand: calling it would re-render asctime and undo the
        # coloring below.
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        for component, color in component_colors.items():
            if not color:
                continue

            if hasattr(record, component):
                original_value = getattr(record, component)
                original_values[component] = original_value
                value_str = str(original_value)
                colored_value = ColorParser.colorize_ansi(value_str, color)
                setattr(record, component, colored_value)

        result = self.formatMessage(record)

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + record.exc_text
        if record.stack_info:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + self.formatStack(record.stack_info)

        # The same record goes on to the file handler, which must see the plain values.
        for component, original_value in original_values.items():
            setattr(record, component, original_value)

        return result


class ColoredHTMLFormatter(logging.Formatter):
    """Notebook formatter that colors the message, the level name or every format component with CSS spans.

    The HTML twin of `ColoredConsoleFormatter`, for the `html` console channel; with colors disabled it is
    a plain `logging.Formatter`.

    Args:
        fmt: Log format string, e.g. `"%(asctime)s [%(levelname)s]: %(message)s"`.
        datefmt: Timestamp format.
        style: Format style (`%`, `{` or `$`).
        color_target: `"message"`, `"levelname"` or `"format"`.
        enable_colors: Whether to color at all.
    """

    LEVEL_COLORS = {
        "DEBUG": "purple",
        "INFO": "navy",
        "WARNING": "orange",
        "ERROR": "orangered",
        "CRITICAL": "darkred bold",
    }

    COMPONENT_COLORS = {
        "asctime": "gray",
        "name": "darkgreen",
        "levelname": "",  # replaced per record by LEVEL_COLORS
        "message": "",  # the message stays plain in format mode
        "pathname": "purple",
        "filename": "purple",
        "funcName": "teal",
        "lineno": "gray",
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        color_target: str = "format",
        enable_colors: bool = True,
    ):
        super().__init__(fmt, datefmt, style)
        self.color_target = color_target
        self.enable_colors = enable_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format the record as HTML, colored according to `color_target`.

        Args:
            record: The record to format.

        Returns:
            The formatted line with `<span>` styles, or plain text when colors are disabled or the target
            is unknown.
        """
        if not self.enable_colors:
            return super().format(record)

        if self.color_target == "message":
            return self._format_message_colored(record)
        elif self.color_target == "levelname":
            return self._format_levelname_colored(record)
        elif self.color_target == "format":
            return self._format_full_colored(record)
        else:
            return super().format(record)

    def _format_message_colored(self, record):
        """Color only the message text with HTML."""
        base_message = super().format(record)
        plain_message = record.getMessage()

        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        if not level_color:
            return base_message

        colored_message = ColorParser.colorize_html(plain_message, level_color)
        return base_message.replace(plain_message, colored_message)

    def _format_levelname_colored(self, record):
        """Color only the levelname with HTML."""
        original_levelname = record.levelname

        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        if level_color:
            record.levelname = ColorParser.colorize_html(original_levelname, level_color)

        result = super().format(record)
        record.levelname = original_levelname
        return result

    def _format_full_colored(self, record):
        """Color each component of the format string separately with HTML."""
        original_values = {}

        component_colors = dict(self.COMPONENT_COLORS)
        component_colors["levelname"] = self.LEVEL_COLORS.get(record.levelname, "")

        # Repeats the steps of `Formatter.format` by hand: calling it would re-render asctime and undo the
        # coloring below.
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        for component, color in component_colors.items():
            if not color:
                continue

            if hasattr(record, component):
                original_value = getattr(record, component)
                original_values[component] = original_value
                value_str = str(original_value)
                colored_value = ColorParser.colorize_html(value_str, color)
                setattr(record, component, colored_value)

        result = self.formatMessage(record)

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + record.exc_text
        if record.stack_info:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + self.formatStack(record.stack_info)

        # The same record goes on to the file handler, which must see the plain values.
        for component, original_value in original_values.items():
            setattr(record, component, original_value)

        return result


class SysLoggingHandler(logging.StreamHandler):
    """The `std` console channel: a stream handler that flushes after every record and skips empty lines."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write the record to the stream and flush.

        Args:
            record: The record to emit; one whose formatted text is empty is dropped.
        """
        try:
            msg = self.format(record)
            if msg:
                super().emit(record)
                self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class TqdmLoggingHandler(logging.StreamHandler):
    """The default console channel: writes through `tqdm.write`, so a log line never breaks a progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        """Print the record above any active progress bar.

        Args:
            record: The record to emit; one whose formatted text is empty is dropped.
        """
        try:
            msg = self.format(record)
            if msg:
                tqdm.tqdm.write(msg, end=self.terminator)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class NotebookLoggingHandler(logging.StreamHandler):
    """The `html` console channel: renders each record as HTML in the notebook cell output."""

    def emit(self, record: logging.LogRecord) -> None:
        """Display the record through `IPython.display.HTML`.

        Args:
            record: The record to emit; one whose formatted text is empty is dropped.
        """
        try:
            msg = self.format(record)
            if msg:
                display(HTML(msg))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


# Relocates the error files; the test suite points it at a temporary directory so no test writes into logs/error.
ERROR_DIR_ENV = "AE_ERROR_DIR"


def error_dir() -> Path:
    """The directory error files are written to.

    Returns:
        `$AE_ERROR_DIR` when set, otherwise `logs/error` relative to the working directory.
    """
    return Path(os.environ.get(ERROR_DIR_ENV) or os.path.join("logs", "error"))


def job_log_paths() -> List[str]:
    """The job log files of this process.

    Returns:
        The `baseFilename` of every `FileHandler` that
        [`configure_logging`][asset_embeddings.logger.configure_logging] installed on the root logger.
    """
    devnull = os.path.abspath(os.devnull)  # pytest's logging plugin keeps a FileHandler on os.devnull
    return [
        h.baseFilename
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename != devnull
    ]


def write_error_block(path: Path, fields: Dict[str, str], exc_info) -> None:
    """Append one block to an error file: `key : value` lines, a rule, the formatted traceback.

    Parent directories are created; an existing file is appended to, so a run's write failures and its
    terminal exception land in one file in order.

    Args:
        path: The error file.
        fields: Header lines, in order; keys are padded to the longest.
        exc_info: The `(type, value, traceback)` triple as returned by `sys.exc_info()`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max((len(k) for k in fields), default=0)
    with open(path, "a", encoding="utf-8") as f:
        for k, v in fields.items():
            f.write(f"{k:<{width}} : {v}\n")
        f.write("-" * 60 + "\n")
        f.write("".join(traceback.format_exception(*exc_info)))
        f.write("\n")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@contextmanager
def failure_report(task: str, *, config_file: Optional[str] = None) -> Iterator[None]:
    """Write an error file for any exception that leaves the block, log it once, and re-raise.

    For scripts that have no `Run`. Scripts that record through `asset_embeddings.records.Run` get the same
    file from `Run.__exit__` (under an `<experiment>/` subdirectory, with the run identity in the name) and
    do not use this.

    Args:
        task: Label of the script; the file is `error_dir()/<task>_<stamp>.log`.
        config_file: The config path to name in the file's header, if the script has one.

    Yields:
        Nothing; the body runs inside the block.

    Raises:
        BaseException: Whatever the block raised, unchanged; a failure to write the error file is only logged.
    """
    try:
        yield
    except BaseException:
        exc_info = sys.exc_info()
        path = error_dir() / f"{task}_{_stamp()}.log"
        fields = {
            "task": task,
            "config": config_file or "-",
            "job log": ", ".join(job_log_paths()) or "-",
            "command": " ".join(sys.argv),
        }
        try:
            write_error_block(path, fields, exc_info)
            where = f"details in {path}"
        except OSError as write_exc:  # the error channel is best effort: never replace the script's own exception
            where = f"error file could not be written ({write_exc})"
        logging.getLogger(__name__).error(f"{task} failed; {where}", exc_info=exc_info)
        raise


PACKAGE_LOGGER = "asset_embeddings"
# Attribute set on every handler this module installs, so a second `configure_logging` removes exactly those
# and leaves handlers owned by pytest, a notebook or the caller alone.
_INSTALLED = "_asset_embeddings_handler"


def is_local_main_process() -> bool:
    """Whether this process owns the console, the job log and the `runs` row.

    Returns:
        `False` only for a non-main rank of a distributed launch (`LOCAL_RANK` set and non-zero).
    """
    return os.environ.get("LOCAL_RANK", "0") in ("", "0")


def module_name(module_globals: Dict[str, Any]) -> str:
    """Dotted module name of a script, also when it runs as the entry point.

    `python -m pkg.mod` sets `__name__ == "__main__"` but keeps the real name in `__spec__`; a file run by
    path (no spec) is named from its location inside the `asset_embeddings` package, or by its stem outside
    it. This is `runs.script` and the display name of the script's logger.

    Args:
        module_globals: The module's globals (`vars(module)` or `globals()`).

    Returns:
        The dotted name, e.g. `asset_embeddings.scripts.train.bert`.
    """
    spec = module_globals.get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return spec.name
    name = module_globals.get("__name__", "")
    if name and name != "__main__":
        return name
    file = module_globals.get("__file__", "")
    if not file:
        return name
    parts = Path(file).resolve().with_suffix("").parts
    if PACKAGE_LOGGER in parts:
        start = len(parts) - 1 - parts[::-1].index(PACKAGE_LOGGER)
        return ".".join(parts[start:])
    return Path(file).stem


def _short_name(name: str) -> str:
    if name == PACKAGE_LOGGER:
        return PACKAGE_LOGGER
    if name.startswith(PACKAGE_LOGGER + "."):
        short = name[len(PACKAGE_LOGGER) + 1 :]
        return short[len("scripts.") :] if short.startswith("scripts.") else short
    if name == "py.warnings":
        return "warnings"
    return name.split(".", 1)[0]


class _ShortName(logging.Filter):
    """Filter that sets `record.short`, the shortened logger name the format string prints.

    `asset_embeddings.scripts.train.bert` → `train.bert`, `asset_embeddings.preparers` → `preparers`,
    `transformers.modeling_utils` → `transformers`, `py.warnings` → `warnings`. The entry-point module logs
    as `__main__`; its records show the script's real name (`train.w2v`), resolved once at construction.
    """

    def __init__(self) -> None:
        super().__init__()
        main = sys.modules.get("__main__")
        self._main = _short_name(module_name(vars(main))) if main is not None else "__main__"

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach `record.short` and pass every record through.

        Args:
            record: The record being handled.

        Returns:
            Always `True`; the filter only annotates.
        """
        record.short = self._main if record.name == "__main__" else _short_name(record.name)
        return True


def _console_handler(cfg: LoggerConfig) -> logging.Handler:
    if cfg.console_stream == "tqdm":
        handler: logging.Handler = TqdmLoggingHandler()
        formatter: logging.Formatter = ColoredConsoleFormatter(
            fmt=cfg.log_format, datefmt=cfg.date_format, color_target=cfg.color_target, enable_colors=cfg.enable_colors
        )
    elif cfg.console_stream == "std":
        handler = SysLoggingHandler(sys.stdout)
        formatter = ColoredConsoleFormatter(
            fmt=cfg.log_format, datefmt=cfg.date_format, color_target=cfg.color_target, enable_colors=cfg.enable_colors
        )
    elif cfg.console_stream == "html":
        handler = NotebookLoggingHandler()
        formatter = ColoredHTMLFormatter(
            fmt=cfg.log_format, datefmt=cfg.date_format, color_target=cfg.color_target, enable_colors=cfg.enable_colors
        )
    else:
        raise ValueError(f"Unrecognized console stream type: {cfg.console_stream}. Supported: tqdm, std, html.")
    handler.setFormatter(formatter)
    return handler


def _route_transformers() -> None:
    """HF transformers keeps its own stderr handler and does not propagate; send its records to the root instead."""
    if "transformers" not in sys.modules:
        return
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except (ImportError, AttributeError):
        pass


def configure_logging(
    config: Optional[LoggerConfig] = None,
    *,
    log_files: Union[str, Sequence[str], None] = None,
    console_level: Optional[str] = None,
) -> None:
    """Install the process-wide logging setup: console and file handlers on the root logger, the package at DEBUG.

    Called once, as the first statement of an entry point's `main()`. Effective levels: `asset_embeddings.*`
    and `__main__` (the script being run, which logs under its real name) records exist from DEBUG (file) /
    `console_level` (screen); every other logger and captured Python warnings from WARNING. Each file gets
    one `=====` header line per process start and is appended to, never rotated; the command line is also
    logged once at DEBUG, so `-v` shows it on the console. On a non-main rank of a distributed launch the
    console shows WARNING and above and no file is written. Idempotent: a second call replaces the handlers
    this function installed and leaves any other handler alone.

    Args:
        config: The full parameter set; a default `LoggerConfig` when omitted.
        log_files: One path or several; overrides `config.log_file`. `None` keeps the config's value.
        console_level: Overrides `config.console_level` (`"DEBUG"` … `"CRITICAL"`).

    Raises:
        ValueError: If `config.console_stream` is not `tqdm`, `std` or `html`.
    """
    cfg = config if config is not None else LoggerConfig()
    if log_files is not None:
        cfg.log_file = list(log_files) if not isinstance(log_files, str) else log_files
    if console_level is not None:
        cfg.console_level = console_level
    main = is_local_main_process()

    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, _INSTALLED, False)]:
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.WARNING)
    logging.getLogger(PACKAGE_LOGGER).setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(
        logging.DEBUG
    )  # the script itself: `logging.getLogger(__name__)` at its entry point
    short = _ShortName()

    console = _console_handler(cfg)
    console.setLevel(cfg.console_level if main else "WARNING")
    console.addFilter(short)
    setattr(console, _INSTALLED, True)
    root.addHandler(console)

    files: List[str] = []
    if main and cfg.log_file:
        files = [cfg.log_file] if isinstance(cfg.log_file, str) else list(cfg.log_file)
    header = f"===== {datetime.now().isoformat(timespec='seconds')} {' '.join(sys.argv)} =====\n"
    for path in files:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setLevel(cfg.file_level)
        fh.setFormatter(logging.Formatter(cfg.log_format, cfg.date_format))
        fh.addFilter(short)
        setattr(fh, _INSTALLED, True)
        root.addHandler(fh)
        fh.stream.write(header)
        fh.stream.flush()

    logging.captureWarnings(True)
    _route_transformers()
    logging.getLogger(__name__).debug(f"Command: {' '.join(sys.argv)}")
