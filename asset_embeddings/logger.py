import logging
from typing import Dict

import tqdm
from colorama import Fore, Style, Back, init
from IPython.display import display, HTML


class ColorParser:
    """Parser for color strings to ANSI/HTML styles.

    This utility class converts human-readable color strings like "red bold"
    or "cyan dim" into ANSI escape codes for terminal output or CSS styles
    for HTML output.

    Supported ANSI colors: black, red, green, yellow, blue, magenta, cyan, white
    Supported ANSI backgrounds: bg_black, bg_red, bg_green, bg_yellow, bg_blue, bg_magenta, bg_cyan, bg_white
    Supported ANSI styles: bold, dim, bright, normal

    For HTML, any valid CSS color name or hex code can be used.
    """

    # ANSI color mappings
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

    # ANSI background color mappings
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

    # ANSI style mappings
    ANSI_STYLES = {
        "bold": Style.BRIGHT,
        "bright": Style.BRIGHT,
        "dim": Style.DIM,
        "normal": Style.NORMAL,
    }

    @staticmethod
    def parse_ansi(color_string: str) -> str:
        """Parse a color string to ANSI escape codes.

        Args:
            color_string: Space-separated color specification like "red bold"
                         or "cyan dim". Empty string returns empty string (no color).

        Returns:
            ANSI escape code string that can be used to colorize terminal text.

        Examples:
            >>> ColorParser.parse_ansi("red bold")
            '\x1b[31m\x1b[1m'
            >>> ColorParser.parse_ansi("cyan")
            '\x1b[36m'
            >>> ColorParser.parse_ansi("")
            ''
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
            # Silently ignore unknown parts

        return "".join(codes)

    @staticmethod
    def parse_html(color_string: str) -> Dict[str, str]:
        """Parse a color string to CSS style dictionary.

        Args:
            color_string: Space-separated color specification like "red bold"
                         or "navy dim". Supports CSS color names, hex codes,
                         and style modifiers.

        Returns:
            Dictionary with CSS property-value pairs.

        Examples:
            >>> ColorParser.parse_html("red bold")
            {'color': 'red', 'font-weight': 'bold'}
            >>> ColorParser.parse_html("#FF5733")
            {'color': '#FF5733'}
            >>> ColorParser.parse_html("navy dim")
            {'color': 'navy', 'opacity': '0.7'}
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
                # Reset to normal (usually default)
                pass
            else:
                # Treat as color (CSS color name or hex code)
                style["color"] = part

        return style

    @staticmethod
    def colorize_ansi(text: str, color_string: str) -> str:
        """Wrap text with ANSI color codes.

        Args:
            text: Text to colorize.
            color_string: Color specification like "red bold".

        Returns:
            Text wrapped with ANSI escape codes.

        Examples:
            >>> ColorParser.colorize_ansi("ERROR", "red bold")
            '\x1b[31m\x1b[1mERROR\x1b[0m'
        """
        if not color_string or not color_string.strip():
            return text

        color_code = ColorParser.parse_ansi(color_string)
        return f"{color_code}{text}{Style.RESET_ALL}"

    @staticmethod
    def colorize_html(text: str, color_string: str) -> str:
        """Wrap text with HTML span tag and CSS styles.

        Args:
            text: Text to colorize.
            color_string: Color specification like "red bold".

        Returns:
            Text wrapped in HTML span with inline CSS.

        Examples:
            >>> ColorParser.colorize_html("ERROR", "red bold")
            '<span style="color:red;font-weight:bold;">ERROR</span>'
        """
        if not color_string or not color_string.strip():
            return text

        style_dict = ColorParser.parse_html(color_string)
        if not style_dict:
            return text

        style_str = ";".join([f"{k}:{v}" for k, v in style_dict.items()])
        return f'<span style="{style_str};">{text}</span>'


class ColoredConsoleFormatter(logging.Formatter):
    """Log formatter with color support for console output.

    This formatter supports multiple coloring strategies:
    - "message": Color only the log message text (original behavior)
    - "levelname": Color only the log level name
    - "format": Color each component of the format string separately

    Args:
        fmt: Log format string (e.g., "%(asctime)s [%(levelname)s]: %(message)s").
        datefmt: Date format string.
        style: Format style ('%', '{', or '$').
        color_target: What to colorize ("message", "levelname", or "format").
        enable_colors: Whether to enable color output.
    """

    # Default color scheme for ANSI console output
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
        "levelname": "",  # Will be overridden by LEVEL_COLORS
        "message": "",  # No color for message in format mode
        "pathname": "magenta",
        "filename": "magenta",
        "funcName": "cyan",
        "lineno": "dim",
    }

    def __init__(self, fmt=None, datefmt=None, style="%", color_target="format", enable_colors=True):
        super().__init__(fmt, datefmt, style)
        init(autoreset=True)  # Initialize colorama
        self.color_target = color_target
        self.enable_colors = enable_colors

    def format(self, record):
        """Format the log record with colors based on the configured strategy."""
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

        # Build component colors, using level-specific color for levelname
        component_colors = dict(self.COMPONENT_COLORS)
        component_colors["levelname"] = self.LEVEL_COLORS.get(record.levelname, "")

        # Set message (like parent format does)
        record.message = record.getMessage()

        # Format asctime first if needed
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        # Temporarily colorize each component
        for component, color in component_colors.items():
            if not color:
                continue

            if hasattr(record, component):
                original_value = getattr(record, component)
                original_values[component] = original_value
                value_str = str(original_value)
                colored_value = ColorParser.colorize_ansi(value_str, color)
                setattr(record, component, colored_value)

        # Use formatMessage instead of format to avoid asctime being overwritten
        result = self.formatMessage(record)

        # Handle exception info if present
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

        # Restore original values
        for component, original_value in original_values.items():
            setattr(record, component, original_value)

        return result


class ColoredHTMLFormatter(logging.Formatter):
    """Log formatter with color support for HTML/notebook output.

    This formatter supports multiple coloring strategies for HTML output:
    - "message": Color only the log message text (original behavior)
    - "levelname": Color only the log level name
    - "format": Color each component of the format string separately

    Args:
        fmt: Log format string (e.g., "%(asctime)s [%(levelname)s]: %(message)s").
        datefmt: Date format string.
        style: Format style ('%', '{', or '$').
        color_target: What to colorize ("message", "levelname", or "format").
        enable_colors: Whether to enable color output.
    """

    # Default color scheme for HTML output (CSS colors)
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
        "levelname": "",  # Will be overridden by LEVEL_COLORS
        "message": "",  # No color for message in format mode
        "pathname": "purple",
        "filename": "purple",
        "funcName": "teal",
        "lineno": "gray",
    }

    def __init__(self, fmt=None, datefmt=None, style="%", color_target="format", enable_colors=True):
        super().__init__(fmt, datefmt, style)
        self.color_target = color_target
        self.enable_colors = enable_colors

    def format(self, record):
        """Format the log record with HTML colors based on the configured strategy."""
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

        # Build component colors, using level-specific color for levelname
        component_colors = dict(self.COMPONENT_COLORS)
        component_colors["levelname"] = self.LEVEL_COLORS.get(record.levelname, "")

        # Set message (like parent format does)
        record.message = record.getMessage()

        # Format asctime first if needed
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        # Temporarily colorize each component
        for component, color in component_colors.items():
            if not color:
                continue

            if hasattr(record, component):
                original_value = getattr(record, component)
                original_values[component] = original_value
                value_str = str(original_value)
                colored_value = ColorParser.colorize_html(value_str, color)
                setattr(record, component, colored_value)

        # Use formatMessage instead of format to avoid asctime being overwritten
        result = self.formatMessage(record)

        # Handle exception info if present
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

        # Restore original values
        for component, original_value in original_values.items():
            setattr(record, component, original_value)

        return result


class SysLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:  # prevent empty returns
                super().emit(record)
                self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class TqdmLoggingHandler(logging.StreamHandler):
    """Use tqdm.write to avoid disrupting progress bars"""

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:  # prevent empty returns
                tqdm.tqdm.write(msg, end=self.terminator)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class NotebookLoggingHandler(logging.StreamHandler):
    """Use IPython's display function for HTML output"""

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:  # prevent empty returns in notebook
                display(HTML(msg))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
