"""String parsers for argparse `type=` callbacks and `key=value` config overrides.

Booleans, dicts, dtype and device names.
"""

import ast
import argparse
from typing import Literal, Optional, Union, List, Dict, Any

import numpy as np
import torch


def str2bool(value: Union[bool, str]) -> Optional[bool]:
    """Parse a boolean flag value: `true/1/yes/y/t` and `false/0/no/n/f`, case-insensitive; bools pass through.

    Args:
        value: A bool, or one of the accepted strings.

    Returns:
        The bool; a string outside both sets falls through and returns `None`.

    Raises:
        argparse.ArgumentTypeError: If `value` is neither a bool nor a string.
    """
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in {"false", "0", "no", "n", "f"}:
            return False
        elif value.lower() in {"true", "1", "yes", "y", "t"}:
            return True
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value or bool like string expected. Get unexpected value {value}, whose type is {type(value)}"
        )


def str2dict(args_list: Optional[List[str]]) -> Dict[str, Any]:
    """Turn `key=value` strings into a dict, parsing each value as a Python literal where possible.

    Args:
        args_list: The `key=value` strings; `None` or empty gives an empty dict.

    Returns:
        The dict; a value that is not a Python literal stays a string.
    """
    result_dict = {}
    if args_list is not None and len(args_list) > 0:
        for arg in args_list:
            key, value = arg.split("=", 1)  # split on the first '=' only: a value may contain '=' itself
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):  # not a Python literal: keep the raw string
                pass
            result_dict[key] = value
    return result_dict


def str2dtype(dtype: Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    """Map a precision name to the torch dtype.

    Args:
        dtype: One of `FP32`, `FP64`, `FP16`, `BF16`.

    Returns:
        The torch dtype.

    Raises:
        ValueError: If the name is not one of the four.
    """
    if dtype == "FP32":
        return torch.float32
    elif dtype == "FP64":
        return torch.float64
    elif dtype == "FP16":
        return torch.float16
    elif dtype == "BF16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")


def dtype2str(dtype: torch.dtype) -> str:
    """Inverse of `str2dtype`: the precision name of a torch floating dtype.

    Args:
        dtype: One of `torch.float32`, `torch.float64`, `torch.float16`, `torch.bfloat16`.

    Returns:
        `FP32`, `FP64`, `FP16` or `BF16`.

    Raises:
        ValueError: If the dtype is not one of the four.
    """
    if dtype == torch.float32:
        return "FP32"
    elif dtype == torch.float64:
        return "FP64"
    elif dtype == torch.float16:
        return "FP16"
    elif dtype == torch.bfloat16:
        return "BF16"
    else:
        raise ValueError(
            f"Unexpected dtype `{dtype}`. dtype must be `torch.float32`, `torch.float64`, `torch.float16` or `torch.bfloat16`."
        )


def str2dtype_np(dtype: Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    """Map a precision name to the NumPy dtype; NumPy has no bfloat16, so `BF16` is rejected.

    Args:
        dtype: One of `FP32`, `FP64`, `FP16`.

    Returns:
        The NumPy dtype (the return annotation is inherited from `str2dtype`).

    Raises:
        ValueError: If the name is not one of the three.
    """
    if dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "FP16":
        return np.float16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")


def dtype2str_np(dtype: torch.dtype) -> str:
    """Inverse of `str2dtype_np`: the precision name of a NumPy floating dtype.

    Args:
        dtype: One of `numpy.float32`, `numpy.float64`, `numpy.float16` (the annotation is inherited from
            `dtype2str`).

    Returns:
        `FP32`, `FP64` or `FP16`.

    Raises:
        ValueError: If the dtype is not one of the three.
    """
    if dtype == np.float32:
        return "FP32"
    elif dtype == np.float64:
        return "FP64"
    elif dtype == np.float16:
        return "FP16"
    else:
        raise ValueError(
            f"Unexpected dtype `{dtype}`. dtype must be `numpy.float32`, `numpy.float64` or `numpy.float16`."
        )


def str2device(device: Literal["auto", "cpu", "cuda"]) -> torch.device:
    """Map a device name to a `torch.device`; `auto` picks CUDA when it is available, else CPU.

    Args:
        device: `auto`, `cpu` or `cuda` (the last two case-insensitive).

    Returns:
        The device.

    Raises:
        argparse.ArgumentTypeError: If the name is not one of the three.
    """
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() == "cuda":
        return torch.device("cuda")
    else:
        raise argparse.ArgumentTypeError(f"Unexpected device `{device}`. dtype must be `cuda`, `cpu` or `auto`.")


def add_run_args(parser: argparse.ArgumentParser, override_example: str) -> argparse.ArgumentParser:
    """Add the six flags every training and evaluation CLI shares.

    ``-c/--config``, ``-l/--log``, ``-r/--result_file``, ``-e/--experiment``, ``-o/--override`` and
    ``-v/--verbose``; the scripts differ only in their description and the config layout of their epilog.

    Args:
        parser: The script's parser.
        override_example: Overrides shown in the ``--override`` help, e.g. ``"ridge_alpha=null seed=7"``.

    Returns:
        The same parser.
    """
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the JSON config file. Required.")
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        nargs="*",
        default=None,
        help="Log file path(s); without one, logs go to the console only.",
    )
    parser.add_argument(
        "--result_file", "-r", type=str, default=None, help="SQLite file the run's records are written to. Optional."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment family stored in the runs table (default: derived from the config path, e.g. "
        "configs/main/... gives main). Optional.",
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="+",
        default=None,
        help=f"Override config fields, key=value with dot notation for nested keys (e.g. {override_example}). Optional.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="DEBUG console output. Default: INFO."
    )
    return parser


def parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """Parse command-line overrides into a flat dict keyed by their dotted paths.

    `"train.max_epochs=10"` becomes `{"train.max_epochs": 10}`: the key is kept as written, so a nested
    config can resolve the dotted path, and the value is parsed as a Python literal where possible.

    Args:
        override_list: Strings of the form `key=value` or `section.key=value`.

    Returns:
        The dict; a value that is not a Python literal stays a string.

    Raises:
        ValueError: If a string has no `=`.
    """
    override_dict = {}

    for override in override_list:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value' format.")

        key, value = override.split("=", 1)

        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):  # not a Python literal: keep the raw string
            parsed_value = value

        override_dict[key] = parsed_value

    return override_dict
