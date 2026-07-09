import ast
import argparse
from typing import Literal, Union, List, Dict, Any

import numpy as np
import torch


def str2bool(value: Union[bool, str]):
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


def str2dict(args_list):
    result_dict = {}
    if args_list is not None and len(args_list) > 0:
        for arg in args_list:
            key, value = arg.split("=", 1)  # 使用 1 限制分割次数，避免错误处理包含 '=' 的值
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):  # 如果 literal_eval 失败，就把 value 当作字符串处理
                pass
            result_dict[key] = value
    return result_dict


def str2dtype(dtype: Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
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
    if dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "FP16":
        return np.float16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")


def dtype2str_np(dtype: torch.dtype) -> str:
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
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() == "cuda":
        return torch.device("cuda")
    else:
        raise argparse.ArgumentTypeError(f"Unexpected device `{device}`. dtype must be `cuda`, `cpu` or `auto`.")


def parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """
    Parse command-line override arguments into a flat dictionary with dotted keys.

    Args:
        override_list: List of override strings in format "key=value" or "section.key=value"

    Returns:
        Flat dictionary with dotted keys, suitable for ConfigContainer.set_nested()

    Examples:
        >>> parse_overrides(["train.max_epoches=10", "optimizer.learning_rate=0.001"])
        {'train.max_epoches': 10, 'optimizer.learning_rate': 0.001}
    """
    override_dict = {}

    for override in override_list:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value' format.")

        key, value = override.split("=", 1)

        # Try to infer type from value using ast.literal_eval
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If literal_eval fails, keep as string
            parsed_value = value

        override_dict[key] = parsed_value

    return override_dict
