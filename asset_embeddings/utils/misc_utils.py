import os
from typing import Optional, List, Dict

import pandas as pd
import torch


def check(tensor: torch.Tensor):
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))


def check_attr_dict_match(obj, dic: Dict, names: List[str]):
    for name in names:
        assert hasattr(obj, name) and name in dic
        assert getattr(obj, name) == dic[name]


def ensure_dir(directory: str):
    """确保文件夹存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_common_root(dirs: str):
    if not dirs:
        raise ValueError("The list of directories is empty.")
    try:
        common_root = os.path.commonpath(dirs)
        if not any([os.path.commonprefix([dir_, common_root]) == common_root for dir_ in dirs]):
            raise RuntimeError("No common root directory found.")
        return common_root
    except ValueError:
        raise RuntimeError("No common root directory found.")


def expand_list_columns(df: pd.DataFrame, col: str, new_col: Optional[str] = None) -> pd.DataFrame:
    """
    将 DataFrame 中的列表列展开为多列 new_col_i
    :param df: 原始 DataFrame
    :param col: 包含列表的列名
    :param new_col: 新列的前缀名称（可选），默认使用原始列名作为前缀
    :return: 展开后的 DataFrame
    """
    # 如果 new_col 未指定，则使用原列名作为前缀
    prefix = new_col if new_col is not None else col

    # 获取最大长度，用于确定生成多少列
    max_len = df[col].str.len().max()

    # 生成新的列
    expanded_df = pd.DataFrame(df[col].tolist(), index=df.index, columns=[f"{prefix}_{i+1}" for i in range(max_len)])

    # 合并回原始 DataFrame 并删除原始列表列
    result_df = df.join(expanded_df).drop(columns=[col])

    return result_df
