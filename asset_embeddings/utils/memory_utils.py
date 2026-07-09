import os
import psutil

import torch


def check_vram(device: torch.device):
    """
    检查显存使用情况并返回当前显存的使用量和缓存量（以MB为单位）。

    返回:
        dict: {"allocated_memory": float, "cached_memory": float}
    """
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
    cached_memory = torch.cuda.memory_reserved(device) / (1024**2)
    return {"allocated_memory": allocated_memory, "cached_memory": cached_memory}


def get_memory_usage():
    """
    获取当前进程占用的内存（以MB为单位）及其占系统总内存的比例。

    返回:
        dict: {"used_memory_mb": float, "percent_of_total": float}
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss  # 使用中的常驻内存大小（单位为字节）
    total_memory = psutil.virtual_memory().total
    used_memory_mb = mem_info / 1024 / 1024
    percent = mem_info / total_memory * 100
    return {"used_memory_mb": used_memory_mb, "percent_of_total": percent}


def calculate_chunk_size(file_path, avg_row_size=None, memory_fraction=0.5):
    """
    根据系统可用内存和文件大小动态计算 chunk_size。用于分块读取大文件时自动指定 chunk_size 参数。

    :param file_path: 文件路径
    :param avg_row_size: 每行数据平均占用的字节数, 默认为 None, 会尝试从文件中读取部分数据进行估算
    :param memory_fraction: 使用的内存比例, 默认为 50%
    :return: 动态计算得到的 chunk_size
    """

    file_size = os.path.getsize(file_path)  # 获取文件大小（字节）
    available_memory = psutil.virtual_memory().available  # 获取系统可用内存（字节）

    if avg_row_size is None:
        with open(file_path, "r") as f:
            sample_lines = [next(f) for _ in range(100)]
            avg_row_size = sum(len(line.encode("utf-8")) for line in sample_lines) // len(
                sample_lines
            )  # 获取每行数据平均占用的字节数

    max_rows_in_memory = (available_memory * memory_fraction) // avg_row_size  # 计算可用行数
    estimated_rows = file_size // avg_row_size  # 计算文件总行数的估计值
    chunk_size = min(max_rows_in_memory, estimated_rows)  # 动态设置 chunk_size，确保不超过可用内存

    return max(chunk_size, 1)  # 确保 chunk_size 至少为 1
