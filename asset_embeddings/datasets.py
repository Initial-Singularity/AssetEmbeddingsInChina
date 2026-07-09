import json
import csv
import base64
import struct
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Literal, Optional, Iterator, Union

import torch
from torch.utils.data import Dataset
import pandas as pd

from transformers import PreTrainedTokenizerFast


def resolve_data_files(path: "Union[str, Path, List[str]]", format: str) -> "List[Path]":
    """将路径（单文件/目录/混合列表）解析为扁平、有序的文件列表。

    Args:
        path: 单个文件路径、目录路径，或包含文件/目录路径的列表。
        format: 数据格式后缀（不含点号），如 ``"csv"``、``"json"``。
              目录按 ``*.{format}`` glob；文件则校验后缀是否匹配。

    Returns:
        排序后的文件 Path 列表（目录内部按文件名排序，列表中多个路径按出现顺序拼接）。

    Raises:
        FileNotFoundError: 路径不存在，或解析结果为空。
        ValueError: 文件后缀与 *format* 不一致。
    """
    paths = path if isinstance(path, list) else [path]
    expected = format.lower()
    files: "List[Path]" = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_dir():
            files.extend(sorted(p.glob(f"*.{expected}")))
        else:
            suffix = p.suffix[1:].lower()
            if suffix != expected:
                raise ValueError(f"File {p} has format '{suffix}', expected '{format}'")
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No '{format}' files found in path: {path}")
    return files


class PortfolioDataEncoder:
    """Encoder/decoder for portfolio data.

    Reads (decodes) and writes (encodes) portfolio data in JSON, CSV, or binary
    format. Each record holds an ID, a string list of asset names/codes, and the
    corresponding holding-proportion lists.

    Args:
        format: Data format; one of "json", "csv", "binary".
        id_key: Name of the ID field.
        portfolio_key: Name of the holding asset name/code field.
        proportion1_key: Name of holding-proportion field 1.
        proportion2_key: Name of holding-proportion field 2.
    """

    def __init__(
        self,
        format: Literal["json", "csv", "binary"] = "json",
        id_key: str = "InvestorID",
        portfolio_key: str = "Portfolio",
        proportion1_key: str = "Proportion1",
        proportion2_key: str = "Proportion2",
    ):
        self.format = format
        self.id_key = id_key
        self.portfolio_key = portfolio_key
        self.proportion1_key = proportion1_key
        self.proportion2_key = proportion2_key

    # ========== 公共接口 ==========
    def encode(self, df: pd.DataFrame, file_path: str | Path, include_proportion: bool = False) -> None:
        """Encode a DataFrame and save it to a file."""
        file_path = Path(file_path).with_suffix(f".{self.format}")

        saver = {
            "json": self._save_json,
            "csv": self._save_csv,
            "binary": self._save_binary,
        }.get(self.format)

        if saver is None:
            raise ValueError(f"Unsupported format: {self.format}")
        saver(df, file_path, include_proportion)

    def decode(self, file_path: str | Path, include_proportion: bool = False) -> pd.DataFrame:
        """Decode from a file and return a DataFrame."""
        file_path = Path(file_path)
        format = file_path.suffix.lstrip(".")

        loader = {
            "json": self._load_json,
            "csv": self._load_csv,
            "binary": self._load_binary,
        }.get(format)

        if loader is None:
            raise ValueError(f"Unsupported format: {format}")

        return loader(file_path, include_proportion)

    # ========== JSON ==========
    def _save_json(self, df: pd.DataFrame, file_path: Path, include_proportion: bool) -> None:
        # Validate required columns
        required_columns = [self.id_key, self.portfolio_key]
        if include_proportion:
            required_columns.extend([self.proportion1_key, self.proportion2_key])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. " f"Available columns: {list(df.columns)}")

        # 转换为字典格式
        data = []
        for _, row in df.iterrows():
            record = {self.id_key: row[self.id_key], self.portfolio_key: row[self.portfolio_key]}
            if include_proportion and self.proportion1_key in df.columns:
                record[self.proportion1_key] = row[self.proportion1_key]
            if include_proportion and self.proportion2_key in df.columns:
                record[self.proportion2_key] = row[self.proportion2_key]
            data.append(record)

        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_json(self, file_path: Path, include_proportion: bool) -> pd.DataFrame:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)

        # Validate required columns
        required_columns = [self.id_key, self.portfolio_key]
        if include_proportion:
            required_columns.extend([self.proportion1_key, self.proportion2_key])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. " f"Available columns: {list(df.columns)}")

        return df

    # ========== CSV ==========
    def _save_csv(self, df: pd.DataFrame, file_path: Path, include_proportion: bool) -> None:
        """CSV保存方法 - 更新以支持类型控制"""
        # Validate required columns
        required_columns = [self.id_key, self.portfolio_key]
        if include_proportion:
            required_columns.extend([self.proportion1_key, self.proportion2_key])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. " f"Available columns: {list(df.columns)}")

        columns = [self.id_key, self.portfolio_key]
        if include_proportion and self.proportion1_key in df.columns:
            columns.append(self.proportion1_key)
        if include_proportion and self.proportion2_key in df.columns:
            columns.append(self.proportion2_key)

        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for _, row in df.iterrows():
                record = {
                    self.id_key: row[self.id_key],
                    # 假设portfolio是字符串列表
                    self.portfolio_key: self._serialize_list(
                        row[self.portfolio_key],
                        dtype="str",
                        quote_strings=False,  # 不为字符串添加引号
                    ),
                }
                if include_proportion:
                    # 假设proportion是浮点数列表，保留6位小数
                    record[self.proportion1_key] = self._serialize_list(
                        row[self.proportion1_key], dtype="float", precision=6
                    )
                    record[self.proportion2_key] = self._serialize_list(
                        row[self.proportion2_key], dtype="float", precision=6
                    )
                writer.writerow(record)

    def _load_csv(self, file_path: Path, include_proportion: bool) -> pd.DataFrame:
        """CSV加载方法 - 更新以支持类型控制"""
        data = []
        with file_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Check if file has required columns by examining the first row
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError("CSV file is empty")

            # Validate required columns
            required_columns = [self.id_key, self.portfolio_key]
            if include_proportion:
                required_columns.extend([self.proportion1_key, self.proportion2_key])
            missing_columns = [col for col in required_columns if col not in first_row]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}. " f"Available columns: {list(first_row.keys())}"
                )

            # Process first row
            record = {
                self.id_key: first_row[self.id_key],
                self.portfolio_key: self._deserialize_list(
                    first_row[self.portfolio_key], dtype="str", quote_strings=False
                ),
            }
            if include_proportion:
                record[self.proportion1_key] = self._deserialize_list(first_row[self.proportion1_key], dtype="float")
                record[self.proportion2_key] = self._deserialize_list(first_row[self.proportion2_key], dtype="float")
            data.append(record)

            # Process remaining rows
            for row in reader:
                record = {
                    self.id_key: row[self.id_key],
                    self.portfolio_key: self._deserialize_list(
                        row[self.portfolio_key], dtype="str", quote_strings=False
                    ),
                }
                if include_proportion:
                    record[self.proportion1_key] = self._deserialize_list(row[self.proportion1_key], dtype="float")
                    record[self.proportion2_key] = self._deserialize_list(row[self.proportion2_key], dtype="float")
                data.append(record)
        return pd.DataFrame(data)

    # ========== Binary ==========
    def _save_binary(self, df: pd.DataFrame, file_path: Path, include_proportion: bool) -> None:
        """Binary保存方法 - 更新以支持精确的类型控制"""
        # Validate required columns
        required_columns = [self.id_key, self.portfolio_key]
        if include_proportion:
            required_columns.extend([self.proportion1_key, self.proportion2_key])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. " f"Available columns: {list(df.columns)}")

        columns = [self.id_key, self.portfolio_key]
        if include_proportion:
            columns.extend([self.proportion1_key, self.proportion2_key])

        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for _, row in df.iterrows():
                record = {
                    self.id_key: row[self.id_key],
                    # 根据数据类型选择合适的序列化方式
                    # 如果portfolio包含字符串（如股票代码），使用string类型
                    # 如果portfolio包含数值，使用float64类型
                    self.portfolio_key: self._serialize_binary(
                        row[self.portfolio_key],
                        dtype="string",  # 假设portfolio是资产名称/代码
                        string_encoding="utf-8",
                    ),
                }
                if include_proportion:
                    # proportion通常是数值，使用float64获得更高精度
                    record[self.proportion1_key] = self._serialize_binary(row[self.proportion1_key], dtype="float64")
                    record[self.proportion2_key] = self._serialize_binary(row[self.proportion2_key], dtype="float64")
                writer.writerow(record)

    def _load_binary(self, file_path: Path, include_proportion: bool) -> pd.DataFrame:
        """Binary加载方法 - 更新以支持精确的类型控制"""
        data = []
        with file_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Check if file has required columns by examining the first row
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError("Binary file is empty")

            # Validate required columns
            required_columns = [self.id_key, self.portfolio_key]
            if include_proportion:
                required_columns.extend([self.proportion1_key, self.proportion2_key])
            missing_columns = [col for col in required_columns if col not in first_row]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}. " f"Available columns: {list(first_row.keys())}"
                )

            # Process first row
            record = {
                self.id_key: first_row[self.id_key],
                # 对应保存时的类型
                self.portfolio_key: self._deserialize_binary(
                    first_row[self.portfolio_key], dtype="string", string_encoding="utf-8"
                ),
            }
            if include_proportion:
                record[self.proportion1_key] = self._deserialize_binary(
                    first_row[self.proportion1_key], dtype="float64"
                )
                record[self.proportion2_key] = self._deserialize_binary(
                    first_row[self.proportion2_key], dtype="float64"
                )
            data.append(record)

            # Process remaining rows
            for row in reader:
                record = {
                    self.id_key: row[self.id_key],
                    # 对应保存时的类型
                    self.portfolio_key: self._deserialize_binary(
                        row[self.portfolio_key], dtype="string", string_encoding="utf-8"
                    ),
                }
                if include_proportion:
                    record[self.proportion1_key] = self._deserialize_binary(row[self.proportion1_key], dtype="float64")
                    record[self.proportion2_key] = self._deserialize_binary(row[self.proportion2_key], dtype="float64")
                data.append(record)
        return pd.DataFrame(data)

    # ========== 序列化辅助方法 ==========
    def _serialize_list(
        self,
        lst: List,
        dtype: Literal["str", "int", "float"] = "str",
        precision: int | None = None,
        separator: str = ",",
        escape_char: str = "\\",
        quote_strings: bool = False,
    ) -> str:
        """
        将列表序列化为分隔符分隔的字符串

        Args:
            lst: 待序列化的列表
            dtype: 数据类型控制 ("str", "int", "float")
            precision: 浮点数精度控制（仅对 float 类型有效）
            separator: 分隔符，默认逗号
            escape_char: 转义字符，用于处理分隔符冲突
            quote_strings: 是否为字符串添加引号
        """
        if not lst:
            return ""

        formatted_items = []
        for item in lst:
            if dtype == "float":
                if precision is not None:
                    formatted_item = f"{float(item):.{precision}f}"
                else:
                    formatted_item = str(float(item))
            elif dtype == "int":
                formatted_item = str(int(item))
            else:  # dtype == "str"
                str_item = str(item)
                # 转义分隔符
                if separator in str_item:
                    str_item = str_item.replace(separator, f"{escape_char}{separator}")
                if escape_char in str_item and escape_char != separator:
                    str_item = str_item.replace(escape_char, f"{escape_char}{escape_char}")

                # 可选的字符串引号
                if quote_strings:
                    str_item = f'"{str_item}"'
                formatted_item = str_item

            formatted_items.append(formatted_item)

        return separator.join(formatted_items)

    def _deserialize_list(
        self,
        s: str,
        dtype: Literal["str", "int", "float"] = "str",
        separator: str = ",",
        escape_char: str = "\\",
        quote_strings: bool = False,
        strict_parsing: bool = True,
    ) -> List:
        """
        将分隔符分隔的字符串反序列化为列表

        Args:
            s: 待反序列化的字符串
            dtype: 目标数据类型
            separator: 分隔符
            escape_char: 转义字符
            quote_strings: 字符串是否被引号包围
            strict_parsing: 严格解析模式，True时解析失败会抛出异常
        """
        if not s or s.strip() == "":
            return []

        # 简单分割（后续可扩展为更复杂的解析器）
        raw_items = s.split(separator)
        result = []

        for raw_item in raw_items:
            item = raw_item.strip()
            if not item:
                continue

            try:
                # 处理引号
                if quote_strings and item.startswith('"') and item.endswith('"'):
                    item = item[1:-1]  # 去除引号

                # 处理转义字符
                item = item.replace(f"{escape_char}{separator}", separator)
                if escape_char != separator:
                    item = item.replace(f"{escape_char}{escape_char}", escape_char)

                # 类型转换
                if dtype == "float":
                    converted_item = float(item)
                elif dtype == "int":
                    converted_item = int(item)
                else:  # dtype == "str"
                    converted_item = item

                result.append(converted_item)

            except (ValueError, TypeError) as e:
                if strict_parsing:
                    raise ValueError(f"Failed to parse item '{raw_item}' as {dtype}: {e}")
                else:
                    # 非严格模式下跳过无效项
                    continue

        return result

    def _serialize_binary(
        self,
        lst: List,
        dtype: Literal["float32", "float64", "int32", "int64", "string"] = "float32",
        endian: Literal["little", "big"] = "little",
        string_encoding: str = "utf-8",
    ) -> str:
        """
        将列表序列化为base64编码的二进制字符串

        Args:
            lst: 待序列化的列表
            dtype: 二进制数据类型
            endian: 字节序（little/big）
            string_encoding: 字符串编码方式（仅对string类型有效）
        """
        if not lst:
            return ""

        ba = bytearray()

        if dtype == "string":
            # 字符串类型的特殊处理
            # 格式：[总长度(4字节)] [字符串1长度(4字节)] [字符串1数据] [字符串2长度(4字节)] [字符串2数据] ...
            endian_prefix = "<" if endian == "little" else ">"
            len_format = endian_prefix + "I"  # 无符号整数用于长度

            # 先编码所有字符串，计算总长度
            encoded_strings = []
            total_length = 0
            for item in lst:
                try:
                    encoded_str = str(item).encode(string_encoding)
                    encoded_strings.append(encoded_str)
                    total_length += 4 + len(encoded_str)  # 4字节长度 + 字符串数据
                except (UnicodeEncodeError, TypeError) as e:
                    raise ValueError(f"Failed to encode string '{item}' with {string_encoding}: {e}")

            # 写入字符串数量
            ba.extend(struct.pack(len_format, len(lst)))

            # 写入每个字符串
            for encoded_str in encoded_strings:
                ba.extend(struct.pack(len_format, len(encoded_str)))  # 字符串长度
                ba.extend(encoded_str)  # 字符串数据

        else:
            # 数值类型的处理
            format_chars = {"float32": "f", "float64": "d", "int32": "i", "int64": "q"}

            # 字节序前缀
            endian_prefix = "<" if endian == "little" else ">"
            format_str = endian_prefix + format_chars[dtype]

            for item in lst:
                try:
                    if dtype.startswith("float"):
                        value = float(item)
                    else:  # int types
                        value = int(item)
                    ba.extend(struct.pack(format_str, value))
                except (ValueError, TypeError, struct.error) as e:
                    raise ValueError(f"Failed to pack item '{item}' as {dtype}: {e}")

        return base64.b64encode(ba).decode()

    def _deserialize_binary(
        self,
        s: str,
        dtype: Literal["float32", "float64", "int32", "int64", "string"] = "float32",
        endian: Literal["little", "big"] = "little",
        string_encoding: str = "utf-8",
    ) -> List:
        """
        将base64编码的二进制字符串反序列化为列表

        Args:
            s: base64编码的字符串
            dtype: 二进制数据类型
            endian: 字节序
            string_encoding: 字符串编码方式（仅对string类型有效）
        """
        if not s:
            return []

        try:
            binary = base64.b64decode(s)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 string: {e}")

        result = []

        if dtype == "string":
            # 字符串类型的特殊处理
            endian_prefix = "<" if endian == "little" else ">"
            len_format = endian_prefix + "I"
            len_size = 4  # unsigned int 大小

            if len(binary) < len_size:
                raise ValueError("Invalid binary data: insufficient data for string count")

            # 读取字符串数量
            try:
                (string_count,) = struct.unpack(len_format, binary[:len_size])
            except struct.error as e:
                raise ValueError(f"Failed to unpack string count: {e}")

            offset = len_size

            # 读取每个字符串
            for i in range(string_count):
                if offset + len_size > len(binary):
                    raise ValueError(f"Invalid binary data: insufficient data for string {i} length")

                # 读取字符串长度
                try:
                    (str_len,) = struct.unpack(len_format, binary[offset : offset + len_size])
                except struct.error as e:
                    raise ValueError(f"Failed to unpack string {i} length: {e}")

                offset += len_size

                if offset + str_len > len(binary):
                    raise ValueError(f"Invalid binary data: insufficient data for string {i} content")

                # 读取字符串数据
                try:
                    str_data = binary[offset : offset + str_len]
                    decoded_str = str_data.decode(string_encoding)
                    result.append(decoded_str)
                except UnicodeDecodeError as e:
                    raise ValueError(f"Failed to decode string {i} with {string_encoding}: {e}")

                offset += str_len

        else:
            # 数值类型的处理
            format_info = {"float32": ("f", 4), "float64": ("d", 8), "int32": ("i", 4), "int64": ("q", 8)}

            format_char, item_size = format_info[dtype]
            endian_prefix = "<" if endian == "little" else ">"
            format_str = endian_prefix + format_char

            for i in range(0, len(binary), item_size):
                if i + item_size <= len(binary):
                    try:
                        (value,) = struct.unpack(format_str, binary[i : i + item_size])
                        result.append(value)
                    except struct.error as e:
                        raise ValueError(f"Failed to unpack binary data at position {i}: {e}")
                else:
                    # 数据不完整，可选择抛出异常或跳过
                    raise ValueError(f"Incomplete binary data: expected {item_size} bytes, got {len(binary) - i}")

        return result


class Word2VecDataset:
    """Word2Vec training dataset."""

    def __init__(
        self,
        path: Union[str, Path, List[str]],
        format: str = "csv",
        id_key: str = "InvestorID",
        portfolio_key: str = "Portfolio",
    ):
        """Initialize the Word2Vec dataset.

        Args:
            path: Dataset path — a file, a folder, or a list of file paths.
            format: Data format; "csv" or "json".
            id_key: Name of the investor-ID field.
            portfolio_key: Name of the holdings field.
        """
        self.format = format
        self.id_key = id_key
        self.portfolio_key = portfolio_key
        self.include_proportion = False

        self.files = resolve_data_files(path, self.format)
        self.encoder = PortfolioDataEncoder(self.format, self.id_key, self.portfolio_key)

    def _process_row(self, portfolio: List[str]) -> List[str]:
        """处理单行数据，转换为token序列"""
        return ["[CLS]"] + list(portfolio) + ["[SEP]"]

    def _process_file(self, file_path: Path) -> Iterator[List[str]]:
        """处理单个文件"""
        df = self.encoder.decode(str(file_path), self.include_proportion)
        for _, row in df.iterrows():
            yield self._process_row(row[self.portfolio_key])

    def __iter__(self) -> Iterator[List[str]]:
        """迭代生成数据"""
        for file_path in self.files:
            yield from self._process_file(file_path)


class RSDataset:
    """RS训练数据集，使用PortfolioDataEncoder加载数据"""

    def __init__(self, path: Union[str, Path], **encoder_kwargs):
        """
        初始化RS数据集

        Args:
            path: 数据路径
            **encoder_kwargs: PortfolioDataEncoder的参数
        """
        self.path = Path(path)
        self.encoder = PortfolioDataEncoder(**encoder_kwargs)

    def load_data(self, include_proportion: bool = False) -> Dict[str, Union[List[str], Tuple[List[str], List[float]]]]:
        """
        加载并返回投资者持仓数据字典

        Args:
            include_proportion: 是否包含持仓比例

        Returns:
            如果 include_proportion=False: Dict[InvestorID, List[Assets]]
            如果 include_proportion=True: Dict[InvestorID, (List[Assets], List[Proportion1])]
        """
        if self.path.is_dir():
            # 加载目录中的所有文件并合并
            dfs = []
            for file_path in self.path.glob(f"*.{self.encoder.format}"):
                df = self.encoder.decode(file_path, include_proportion=include_proportion)
                dfs.append(df)
            if not dfs:
                raise FileNotFoundError(f"目录 {self.path} 中没有找到 {self.encoder.format} 格式文件")
            df = pd.concat(dfs, ignore_index=True)
        else:
            # 加载单个文件
            df = self.encoder.decode(self.path, include_proportion=include_proportion)

        # 转换为字典格式
        result = {}
        for _, row in df.iterrows():
            investor_id = str(row[self.encoder.id_key]).strip()
            portfolio = row[self.encoder.portfolio_key]

            # 处理portfolio格式
            if isinstance(portfolio, str):
                portfolio = [asset.strip() for asset in portfolio.split(",") if asset.strip()]
            elif not isinstance(portfolio, list):
                portfolio = []

            if include_proportion:
                proportion1 = row[self.encoder.proportion1_key]
                # 处理proportion格式
                if isinstance(proportion1, str):
                    proportion1 = [float(p.strip()) for p in proportion1.split(",") if p.strip()]
                elif not isinstance(proportion1, list):
                    proportion1 = []
                result[investor_id] = (portfolio, proportion1)
            else:
                result[investor_id] = portfolio

        return result


class TokenMasker:
    """Token掩码处理器"""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    def mask_tokens_random(
        self,
        input_ids: torch.Tensor,
        mask_prob: float = 0.15,
        special_tokens_mask: Optional[torch.Tensor] = None,
        proportions1: Optional[torch.Tensor] = None,
        proportions2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """随机掩码策略（BERT风格）"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool,
            )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mask_positions = torch.bernoulli(probability_matrix).bool()
        labels[~mask_positions] = -100

        # BERT掩码策略：80% [MASK], 10% random, 10% unchanged.
        # Shuffle the masked positions before partitioning so the 80/10/10 split is assigned
        # at random rather than by sequence order (release-audit C-M4: torch.where returns
        # ascending indices, which otherwise biased earlier-ranked holdings toward [MASK]).
        mask_indices = torch.where(mask_positions)[0]
        mask_indices = mask_indices[torch.randperm(len(mask_indices))]

        # 80% 替换为 [MASK]
        mask_count = int(0.8 * len(mask_indices))
        input_ids[mask_indices[:mask_count]] = self.mask_token_id

        # 10% 替换为随机token
        random_count = int(0.1 * len(mask_indices))
        if random_count > 0:
            random_indices = mask_indices[mask_count : mask_count + random_count]
            random_tokens = torch.randint(len(self.tokenizer), (len(random_indices),), dtype=torch.long)
            input_ids[random_indices] = random_tokens

        # 处理proportions
        if proportions1 is not None:
            proportions1[special_tokens_mask] = 0.0
            proportions1[mask_positions] = 0.0
        if proportions2 is not None:
            proportions2[special_tokens_mask] = 0.0
            proportions2[mask_positions] = 0.0
        return input_ids, labels, proportions1, proportions2

    def mask_tokens_fixed(
        self,
        input_ids: torch.Tensor,
        mask_indices: List[int],
        special_tokens_mask: Optional[torch.Tensor] = None,
        proportions1: Optional[torch.Tensor] = None,
        proportions2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """固定位置掩码策略"""
        labels = input_ids.clone()
        probability_matrix = torch.zeros_like(input_ids, dtype=torch.float)

        # 设置要掩码的位置
        for idx in mask_indices:
            if idx < len(probability_matrix):
                probability_matrix[idx] = 1.0

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool,
            )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mask_positions = probability_matrix.bool()
        labels[~mask_positions] = -100

        # 100% 替换为 [MASK]
        input_ids[mask_positions] = self.mask_token_id

        # 处理proportions
        if proportions1 is not None:
            proportions1[special_tokens_mask] = 0.0
            proportions1[mask_positions] = 0.0
        if proportions2 is not None:
            proportions2[special_tokens_mask] = 0.0
            proportions2[mask_positions] = 0.0

        return input_ids, labels, proportions1, proportions2


class FileCache:
    """线程安全的文件缓存管理器"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()

    def get(self, file_path: str, loader_func) -> pd.DataFrame:
        """获取文件数据，如果不在缓存中则加载"""
        with self._lock:
            if file_path in self._cache:
                # 更新访问顺序
                self._access_order.remove(file_path)
                self._access_order.append(file_path)
                return self._cache[file_path]

            # 如果缓存已满，移除最久未使用的项
            if len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            # 加载新文件
            data = loader_func(file_path)
            self._cache[file_path] = data
            self._access_order.append(file_path)
            return data

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class AssetBERTMLMDataset(Dataset):
    """Asset BERT MLM training dataset (multiprocessing-capable)."""

    def __init__(
        self,
        path: Union[str, Path, List[str]],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 20,
        mask_prob: Optional[float] = 0.15,
        mask_indices: Optional[List[int]] = None,
        format: Literal["csv", "json"] = "csv",
        include_proportion: bool = False,
        cache_size: int = 10,
        num_repeats: int = 1,
        id_key: str = "InvestorID",
        portfolio_key: str = "Portfolio",
        proportion1_key: str = "Proportion1",
        proportion2_key: str = "Proportion2",
    ):
        """Initialize the Asset BERT MLM dataset.

        Args:
            path: Data file, folder path, or list of files.
            tokenizer: Pretrained tokenizer.
            max_length: Maximum sequence length.
            mask_prob: MLM masking probability (for BERT training).
            mask_indices: Fixed mask positions (for ASMP training and testing).
            format: Data format.
            include_proportion: Whether to include holding proportions.
            cache_size: File-cache size.
            num_repeats: Dataset repeat factor (extends the training-set length to
                reduce frequent epoch restarts on small samples).
            id_key: Name of the investor-ID field.
            portfolio_key: Name of the holdings field.
            proportion1_key: Name of holding-proportion field 1.
            proportion2_key: Name of holding-proportion field 2.
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_indices = mask_indices
        self.format = format
        self.include_proportion = include_proportion
        self.cache_size = cache_size
        self.num_repeats = int(num_repeats) if num_repeats is not None else 1
        if self.num_repeats < 1:
            raise ValueError("repeat 必须为正整数")
        self.id_key = id_key
        self.portfolio_key = portfolio_key
        self.proportion1_key = proportion1_key
        self.proportion2_key = proportion2_key

        # 初始化组件（延迟初始化，避免pickle问题）
        self.encoder = None
        self.masker = None
        self.file_cache = None

        # 获取文件列表和行数统计
        self._setup_files()

    def _ensure_components(self):
        """确保组件已初始化（延迟初始化）"""
        if self.encoder is None:
            self.encoder = PortfolioDataEncoder(
                self.format,
                self.id_key,
                self.portfolio_key,
                proportion1_key=self.proportion1_key,
                proportion2_key=self.proportion2_key,
            )
        if self.masker is None:
            self.masker = TokenMasker(self.tokenizer)
        if self.file_cache is None:
            self.file_cache = FileCache(self.cache_size)

    def _setup_files(self):
        """设置文件列表和行数统计"""
        self.files = resolve_data_files(self.path, self.format)

        # 计算每个文件的行数
        self.file_line_counts = {}
        for file_path in self.files:
            line_count = self._count_file_lines(file_path)
            self.file_line_counts[str(file_path)] = line_count

        self.total_rows = sum(self.file_line_counts.values())

    def _count_file_lines(self, file_path: Path) -> int:
        """计算文件行数（记录数）。

        Assumes one record per physical line — valid here because the CSV serialization
        (``PortfolioDataEncoder._serialize_list`` comma-joins, no embedded newlines) never
        produces quoted multiline fields. (release-audit C-L4)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            if self.format == "csv":
                return sum(1 for _ in f) - 1  # 减去header行
            else:
                return sum(1 for _ in f)

    def _load_file_data(self, file_path: str) -> pd.DataFrame:
        """加载文件数据的辅助函数"""
        self._ensure_components()
        return self.encoder.decode(file_path, self.include_proportion)

    def _get_file_and_line_index(self, global_index: int) -> Tuple[str, int]:
        """根据全局索引获取文件路径和行索引"""
        cumulative_lines = 0
        for file_path, line_count in self.file_line_counts.items():
            if global_index < cumulative_lines + line_count:
                return file_path, global_index - cumulative_lines
            cumulative_lines += line_count
        raise IndexError(f"索引 {global_index} 超出数据集范围")

    def _prepare_tokens_and_proportions(
        self, row: pd.Series
    ) -> Tuple[List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """准备tokens和proportions"""
        if self.include_proportion:
            tokens = row[self.portfolio_key]
            proportions1 = torch.tensor(row[self.proportion1_key], dtype=torch.float32)
            proportions2 = torch.tensor(row[self.proportion2_key], dtype=torch.float32)
        else:
            tokens = row[self.portfolio_key]
            proportions1 = None
            proportions2 = None

        return tokens, proportions1, proportions2

    def _tokenize_and_pad(
        self, tokens: List[str], proportions1: Optional[torch.Tensor], proportions2: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """标记化和填充"""
        # 转换为token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 添加特殊tokens
        input_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # 处理proportions
        if proportions1 is not None:
            # 为[CLS]和[SEP]添加0值
            proportions1 = torch.cat(
                [torch.zeros(1, dtype=torch.float32), proportions1, torch.zeros(1, dtype=torch.float32)]
            )
        if proportions2 is not None:
            # 为[CLS]和[SEP]添加0值
            proportions2 = torch.cat(
                [torch.zeros(1, dtype=torch.float32), proportions2, torch.zeros(1, dtype=torch.float32)]
            )

        # 截断或填充
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            if proportions1 is not None:
                proportions1 = proportions1[: self.max_length]
            if proportions2 is not None:
                proportions2 = proportions2[: self.max_length]
        else:
            pad_length = self.max_length - len(input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0

            input_ids.extend([pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)

            if proportions1 is not None:
                proportions1 = torch.cat([proportions1, torch.zeros(pad_length, dtype=torch.float32)])
            if proportions2 is not None:
                proportions2 = torch.cat([proportions2, torch.zeros(pad_length, dtype=torch.float32)])

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids_tensor, attention_mask_tensor, proportions1, proportions2

    def __len__(self) -> int:
        return self.total_rows * self.num_repeats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        # 确保所有组件都已初始化
        self._ensure_components()

        # 越界检查
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出数据集范围 [0, {len(self) - 1}]")

        # 支持重复：将全局索引映射到原始数据范围
        base_idx = idx % self.total_rows

        file_path, line_idx = self._get_file_and_line_index(base_idx)

        # 使用缓存加载文件
        df = self.file_cache.get(file_path, self._load_file_data)
        row = df.iloc[line_idx]

        # 准备数据
        tokens, proportions1, proportions2 = self._prepare_tokens_and_proportions(row)
        input_ids, attention_mask, proportions1, proportions2 = self._tokenize_and_pad(
            tokens, proportions1, proportions2
        )

        # 应用掩码
        if self.mask_prob is not None:
            input_ids, labels, proportions1, proportions2 = self.masker.mask_tokens_random(
                input_ids, self.mask_prob, proportions1=proportions1, proportions2=proportions2
            )
        elif self.mask_indices is not None:
            input_ids, labels, proportions1, proportions2 = self.masker.mask_tokens_fixed(
                input_ids, self.mask_indices, proportions1=proportions1, proportions2=proportions2
            )
        else:
            # 不进行掩码处理
            input_ids, labels, proportions1, proportions2 = self.masker.mask_tokens_fixed(
                input_ids, [], proportions1=proportions1, proportions2=proportions2
            )

        # 确保proportions不为None
        if proportions1 is None:
            proportions1 = torch.zeros(self.max_length, dtype=torch.float32)
        if proportions2 is None:
            proportions2 = torch.zeros(self.max_length, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "proportions1": proportions1,
            "proportions2": proportions2,
        }

    def __getstate__(self):
        """支持pickle序列化"""
        state = self.__dict__.copy()
        # 移除不能序列化的组件，在子进程中重新初始化
        state["encoder"] = None
        state["masker"] = None
        state["file_cache"] = None
        return state

    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.__dict__.update(state)
        # 组件将在第一次访问时重新初始化
