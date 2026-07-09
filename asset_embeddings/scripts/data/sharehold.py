import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Callable, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asset_embeddings.datasets import PortfolioDataEncoder
from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.utils.log_utils import log_exceptions_inclass
from asset_embeddings import utils


# -------------------- Period Mapping --------------------
# 预设频率类型: Y=年度, HY=半年, Q=季度, M=月度, N=不分组
PresetFrequency = Literal["Y", "HY", "Q", "M", "N"]

# 自定义 period 映射函数签名: 接收 Timestamp，返回 period 字符串
PeriodMapperFunc = Callable[[pd.Timestamp], str]

# 混合类型：支持预设字符串或自定义函数
PeriodMapper = Union[PresetFrequency, PeriodMapperFunc]


def _period_yearly(dt: pd.Timestamp) -> str:
    """年度: 2023, 2024, ..."""
    return str(dt.year)


def _period_semiannual(dt: pd.Timestamp) -> str:
    """半年: 2023H1, 2023H2"""
    return f"{dt.year}H{1 if dt.month <= 6 else 2}"


def _period_quarterly(dt: pd.Timestamp) -> str:
    """季度: 2023Q1, 2023Q2, ..."""
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def _period_monthly(dt: pd.Timestamp) -> str:
    """月度: 2023-01, 2023-02, ..."""
    return f"{dt.year}-{dt.month:02d}"


def _period_none(dt: pd.Timestamp) -> str:
    """不分组: 所有数据归入 'all'"""
    return "all"


# 预设映射表
PERIOD_MAPPERS: Dict[str, PeriodMapperFunc] = {
    "Y": _period_yearly,
    "HY": _period_semiannual,
    "Q": _period_quarterly,
    "M": _period_monthly,
    "N": _period_none,
}

# -------------------- Aggregation Strategy --------------------
# 聚合策略: all=保留所有, last=最新, first=最早, max=比例最大, mean=平均值
AggregationStrategy = Literal["all", "last", "first", "max", "mean"]


# -------------------- Loader --------------------
class ShareholdingLoader:
    def __init__(
        self,
        logger: logging.Logger,
        data_folder: str,
        source_shareholder_id_column: str = "ShareHolderID",
        source_stock_symbol_column: str = "Symbol",
        source_end_date_column: str = "EndDate",
        date_format: str = None,
        frequency: PeriodMapper = "Q",
    ):
        self.data_folder: Path = Path(data_folder)
        self.source_shareholder_id_column: str = source_shareholder_id_column
        self.source_stock_symbol_column: str = source_stock_symbol_column
        self.source_end_date_column: str = source_end_date_column
        self.date_format: str = date_format
        self.logger: logging.Logger = logger

        # 解析 frequency 参数：预设字符串转换为映射函数
        if callable(frequency):
            self._period_mapper: PeriodMapperFunc = frequency
            self._frequency_name: str = "custom"
        elif frequency in PERIOD_MAPPERS:
            self._period_mapper = PERIOD_MAPPERS[frequency]
            self._frequency_name = frequency
        else:
            raise ValueError(f"Unknown frequency: {frequency}. Available presets: {list(PERIOD_MAPPERS.keys())}")

    @log_exceptions_inclass("logger")
    def load_all(self) -> Dict[str, pd.DataFrame]:
        files = sorted(f for f in self.data_folder.glob("*.csv") if not f.name.startswith("~$"))

        if not files:
            self.logger.warning(f"No CSV files found in {self.data_folder}")
            return {}

        self.logger.info(f"Starting data loading: {len(files)} CSV files from {self.data_folder}")
        period_data: Dict[str, List[pd.DataFrame]] = {}
        total_records = 0
        skipped_files = 0

        # frequency="N" 时不解析日期列
        no_date_grouping = self._frequency_name == "N"

        for path in tqdm(files, desc="Loading CSV files", unit="file"):
            self.logger.debug(f"Processing file: {path.name}")

            if no_date_grouping:
                # 无日期模式：不解析日期列
                df = pd.read_csv(
                    path,
                    dtype={self.source_stock_symbol_column: str, self.source_shareholder_id_column: str},
                )

                self.logger.debug(f"Initial records: {len(df)}")
                # 检查必要列（不包括日期列）
                if df.empty:
                    self.logger.warning(f"Empty file: {path}, skipping.")
                    continue
                for col in [self.source_shareholder_id_column, self.source_stock_symbol_column]:
                    if col not in df.columns:
                        self.logger.error(f"Missing column `{col}` in file: {path}, skipping.")
                        continue

                # 对列进行字符串处理
                df[self.source_stock_symbol_column] = (
                    df[self.source_stock_symbol_column].astype(str).str.strip().str.zfill(6)
                )
                df[self.source_shareholder_id_column] = df[self.source_shareholder_id_column].astype(str).str.strip()

                total_records += len(df)
                # 所有数据归入 "all" 键
                period_data.setdefault("all", []).append(df)
                self.logger.debug(f"all: {len(df)} records")
            else:
                # 读取CSV文件
                df = pd.read_csv(
                    path,
                    dtype={
                        self.source_end_date_column: str,
                        self.source_stock_symbol_column: str,
                        self.source_shareholder_id_column: str,
                    },
                )
                df[self.source_end_date_column] = pd.to_datetime(
                    df[self.source_end_date_column],
                    format=self.date_format,  # 例如 "%Y%m%d"
                    errors="raise",
                )

                # 检查数据完整性
                if df.empty:
                    self.logger.warning(f"Empty file skipped: {path.name}")
                    skipped_files += 1
                    continue

                required_columns = [
                    self.source_shareholder_id_column,
                    self.source_stock_symbol_column,
                    self.source_end_date_column,
                ]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.logger.error(f"Missing columns {missing_columns} in {path.name}, skipping")
                    skipped_files += 1
                    continue

                self.logger.debug(f"Loaded {len(df):,} records from {path.name}")
                total_records += len(df)

                # 对列进行字符串处理
                df[self.source_stock_symbol_column] = (
                    df[self.source_stock_symbol_column].astype(str).str.strip().str.zfill(6)
                )
                df[self.source_shareholder_id_column] = df[self.source_shareholder_id_column].astype(str).str.strip()

                # 按时间周期分组
                df["Period"] = df[self.source_end_date_column].apply(self._period_mapper)

                for period, group in df.groupby("Period"):
                    period_data.setdefault(period, []).append(group)
                    self.logger.debug(f"{period}: +{len(group):,} records")

        # 合并同期数据
        result = {}
        for period, dfs in period_data.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            result[period] = merged_df
            self.logger.debug(f"Period {period}: {len(merged_df):,} total records")

        # 输出统计摘要
        self.logger.info(f"Data loading completed: {len(result)} periods, {total_records:,} total records")
        if skipped_files > 0:
            self.logger.warning(f"{skipped_files} files were skipped due to errors")

        return result


# -------------------- Aggregator --------------------
class ShareholdingAggregator:
    """聚合同一 period 内的重复 (investor, stock) 记录"""

    def __init__(
        self,
        logger: logging.Logger,
        strategy: AggregationStrategy = "all",
        source_shareholder_id_column: str = "ShareHolderID",
        source_stock_symbol_column: str = "Symbol",
        source_end_date_column: str = "EndDate",
        source_proportion1_column: str = "HoldProportion",
        source_proportion2_column: str = "HoldProportion1",
    ):
        self.strategy: AggregationStrategy = strategy
        self.source_shareholder_id_column: str = source_shareholder_id_column
        self.source_stock_symbol_column: str = source_stock_symbol_column
        self.source_end_date_column: str = source_end_date_column
        self.source_proportion1_column: str = source_proportion1_column
        self.source_proportion2_column: str = source_proportion2_column
        self.logger: logging.Logger = logger

    @log_exceptions_inclass("logger")
    def aggregate(self, period_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not period_data:
            self.logger.warning("No data to aggregate")
            return {}

        # all 策略：不做任何处理
        if self.strategy == "all":
            self.logger.info("Aggregation strategy is 'all', skipping aggregation")
            return period_data

        self.logger.info(f"Starting aggregation with strategy: {self.strategy}")
        result = {}
        group_keys = [self.source_shareholder_id_column, self.source_stock_symbol_column]

        for period, df in tqdm(period_data.items(), desc="Aggregating periods", unit="period"):
            if df.empty:
                self.logger.warning(f"Period {period}: empty data, skipping")
                continue

            original_count = len(df)

            # 检查日期列是否存在（last/first 策略需要）
            has_date_column = self.source_end_date_column in df.columns
            if self.strategy in ("last", "first") and not has_date_column:
                raise ValueError(
                    f"Strategy '{self.strategy}' requires date column '{self.source_end_date_column}', "
                    f"but it's not present in data. Use 'all', 'max', or 'mean' strategy instead, "
                    f"or ensure frequency is not 'N'."
                )

            if self.strategy == "last":
                df_agg = (
                    df.sort_values(self.source_end_date_column, kind="mergesort")
                    .groupby(group_keys, as_index=False)
                    .last()
                )
            elif self.strategy == "first":
                df_agg = (
                    df.sort_values(self.source_end_date_column, kind="mergesort")
                    .groupby(group_keys, as_index=False)
                    .first()
                )
            elif self.strategy == "max":
                # 确保 proportion 列是数值类型（避免字符串字典序排序）
                df[self.source_proportion1_column] = pd.to_numeric(df[self.source_proportion1_column], errors="coerce")
                df_agg = (
                    df.sort_values(self.source_proportion1_column, kind="mergesort")
                    .groupby(group_keys, as_index=False)
                    .last()
                )
            elif self.strategy == "mean":
                # 数值列取平均，其他列取第一个值
                numeric_cols = [self.source_proportion1_column, self.source_proportion2_column]
                numeric_cols = [c for c in numeric_cols if c in df.columns]

                # 确保数值列是数值类型（CSV 读取时可能为字符串）
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # 先对数值列求平均
                df_numeric = df.groupby(group_keys, as_index=False)[numeric_cols].mean()

                # 获取非数值列（取第一个值）
                other_cols = [c for c in df.columns if c not in group_keys and c not in numeric_cols]
                if other_cols:
                    df_other = df.groupby(group_keys, as_index=False)[other_cols].first()
                    df_agg = df_numeric.merge(df_other, on=group_keys)
                else:
                    df_agg = df_numeric
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

            result[period] = df_agg
            aggregated_count = len(df_agg)
            reduction = (1 - aggregated_count / original_count) * 100 if original_count > 0 else 0

            self.logger.info(
                f"{period}: {original_count:,} -> {aggregated_count:,} records " f"({reduction:.1f}% reduced)"
            )

        self.logger.info(f"Aggregation completed: {len(result)} periods processed")
        return result


# -------------------- Filter --------------------
class ShareholdingFilter:
    def __init__(
        self,
        logger: logging.Logger,
        investor_lb: int,
        stock_lb: int,
        source_shareholder_id_column: str = "ShareHolderID",
        source_stock_symbol_column: str = "Symbol",
    ):
        self.investor_lb: int = investor_lb
        self.stock_lb: int = stock_lb
        self.source_shareholder_id_column: str = source_shareholder_id_column
        self.source_stock_symbol_column: str = source_stock_symbol_column
        self.logger: logging.Logger = logger

    @log_exceptions_inclass("logger")
    def filter(self, period_data: Dict[str, pd.DataFrame]):
        if not period_data:
            self.logger.warning("No data to filter")
            return {}, {}, {}

        self.logger.info(
            f"Starting data filtering with thresholds: investors≥{self.investor_lb}, stocks≥{self.stock_lb}"
        )

        p_filtered, p_investors, p_stocks = {}, {}, {}
        total_original_records = sum(len(df) for df in period_data.values())
        total_filtered_records = 0

        for period, df in tqdm(period_data.items(), desc="Filtering periods", unit="period"):
            if df.empty:
                self.logger.warning(f"Period {period} is empty, skipping")
                continue

            self.logger.debug(f"Filtering {period}: {len(df):,} initial records")

            # 计算投资者和股票的持股数量
            inv_count = df.groupby(self.source_shareholder_id_column)[self.source_stock_symbol_column].count()
            stk_count = df.groupby(self.source_stock_symbol_column)[self.source_shareholder_id_column].count()

            # 筛选有效的投资者和股票
            valid_inv = inv_count[inv_count >= self.investor_lb].index
            valid_stk = stk_count[stk_count >= self.stock_lb].index

            # 应用过滤条件
            df_filtered = df[
                df[self.source_shareholder_id_column].isin(valid_inv)
                & df[self.source_stock_symbol_column].isin(valid_stk)
            ]

            if df_filtered.empty:
                self.logger.warning(f"Period {period} has no data after filtering")
                continue

            p_filtered[period] = df_filtered
            p_investors[period] = sorted(df_filtered[self.source_shareholder_id_column].unique())
            p_stocks[period] = sorted(df_filtered[self.source_stock_symbol_column].unique())

            total_filtered_records += len(df_filtered)
            retention_rate = len(df_filtered) / len(df) * 100

            self.logger.info(
                f"{period}: {len(p_investors[period]):,} investors, {len(p_stocks[period]):,} stocks, "
                f"{len(df_filtered):,} records ({retention_rate:.1f}% retained)"
            )

        # 输出过滤统计摘要
        overall_retention = total_filtered_records / total_original_records * 100 if total_original_records > 0 else 0
        self.logger.info(
            f"Filtering completed: {len(p_filtered)} periods, {total_filtered_records:,}/{total_original_records:,} "
            f"records retained ({overall_retention:.1f}%)"
        )

        return p_filtered, p_investors, p_stocks


# -------------------- Portfolio Generator --------------------
class PortfolioGenerator:
    def __init__(
        self,
        logger: logging.Logger,
        include_proportion: bool = False,
        source_shareholder_id_column: str = "ShareHolderID",
        source_stock_symbol_column: str = "Symbol",
        source_proportion1_column: str = "HoldProportion",
        source_proportion2_column: str = "HoldProportion1",
    ):
        self.include_proportion: bool = include_proportion
        self.source_shareholder_id_column: str = source_shareholder_id_column
        self.source_stock_symbol_column: str = source_stock_symbol_column
        self.source_proportion1_column: str = source_proportion1_column
        self.source_proportion2_column: str = source_proportion2_column
        self.logger: logging.Logger = logger

    def _portfolio_row(self, group: pd.DataFrame) -> pd.Series:
        """Sort one investor's holdings once (Proportion1 desc, stable) and return
        the row-aligned stock / proportion lists. Replaces three redundant sort
        passes that each re-sorted identical data."""
        s = group.sort_values(self.source_proportion1_column, ascending=False, kind="mergesort")
        return pd.Series(
            {
                "Portfolio": s[self.source_stock_symbol_column].tolist(),
                "Proportion1": s[self.source_proportion1_column].tolist(),
                "Proportion2": s[self.source_proportion2_column].tolist(),
            }
        )

    @log_exceptions_inclass("logger")
    def generate(self, period_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not period_data:
            self.logger.warning("No data available for portfolio generation")
            return {}

        self.logger.info(f"Starting portfolio generation (include_proportion={self.include_proportion})")

        result = {}
        total_portfolios = 0

        for period, df in tqdm(period_data.items(), desc="Generating portfolios", unit="period"):
            if df.empty:
                self.logger.warning(f"Period {period}: No data, skipping portfolio generation")
                continue

            self.logger.debug(f"Processing {period}: {len(df):,} records")

            if self.include_proportion:
                # 创建包含比例信息的投资组合：每个投资者只排序一次，同时取出三列
                portfolio = (
                    df.groupby(self.source_shareholder_id_column)
                    .apply(self._portfolio_row, include_groups=False)
                    .reset_index()
                )
            else:
                # 仅创建股票列表（按比例降序排列）
                portfolio = (
                    df.groupby(self.source_shareholder_id_column)
                    .apply(
                        lambda x: x.sort_values(self.source_proportion1_column, ascending=False, kind="mergesort")[
                            self.source_stock_symbol_column
                        ].tolist(),
                        include_groups=False,
                    )
                    .reset_index(name="Portfolio")
                )

            result[period] = portfolio
            portfolio_count = len(portfolio)
            total_portfolios += portfolio_count

            # 计算投资组合统计信息
            avg_portfolio_size = portfolio["Portfolio"].apply(len).mean()
            self.logger.info(
                f"{period}: {portfolio_count:,} portfolios generated, " f"avg size: {avg_portfolio_size:.1f} stocks"
            )

        self.logger.info(
            f"Portfolio generation completed: {len(result)} periods, {total_portfolios:,} total portfolios"
        )
        return result


# -------------------- Saver --------------------
class PortfolioSaver:
    def __init__(
        self,
        logger: logging.Logger,
        output_folder: str,
        format: str,
        source_shareholder_id_column: str = "ShareHolderID",
        source_stock_symbol_column: str = "Symbol",
        include_proportion: bool = False,
        id_key: str = "InvestorID",
        portfolio_key: str = "Portfolio",
        proportion1_key: str = "Proportion1",
        proportion2_key: str = "Proportion2",
    ):
        self.output_folder = Path(output_folder)
        self.encoder = PortfolioDataEncoder(
            format=format,
            id_key=id_key,
            portfolio_key=portfolio_key,
            proportion1_key=proportion1_key,
            proportion2_key=proportion2_key,
        )
        self.source_shareholder_id_column = source_shareholder_id_column
        self.source_stock_symbol_column = source_stock_symbol_column
        self.include_proportion = include_proportion
        self.logger = logger
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @log_exceptions_inclass("logger")
    def save(self, period_portfolio: Dict[str, pd.DataFrame]):
        if not period_portfolio:
            self.logger.warning("No portfolios to save")
            return

        self.logger.info(f"Starting portfolio saving: {len(period_portfolio)} periods to {self.output_folder}")

        saved_count = 0
        total_portfolios = 0

        for period, df in tqdm(period_portfolio.items(), desc="Saving portfolios", unit="period"):
            if df.empty:
                self.logger.warning(f"Period {period}: No portfolios to save")
                continue

            output_path = self.output_folder / period

            # 重命名列以匹配输出格式
            df_renamed = df.rename(
                columns={
                    self.source_shareholder_id_column: self.encoder.id_key,
                    "Portfolio": self.encoder.portfolio_key,
                }
            )

            if self.include_proportion:
                df_renamed = df_renamed.rename(
                    columns={"Proportion1": self.encoder.proportion1_key, "Proportion2": self.encoder.proportion2_key}
                )

            # 执行编码和保存
            self.encoder.encode(df_renamed, output_path, self.include_proportion)

            portfolio_count = len(df)
            total_portfolios += portfolio_count
            saved_count += 1

            self.logger.debug(f"{period}: {portfolio_count:,} portfolios saved to {output_path}")

        self.logger.info(
            f"Portfolio saving completed: {saved_count}/{len(period_portfolio)} periods, {total_portfolios:,} total portfolios"
        )

    @log_exceptions_inclass("logger")
    def save_intermediate(
        self,
        period_data: Dict[str, pd.DataFrame],
        investors: Dict[str, List[str]],
        stocks: Dict[str, List[str]],
        intermediate_folder: str,
        stats: bool = True,
        plot: bool = False,
    ):
        if not period_data:
            self.logger.warning("No intermediate data to save")
            return

        inter = Path(intermediate_folder)
        inter.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving intermediate data to {inter}")

        summary = {}

        for period, df in tqdm(period_data.items(), desc="Saving intermediate", unit="period"):
            period_dir: Path = inter / period
            period_dir.mkdir(parents=True, exist_ok=True)

            # 保存数据文件
            csv_path = period_dir / f"{period}.csv"
            df.to_csv(csv_path, index=False)

            # 保存投资者和股票列表
            with open(period_dir / "valid_investors.json", "w") as f:
                json.dump(investors[period], f, indent=2)

            with open(period_dir / "valid_stocks.json", "w") as f:
                json.dump(stocks[period], f, indent=2)

            # 收集统计信息
            summary[period] = {
                "valid_investors": len(investors[period]),
                "valid_stocks": len(stocks[period]),
                "samples": len(df),
            }

            # 计算序列长度和频率（stats 和 plot 共用）
            if stats or plot:
                portfolio_lengths = df.groupby(self.source_shareholder_id_column).size().values
                stock_frequencies = df.groupby(self.source_stock_symbol_column).size().values

            if stats:
                # 保存详细统计数据
                statistics = {
                    "portfolio_length": {
                        "mean": float(np.mean(portfolio_lengths)),
                        "median": float(np.median(portfolio_lengths)),
                        "std": float(np.std(portfolio_lengths)),
                        "min": int(np.min(portfolio_lengths)),
                        "max": int(np.max(portfolio_lengths)),
                        "quantiles": {
                            "25%": float(np.percentile(portfolio_lengths, 25)),
                            "50%": float(np.percentile(portfolio_lengths, 50)),
                            "75%": float(np.percentile(portfolio_lengths, 75)),
                            "90%": float(np.percentile(portfolio_lengths, 90)),
                            "95%": float(np.percentile(portfolio_lengths, 95)),
                            "99%": float(np.percentile(portfolio_lengths, 99)),
                        },
                    },
                    "stock_frequency": {
                        "mean": float(np.mean(stock_frequencies)),
                        "median": float(np.median(stock_frequencies)),
                        "std": float(np.std(stock_frequencies)),
                        "min": int(np.min(stock_frequencies)),
                        "max": int(np.max(stock_frequencies)),
                        "quantiles": {
                            "25%": float(np.percentile(stock_frequencies, 25)),
                            "50%": float(np.percentile(stock_frequencies, 50)),
                            "75%": float(np.percentile(stock_frequencies, 75)),
                            "90%": float(np.percentile(stock_frequencies, 90)),
                            "95%": float(np.percentile(stock_frequencies, 95)),
                            "99%": float(np.percentile(stock_frequencies, 99)),
                        },
                    },
                }
                with open(period_dir / "statistics.json", "w") as f:
                    json.dump(statistics, f, indent=2)

                # Per-stock institutional coverage statistics
                stock_coverage = df.groupby(self.source_stock_symbol_column)[
                    self.source_shareholder_id_column
                ].nunique()
                stock_coverage.name = "coverage_count"
                stock_coverage.to_csv(period_dir / "stock_coverage.csv")
                self.logger.debug(f"Saved stock_coverage.csv: {len(stock_coverage)} stocks")

            if plot:
                # 创建可视化
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # 1. 序列长度分布直方图
                axes[0, 0].hist(portfolio_lengths, bins=50, edgecolor="black", alpha=0.7)
                axes[0, 0].set_xlabel("Portfolio Length (Number of Stocks held by an Investor)", fontsize=12)
                axes[0, 0].set_ylabel("Frequency", fontsize=12)
                axes[0, 0].set_title(
                    f"Portfolio Sequence Length Distribution - {period}", fontsize=14, fontweight="bold"
                )
                axes[0, 0].grid(True, alpha=0.3)

                # 2. 序列长度分布箱线图
                axes[0, 1].boxplot(portfolio_lengths, vert=True)
                axes[0, 1].set_ylabel("Portfolio Length (Number of Stocks held by an Investor)", fontsize=12)
                axes[0, 1].set_title(f"Portfolio Sequence Length Box Plot - {period}", fontsize=14, fontweight="bold")
                axes[0, 1].grid(True, alpha=0.3)

                # 3. 股票频率分布直方图
                axes[1, 0].hist(stock_frequencies, bins=50, edgecolor="black", alpha=0.7, color="orange")
                axes[1, 0].set_xlabel("Stock Frequency (Number of Investors holding a certain Stock)", fontsize=12)
                axes[1, 0].set_ylabel("Frequency", fontsize=12)
                axes[1, 0].set_title(f"Token Frequency Distribution - {period}", fontsize=14, fontweight="bold")
                axes[1, 0].grid(True, alpha=0.3)

                # 4. 股票频率分布箱线图
                axes[1, 1].boxplot(stock_frequencies, vert=True)
                axes[1, 1].set_ylabel("Stock Frequency (Number of Investors holding a certain Stock)", fontsize=12)
                axes[1, 1].set_title(f"Token Frequency Box Plot - {period}", fontsize=14, fontweight="bold")
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(period_dir / "distribution_analysis.png", dpi=300, bbox_inches="tight")
                plt.close()

        # 保存汇总统计
        self.logger.info(f"Saving summary to {inter / 'summary.json'}")
        with open(inter / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info("Intermediate data summary saved.")


# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ShareHolding Data Processing")
    parser.add_argument(
        "--input_folder", "-i", type=str, required=True, help="Folder containing raw ShareHolding CSV data. Required."
    )
    parser.add_argument(
        "--output_folder", "-o", type=str, required=True, help="Output folder for processed portfolio data. Required."
    )
    parser.add_argument(
        "--intermediate_folder",
        "-inter",
        type=str,
        required=False,
        default=None,
        help="Output folder for intermediate processed data. Optional.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "csv", "binary"],
        default="csv",
        help="Output format for portfolio data. Choices: json, csv, binary. Default: csv",
    )
    parser.add_argument(
        "--frequency",
        "-freq",
        type=str,
        choices=["Y", "HY", "Q", "M", "N"],
        default="Q",
        help="Frequency for grouping data. Choices: Y (yearly), HY (half-year), Q (quarterly), M (monthly), N (no grouping). Default: Q",
    )
    parser.add_argument(
        "--date_format",
        type=str,
        default=None,
        help="Date format for parsing EndDate column (ignored when frequency=N). Default: None (auto-detect)",
    )
    parser.add_argument(
        "--aggregation",
        "-agg",
        type=str,
        choices=["all", "last", "first", "max", "mean"],
        default="all",
        help="Aggregation strategy for duplicate (investor, stock) pairs within a period. "
        "Choices: all (keep all), last (latest date), first (earliest date), max (highest proportion), mean (average). "
        "Note: 'last' and 'first' require date column (incompatible with frequency=N). Default: all",
    )
    parser.add_argument(
        "--include_proportion",
        "-p",
        type=utils.str2bool,
        default=False,
        help="Include proportion in portfolio data (True/False). Default: False",
    )
    parser.add_argument(
        "--investor_lowerbound",
        "-il",
        type=int,
        default=10,
        help="Filter out investors holding fewer stocks than this threshold. Default: 10",
    )
    parser.add_argument(
        "--stock_lowerbound",
        "-sl",
        type=int,
        default=10,
        help="Filter out stocks held fewer times than this threshold. Default: 10",
    )
    parser.add_argument(
        "--source_shareholder_id_column",
        type=str,
        default="ShareHolderID",
        help="Column name for investor ID in source data. Default: `ShareHolderID`",
    )
    parser.add_argument(
        "--source_stock_symbol_column",
        type=str,
        default="Symbol",
        help="Column name for stock symbol in source data. Default: `Symbol`",
    )
    parser.add_argument(
        "--source_end_date_column",
        type=str,
        default="EndDate",
        help="Column name for end date in source data. Default: `EndDate`",
    )
    parser.add_argument(
        "--source_proportion1_column",
        type=str,
        default="HoldProportion",
        help="Column name for holding proportion1 in source data. Calculated as: Institutional investor shareholding proportion = Total shares held by institutional investors / Total shares. Default: `HoldProportion`",
    )
    parser.add_argument(
        "--source_proportion2_column",
        type=str,
        default="HoldProportion1",
        help="Column name for holding proportion2 in source data. Calculated as: Institutional investor shareholding proportion = Total shares held by institutional investors / Tradable A-shares. Default: `HoldProportion1`",
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="InvestorID",
        help="Key name for investor ID in output data. Default: `InvestorID`",
    )
    parser.add_argument(
        "--portfolio_key",
        type=str,
        default="Portfolio",
        help="Key name for portfolio in output data. Default: `Portfolio`",
    )
    parser.add_argument(
        "--proportion1_key",
        type=str,
        default="Proportion1",
        help="Key name for proportion1 (relative to total shares) in output data. Default: `Proportion1`",
    )
    parser.add_argument(
        "--proportion2_key",
        type=str,
        default="Proportion2",
        help="Key name for proportion2 (relative to tradable A-shares) in output data. Default: `Proportion2`",
    )
    parser.add_argument(
        "--stats",
        type=utils.str2bool,
        default=True,
        help="Save per-period statistics (statistics.json, stock_coverage.csv) when intermediate_folder is set. Default: True",
    )
    parser.add_argument(
        "--plot",
        type=utils.str2bool,
        default=False,
        help="Generate distribution analysis visualizations (distribution_analysis.png). Default: False",
    )
    parser.add_argument(
        "--verbose", "-v", type=utils.str2bool, default=False, help="Enable verbose logging to console. Default: False"
    )
    parser.add_argument("--log", "-l", type=str, default=None, help="Log file path. Default: None (no log file)")
    return parser.parse_args()


def main():
    # CLI-run noise suppression (scoped to script execution, not import).
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)

    args = parse_args()

    # 初始化日志系统
    logger = (
        Logger_Preparer()
        .set_config(
            LoggerConfig(
                log_name="ShareHolding DataProcessor",
                log_file=args.log,
                console_stream="tqdm",
                console_level="DEBUG" if args.verbose else "INFO",
                file_level="DEBUG",
                enable_colors=True,
                color_target="format",
            )
        )
        .prepare()
    )

    logger.debug(f"Command: {' '.join(sys.argv)}")

    # 1. 数据加载阶段
    loader = ShareholdingLoader(
        logger=logger,
        data_folder=args.input_folder,
        source_shareholder_id_column=args.source_shareholder_id_column,
        source_stock_symbol_column=args.source_stock_symbol_column,
        source_end_date_column=args.source_end_date_column,
        date_format=args.date_format,
        frequency=args.frequency,
    )
    raw_data = loader.load_all()

    # 2. 数据聚合阶段
    aggregator = ShareholdingAggregator(
        logger=logger,
        strategy=args.aggregation,
        source_shareholder_id_column=args.source_shareholder_id_column,
        source_stock_symbol_column=args.source_stock_symbol_column,
        source_end_date_column=args.source_end_date_column,
        source_proportion1_column=args.source_proportion1_column,
        source_proportion2_column=args.source_proportion2_column,
    )
    aggregated_data = aggregator.aggregate(raw_data)

    # 3. 数据过滤阶段
    filter_processor = ShareholdingFilter(
        logger=logger,
        investor_lb=args.investor_lowerbound,
        stock_lb=args.stock_lowerbound,
        source_shareholder_id_column=args.source_shareholder_id_column,
        source_stock_symbol_column=args.source_stock_symbol_column,
    )
    filtered_data, investors, stocks = filter_processor.filter(aggregated_data)

    # 4. 投资组合生成阶段
    portfolio_generator = PortfolioGenerator(
        logger=logger,
        include_proportion=args.include_proportion,
        source_shareholder_id_column=args.source_shareholder_id_column,
        source_stock_symbol_column=args.source_stock_symbol_column,
        source_proportion1_column=args.source_proportion1_column,
        source_proportion2_column=args.source_proportion2_column,
    )
    portfolios = portfolio_generator.generate(filtered_data)

    # 5. 数据保存阶段
    saver = PortfolioSaver(
        logger=logger,
        output_folder=args.output_folder,
        format=args.output_format,
        source_shareholder_id_column=args.source_shareholder_id_column,
        source_stock_symbol_column=args.source_stock_symbol_column,
        include_proportion=args.include_proportion,
        id_key=args.id_key,
        portfolio_key=args.portfolio_key,
        proportion1_key=args.proportion1_key,
        proportion2_key=args.proportion2_key,
    )

    # 保存中间数据（如果指定）
    if args.intermediate_folder:
        saver.save_intermediate(
            period_data=filtered_data,
            investors=investors,
            stocks=stocks,
            intermediate_folder=args.intermediate_folder,
            stats=args.stats,
            plot=args.plot,
        )

    # 保存最终投资组合
    saver.save(portfolios)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
