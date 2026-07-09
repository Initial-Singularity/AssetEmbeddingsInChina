import os
import sys
import time
import shutil
import logging
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Literal, Optional, Protocol, Self

import pandas as pd
from tqdm import tqdm

from asset_embeddings import utils
from asset_embeddings.scripts.data.sharehold import PortfolioDataEncoder
from asset_embeddings.configs import Config, ConfigField, LoggerConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.utils.log_utils import log_exceptions_inclass


class DistributionConfig(Config):
    """Configuration for dataset distribution with validation and CLI integration.

    Inherits from Config base class to leverage the advanced configuration framework
    including validation, chainable loading, and argparse integration.
    """

    # Required fields
    input_path: str = ConfigField(required=True, doc="Path to input dataset (file or directory)")
    output_path: str = ConfigField(required=True, doc="Path for output splits")

    # Data format options
    file_format: Literal["csv", "json"] = ConfigField(default="csv", doc="Input file format")
    use_portfolio_encoder: bool = ConfigField(default=False, doc="Use portfolio data encoder")
    include_proportion: bool = ConfigField(default=False, doc="Include proportion data (portfolio encoder only)")

    # Portfolio data keys
    id_key: str = ConfigField(default="InvestorID", doc="Investor ID column name")
    portfolio_key: str = ConfigField(default="Portfolio", doc="Portfolio column name")
    proportion1_key: str = ConfigField(default="Proportion1", doc="Proportion1 column name")
    proportion2_key: str = ConfigField(default="Proportion2", doc="Proportion2 column name")

    # Split configuration
    split_ratios: List[float] = ConfigField(
        default=[0.8, 0.2],
        preprocessor=lambda x: [float(r) for r in x] if isinstance(x, (list, tuple)) else x,
        doc="Split ratios (will be normalized to sum to 1.0)",
    )
    dir_names: Optional[List[str]] = ConfigField(
        default=None, doc="Custom directory names for splits (default: split_0, split_1, ...)"
    )

    # Distribution strategy
    keep_file_integrity: bool = ConfigField(default=True, doc="Maintain file boundaries during split")
    strategy: Literal["concat", "streaming"] = ConfigField(
        default="concat", doc="Row-level split strategy: concat (fast, high memory) or streaming (slow, low memory)"
    )
    split_shuffle: bool = ConfigField(default=True, doc="Shuffle data before row-level splitting")
    max_lines_per_file: Optional[int] = ConfigField(
        default=None, doc="Maximum lines per output file (creates multiple files if exceeded)"
    )

    # General options
    seed: int = ConfigField(default=42, doc="Random seed")

    def validate(self) -> Self:
        """Validate configuration after all values are set (chainable)."""
        super().validate()
        # Validate split ratios
        if not self.split_ratios or len(self.split_ratios) == 0:
            raise ValueError("Split ratios cannot be empty")
        if any(r <= 0 for r in self.split_ratios):
            raise ValueError("All split ratios must be positive")

        # Normalize split ratios
        total = sum(self.split_ratios)
        self.split_ratios = [r / total for r in self.split_ratios]

        # Set default directory names if not provided
        if self.dir_names is None:
            self.dir_names = [f"split_{i}" for i in range(len(self.split_ratios))]
        elif len(self.dir_names) != len(self.split_ratios):
            raise ValueError("Number of directory names must match number of split ratios")

        # Validate paths - only check if path exists and is accessible
        if self.input_path and not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        # Validate max lines per file
        if self.max_lines_per_file is not None and self.max_lines_per_file <= 0:
            raise ValueError("max_lines_per_file must be positive if specified")

        return self


class FileProcessor(Protocol):
    """Protocol for file processing operations."""

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load a single file and return DataFrame."""
        ...

    def save_file(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to file."""
        ...


class StandardFileProcessor:
    """Standard file processor for CSV files."""

    def __init__(self, file_format: str = "csv"):
        self.file_format = file_format

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load a file based on format with enhanced error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

        try:
            if self.file_format == "csv":
                df = pd.read_csv(file_path)
            elif self.file_format == "json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

            if df.empty:
                raise ValueError(f"Loaded DataFrame is empty: {file_path}")

            return df
        except pd.errors.EmptyDataError:
            raise ValueError(f"No data found in file: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse {self.file_format} file '{file_path}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load file '{file_path}' as {self.file_format}: {str(e)}")

    def save_file(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to file with enhanced error handling."""
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame to: {file_path}")

        output_dir = os.path.dirname(file_path)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Permission denied creating directory: {output_dir}")
        except Exception as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {str(e)}")

        try:
            if self.file_format == "csv":
                df.to_csv(file_path, index=False)
            elif self.file_format == "json":
                df.to_json(file_path, orient="records")
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

            # Verify file was created and is not empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise OSError(f"Failed to create or file is empty after save: {file_path}")

        except PermissionError:
            raise PermissionError(f"Permission denied writing to file: {file_path}")
        except Exception as e:
            raise OSError(f"Failed to save {self.file_format} file '{file_path}': {str(e)}")


class PortfolioFileProcessor:
    """File processor for portfolio data with encoder."""

    def __init__(self, config: DistributionConfig):
        self.config = config
        self.encoder = PortfolioDataEncoder(
            format=config.file_format,
            id_key=config.id_key,
            portfolio_key=config.portfolio_key,
            proportion1_key=config.proportion1_key,
            proportion2_key=config.proportion2_key,
        )

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load portfolio file using encoder with enhanced error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio file not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Portfolio file is empty: {file_path}")

        try:
            df = self.encoder.decode(file_path, include_proportion=self.config.include_proportion)
            if df.empty:
                raise ValueError(f"Portfolio encoder produced empty DataFrame from: {file_path}")
            return df
        except Exception as e:
            raise ValueError(f"Portfolio encoder failed to decode '{file_path}': {str(e)}")

    def save_file(self, df: pd.DataFrame, file_path: str) -> None:
        """Save portfolio data using encoder with enhanced error handling."""
        if df.empty:
            raise ValueError(f"Cannot save empty portfolio DataFrame to: {file_path}")

        # Validate required columns for portfolio encoder
        required_cols = [self.config.id_key, self.config.portfolio_key]
        if self.config.include_proportion:
            required_cols.extend([self.config.proportion1_key, self.config.proportion2_key])

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Portfolio DataFrame missing required columns {missing_cols} for file: {file_path}")

        output_dir = os.path.dirname(file_path)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {str(e)}")

        try:
            self.encoder.encode(df, file_path, include_proportion=self.config.include_proportion)

            # Verify file was created and is not empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise OSError(f"Portfolio encoder failed to create file or file is empty: {file_path}")

        except Exception as e:
            raise OSError(f"Portfolio encoder failed to save '{file_path}': {str(e)}")


class DistributionStrategy(ABC):
    """Abstract base class for distribution strategies."""

    def __init__(self, config: DistributionConfig, file_processor: FileProcessor, logger: logging.Logger):
        self.config = config
        self.file_processor = file_processor
        self.logger = logger

    @abstractmethod
    def distribute(self, file_list: List[str]) -> None:
        """Distribute files according to the strategy."""
        pass

    def _create_output_directories(self) -> List[str]:
        """Create output directories and return paths with safety checks."""
        output_dirs = [os.path.join(self.config.output_path, name) for name in self.config.dir_names]

        # Safety check: warn if output directories exist and are not empty
        for i, dir_path in enumerate(output_dirs):
            if os.path.exists(dir_path) and os.listdir(dir_path):
                file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                self.logger.warning(
                    f"Output directory already exists with {file_count} files, may be overwritten: {dir_path}"
                )
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Created output directory: {dir_path}")

        self.logger.info(f"Prepared {len(output_dirs)} output directories: {self.config.dir_names}")
        return output_dirs

    def _calculate_target_sizes(self, total_rows: int) -> List[int]:
        """Calculate target sizes for each split, handling rounding issues."""
        sizes = [round(total_rows * ratio) for ratio in self.config.split_ratios]

        # Fix rounding issues to ensure sum equals total_rows
        while sum(sizes) > total_rows:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] -= 1
        while sum(sizes) < total_rows:
            min_idx = sizes.index(min(sizes))
            sizes[min_idx] += 1

        return sizes


class DatasetDistributor:
    """Main dataset distributor class with strategy pattern and error handling."""

    def __init__(self, config: DistributionConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Validate configuration
        try:
            config.validate()
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise

        # Set random seed
        try:
            utils.seed_basic(config.seed)
            self.logger.debug(f"Random seed set to: {config.seed}")
        except Exception as e:
            self.logger.warning(f"Failed to set random seed {config.seed}: {str(e)}")

        # Create main output directory with safety check and proper error handling
        try:
            if os.path.exists(config.output_path):
                existing_items = os.listdir(config.output_path)
                if existing_items:
                    self.logger.warning(
                        f"Main output directory '{config.output_path}' already exists and contains {len(existing_items)} items"
                    )
            os.makedirs(config.output_path, exist_ok=True)
            self.logger.debug(f"Main output directory prepared: {config.output_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied creating main output directory: {config.output_path}")
        except Exception as e:
            raise OSError(f"Failed to create main output directory '{config.output_path}': {str(e)}")

        # Initialize file processor
        if config.use_portfolio_encoder:
            self.file_processor = PortfolioFileProcessor(config)
            self.logger.info(
                f"Using PortfolioFileProcessor with keys: id={config.id_key}, portfolio={config.portfolio_key}"
            )
        else:
            self.file_processor = StandardFileProcessor(config.file_format)
            self.logger.info(f"Using StandardFileProcessor for {config.file_format} files")

        # Initialize strategy
        self.strategy = self._create_strategy()
        self.logger.info(f"Distribution strategy initialized: {type(self.strategy).__name__}")

    @log_exceptions_inclass("logger")
    def _create_strategy(self) -> DistributionStrategy:
        """Create distribution strategy based on configuration."""
        if self.config.keep_file_integrity:
            return FileIntegrityStrategy(self.config, self.file_processor, self.logger)
        elif self.config.strategy == "concat":
            return ConcatStrategy(self.config, self.file_processor, self.logger)
        elif self.config.strategy == "streaming":
            return StreamingStrategy(self.config, self.file_processor, self.logger)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    @log_exceptions_inclass("logger")
    def _discover_files(self) -> List[str]:
        """Discover input files based on input path."""
        input_path = Path(self.config.input_path)
        self.logger.info(f"Discovering files from input path: {input_path}")

        if input_path.is_file():
            file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            self.logger.info(f"Single file detected: {input_path.name} ({file_size:.1f} MB)")
            return [str(input_path)]
        elif input_path.is_dir():
            pattern = f"*.{self.config.file_format}"
            self.logger.debug(f"Searching for files with pattern: {pattern}")
            files = list(input_path.glob(pattern))
            if not files:
                self.logger.error(f"No {self.config.file_format} files found in directory: {input_path}")
                raise ValueError(f"No {self.config.file_format} files found in {input_path}")

            # Log file discovery statistics
            total_size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)  # MB
            self.logger.info(
                f"Discovered {len(files)} {self.config.file_format} files "
                f"(total size: {total_size:.1f} MB, avg: {total_size/len(files):.1f} MB/file)"
            )

            return sorted([str(f) for f in files])
        else:
            self.logger.error(f"Input path does not exist or is not accessible: {input_path}")
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    @log_exceptions_inclass("logger")
    def distribute(self) -> None:
        """Distribute the dataset using the configured strategy."""
        self.logger.info(f"Configuration: {self.config.to_dict()}")

        files = self._discover_files()
        self.logger.info(f"Processing {len(files)} files with {type(self.strategy).__name__}")

        start_time = time.time()
        self.strategy.distribute(files)
        duration = time.time() - start_time

        self.logger.info(f"Dataset distribution completed in {duration:.1f} seconds")


class FileIntegrityStrategy(DistributionStrategy):
    """Strategy that maintains file integrity during distribution."""

    @log_exceptions_inclass("logger")
    def distribute(self, file_list: List[str]) -> None:
        self.logger.info("Starting file integrity strategy (maintains complete files in each split)")

        # Get file sizes
        file_sizes = self._get_file_sizes(file_list)
        if not file_sizes:
            self.logger.error("No valid files found to distribute - all files failed to load")
            raise ValueError("No valid files found to distribute - check input path and file permissions")

        total_rows = sum(size for _, size in file_sizes)
        target_sizes = self._calculate_target_sizes(total_rows)

        # Log distribution planning
        self.logger.info(f"Distribution plan: {total_rows:,} total rows across {len(file_sizes)} files")
        for i, target in enumerate(target_sizes):
            target_pct = target / total_rows * 100
            split_name = self.config.dir_names[i]
            self.logger.info(f"  {split_name}: target {target:,} rows ({target_pct:.1f}%)")

        # Create output directories
        output_dirs = self._create_output_directories()

        # Distribute files
        self._assign_files_to_splits(file_sizes, target_sizes, output_dirs)

    def _get_file_sizes(self, file_list: List[str]) -> List[Tuple[str, int]]:
        """Get file sizes efficiently with detailed progress tracking."""
        self.logger.info(f"Analyzing sizes of {len(file_list)} files...")
        file_sizes = []
        failed_files = 0
        total_rows = 0

        for i, file_path in enumerate(tqdm(file_list, desc="Analyzing files"), 1):
            try:
                df = self.file_processor.load_file(file_path)
                rows = len(df)
                file_sizes.append((file_path, rows))
                total_rows += rows

                if i % 10 == 0 or i == len(file_list):  # Log progress every 10 files or at end
                    self.logger.debug(
                        f"Analyzed {i}/{len(file_list)} files. "
                        f"Current total: {total_rows:,} rows, Failed: {failed_files}"
                    )

            except Exception as e:
                failed_files += 1
                self.logger.warning(f"Failed to load {os.path.basename(file_path)}: {str(e)[:100]}...")
                continue

        self.logger.info(
            f"File analysis complete: {len(file_sizes)}/{len(file_list)} files loaded, "
            f"{total_rows:,} total rows, {failed_files} failed"
        )
        return file_sizes

    def _assign_files_to_splits(
        self, file_sizes: List[Tuple[str, int]], target_sizes: List[int], output_dirs: List[str]
    ) -> None:
        """Assign files to splits while trying to meet target sizes."""
        current_split = 0
        assigned_rows = [0] * len(target_sizes)
        assigned_files = [0] * len(target_sizes)
        failed_files = 0
        total_processed_size = 0  # in MB
        last_logged_split = -1  # Track which split we last logged

        self.logger.info(f"Assigning {len(file_sizes)} files to {len(target_sizes)} splits...")

        for i, (file_path, rows) in enumerate(tqdm(file_sizes, desc="Distributing files"), 1):
            # Find best split for this file
            while (
                current_split < len(target_sizes) - 1
                and assigned_rows[current_split] + rows > target_sizes[current_split]
            ):
                current_split += 1

            # Log when entering a new split
            if current_split != last_logged_split:
                split_name = self.config.dir_names[current_split]
                self.logger.info(f"Distributing to split {current_split} ({split_name}): {output_dirs[current_split]}")
                last_logged_split = current_split

            # Copy/process file to target split
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dirs[current_split], filename)
            split_name = self.config.dir_names[current_split]

            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

                if isinstance(self.file_processor, StandardFileProcessor):
                    shutil.copy2(file_path, output_path)
                    operation = "Copied"
                else:
                    df = self.file_processor.load_file(file_path)
                    self.file_processor.save_file(df, output_path)
                    operation = "Processed"

                assigned_rows[current_split] += rows
                assigned_files[current_split] += 1
                total_processed_size += file_size_mb

                self.logger.debug(
                    f"{operation} {filename} ({rows:,} rows, {file_size_mb:.1f}MB) "
                    f"-> {split_name} [{assigned_files[current_split]}/{len(file_sizes)} in split]"
                )

                # Progress logging every 20 files or at significant milestones
                if i % 20 == 0 or i == len(file_sizes):
                    progress_pct = i / len(file_sizes) * 100
                    self.logger.info(
                        f"Progress: {i}/{len(file_sizes)} files ({progress_pct:.1f}%), "
                        f"processed {total_processed_size:.1f}MB, failed: {failed_files}"
                    )

            except Exception as e:
                failed_files += 1
                self.logger.error(f"Failed to process {filename}: {str(e)[:150]}...")
                continue

        # Log final distribution statistics
        self.logger.info("Distribution summary:")
        total_assigned_rows = sum(assigned_rows)
        total_assigned_files = sum(assigned_files)

        for i, (target, actual_rows, actual_files, dir_path) in enumerate(
            zip(target_sizes, assigned_rows, assigned_files, output_dirs)
        ):
            split_name = os.path.basename(dir_path)
            row_pct = actual_rows / total_assigned_rows * 100 if total_assigned_rows > 0 else 0
            file_pct = actual_files / total_assigned_files * 100 if total_assigned_files > 0 else 0
            target_pct = target / sum(target_sizes) * 100

            self.logger.info(
                f"  {split_name}: {actual_files} files, {actual_rows:,} rows "
                f"({row_pct:.1f}% vs target {target_pct:.1f}%), {file_pct:.1f}% of files"
            )

        if failed_files > 0:
            self.logger.warning(
                f"Distribution completed with {failed_files} failed files out of {len(file_sizes)} total"
            )
        else:
            self.logger.info(f"All {len(file_sizes)} files distributed successfully")


class ConcatStrategy(DistributionStrategy):
    """Strategy that concatenates all data in memory before splitting."""

    @log_exceptions_inclass("logger")
    def distribute(self, file_list: List[str]) -> None:
        self.logger.info("Starting concat strategy (loads all data into memory, then splits at row level)")
        self.logger.warning("High memory usage expected")

        # Load and concatenate all data
        combined_df = self._load_and_combine_files(file_list)
        if combined_df.empty:
            self.logger.error("No data loaded from input files - all files were empty or failed to load")
            raise ValueError("No data loaded from input files - check file contents and formats")

        # Shuffle if requested
        if self.config.split_shuffle:
            self.logger.info(f"Shuffling {len(combined_df):,} rows (seed={self.config.seed})")
            combined_df = combined_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

        # Calculate split sizes
        total_rows = len(combined_df)
        split_sizes = self._calculate_target_sizes(total_rows)
        memory_usage = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        self.logger.info(
            f"Dataset loaded: {total_rows:,} rows, {len(combined_df.columns)} columns, {memory_usage:.1f}MB"
        )
        for i, size in enumerate(split_sizes):
            size_pct = size / total_rows * 100
            split_name = self.config.dir_names[i]
            self.logger.info(f"  {split_name}: {size:,} rows ({size_pct:.1f}%)")

        # Create output directories
        output_dirs = self._create_output_directories()

        # Split and save
        self._split_and_save(combined_df, split_sizes, output_dirs)

    def _load_and_combine_files(self, file_list: List[str]) -> pd.DataFrame:
        """Load and combine all files into a single DataFrame with detailed progress tracking."""
        self.logger.info(f"Loading and combining {len(file_list)} files into memory...")

        dataframes = []
        failed_files = 0
        total_rows = 0
        total_size = 0  # MB

        for i, file_path in enumerate(tqdm(file_list, desc="Loading files"), 1):
            try:
                df = self.file_processor.load_file(file_path)
                if not df.empty:
                    dataframes.append(df)
                    rows = len(df)
                    size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    total_rows += rows
                    total_size += size_mb

                    self.logger.debug(f"Loaded {os.path.basename(file_path)}: {rows:,} rows, {size_mb:.1f}MB")

                    # Progress logging every 10 files or at completion
                    if i % 10 == 0 or i == len(file_list):
                        self.logger.info(
                            f"Loaded {i}/{len(file_list)} files: {total_rows:,} total rows, "
                            f"{total_size:.1f}MB in memory, {failed_files} failed"
                        )
                else:
                    self.logger.warning(f"Empty file skipped: {os.path.basename(file_path)}")

            except Exception as e:
                failed_files += 1
                self.logger.warning(f"Failed to load {os.path.basename(file_path)}: {str(e)[:100]}...")
                continue

        if not dataframes:
            self.logger.error(f"No valid dataframes loaded from {len(file_list)} files")
            raise ValueError("No valid dataframes loaded")

        self.logger.info(f"Concatenating {len(dataframes)} dataframes...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        final_memory = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)

        self.logger.info(
            f"Combination complete: {len(combined_df):,} rows, {len(combined_df.columns)} columns, "
            f"{final_memory:.1f}MB ({len(dataframes)}/{len(file_list)} files loaded)"
        )

        return combined_df

    def _split_and_save(self, df: pd.DataFrame, split_sizes: List[int], output_dirs: List[str]) -> None:
        """Split DataFrame and save to files with comprehensive progress tracking."""
        self.logger.info(f"Splitting dataset into {len(split_sizes)} parts...")

        start_idx = 0
        saved_files = []

        for i, size in enumerate(tqdm(split_sizes, desc="Saving splits")):
            split_df = df.iloc[start_idx : start_idx + size]
            split_dir = output_dirs[i]
            split_name = self.config.dir_names[i]

            # Log when starting to save to a new split
            self.logger.info(f"Saving to split {i} ({split_name}): {split_dir}")

            if self.config.max_lines_per_file is None:
                # Save as single file
                output_path = os.path.join(split_dir, f"split_{i}.{self.config.file_format}")
                self.file_processor.save_file(split_df, output_path)
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                saved_files.append((output_path, len(split_df), file_size))

                self.logger.debug(
                    f"Saved {os.path.basename(output_path)} ({len(split_df):,} rows, {file_size:.1f}MB) "
                    f"-> {split_name}"
                )
            else:
                # Save as multiple chunks
                chunk_files = self._save_chunked(split_df, split_dir, f"split_{i}", split_name)
                saved_files.extend(chunk_files)

                total_chunk_size = sum(s for _, _, s in chunk_files)
                self.logger.info(
                    f"Saved {len(chunk_files)} chunks ({len(split_df):,} rows, {total_chunk_size:.1f}MB) "
                    f"-> {split_name}"
                )

            start_idx += size

        # Final summary
        total_saved_rows = sum(rows for _, rows, _ in saved_files)
        total_saved_size = sum(s for _, _, s in saved_files)

        self.logger.info(
            f"All splits saved: {len(saved_files)} files, {total_saved_rows:,} rows, " f"{total_saved_size:.1f}MB total"
        )

    def _save_chunked(
        self, df: pd.DataFrame, output_dir: str, base_name: str, split_name: str
    ) -> List[Tuple[str, int, float]]:
        """Save DataFrame in chunks and return file information."""
        max_lines = self.config.max_lines_per_file
        chunk_files = []

        self.logger.debug(f"Saving {len(df):,} rows in chunks of {max_lines:,} lines")

        for chunk_idx in range(0, len(df), max_lines):
            chunk = df.iloc[chunk_idx : chunk_idx + max_lines]
            chunk_name = f"{base_name}_part{chunk_idx // max_lines}.{self.config.file_format}"
            output_path = os.path.join(output_dir, chunk_name)

            self.file_processor.save_file(chunk, output_path)
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            chunk_files.append((output_path, len(chunk), file_size))

            self.logger.debug(f"Saved {chunk_name} ({len(chunk):,} rows, {file_size:.1f}MB) -> {split_name}")

        return chunk_files


class StreamingStrategy(DistributionStrategy):
    """Memory-efficient streaming strategy for large datasets."""

    @log_exceptions_inclass("logger")
    def distribute(self, file_list: List[str]) -> None:
        self.logger.info("Starting streaming strategy (row-by-row processing, minimal memory usage)")
        self.logger.info("Note: slower but suitable for very large datasets")

        # Calculate total size and split targets
        total_rows = self._calculate_total_rows(file_list)
        if total_rows == 0:
            self.logger.error("No rows found in input files during pre-scan - all files appear to be empty")
            raise ValueError("No rows found in input files - check file contents and accessibility")

        split_sizes = self._calculate_target_sizes(total_rows)

        self.logger.info(f"Streaming distribution: {total_rows:,} total rows from {len(file_list)} files")
        for i, size in enumerate(split_sizes):
            size_pct = size / total_rows * 100
            split_name = self.config.dir_names[i]
            self.logger.info(f"  {split_name}: target {size:,} rows ({size_pct:.1f}%)")

        # Create output directories
        output_dirs = self._create_output_directories()

        # Initialize streaming state
        streaming_state = StreamingState(split_sizes, output_dirs, self.config, self.logger)

        # Process files in streaming fashion
        self._process_files_streaming(file_list, streaming_state)

        # Flush any remaining data and get final statistics
        streaming_state.flush_all(self.file_processor)

        # Log final streaming results
        self.logger.info("Distribution summary:")
        for i, (target, actual) in enumerate(zip(split_sizes, streaming_state.row_counts)):
            actual_pct = actual / sum(streaming_state.row_counts) * 100 if sum(streaming_state.row_counts) > 0 else 0
            target_pct = target / total_rows * 100
            split_name = self.config.dir_names[i]
            self.logger.info(f"  {split_name}: {actual:,} rows ({actual_pct:.1f}% vs target {target_pct:.1f}%)")

    def _calculate_total_rows(self, file_list: List[str]) -> int:
        """Calculate total number of rows across all files with detailed tracking."""
        self.logger.info(f"Pre-scanning {len(file_list)} files to calculate total size...")
        total = 0
        failed_files = 0

        for i, file_path in enumerate(tqdm(file_list, desc="Calculating sizes"), 1):
            try:
                df = self.file_processor.load_file(file_path)
                rows = len(df)
                total += rows

                self.logger.debug(f"Scanned {os.path.basename(file_path)}: {rows:,} rows")

                # Progress logging every 20 files or at completion
                if i % 20 == 0 or i == len(file_list):
                    self.logger.info(f"Scanned {i}/{len(file_list)} files: {total:,} total rows, {failed_files} failed")

            except Exception as e:
                failed_files += 1
                self.logger.warning(f"Failed to read {os.path.basename(file_path)}: {str(e)[:100]}...")

        self.logger.info(
            f"Pre-scan complete: {total:,} total rows from {len(file_list) - failed_files}/{len(file_list)} files"
        )
        return total

    def _process_files_streaming(self, file_list: List[str], state: "StreamingState") -> None:
        """Process files in streaming fashion with detailed progress tracking."""
        self.logger.info("Starting row-by-row streaming...")
        failed_files = 0
        processed_rows = 0

        for i, file_path in enumerate(tqdm(file_list, desc="Streaming files"), 1):
            try:
                df = self.file_processor.load_file(file_path)
                file_rows = len(df)

                self.logger.debug(f"Streaming {os.path.basename(file_path)}: {file_rows:,} rows")

                for row_idx, (_, row) in enumerate(df.iterrows()):
                    if not state.add_row(row, self.file_processor):
                        # All splits are full, exit early
                        self.logger.info(
                            f"All splits full. Stopped at file {i}/{len(file_list)}, row {row_idx + 1}/{file_rows}"
                        )
                        return
                    processed_rows += 1

                    # Log progress every 10K rows
                    if processed_rows % 10000 == 0:
                        self.logger.debug(f"Streamed {processed_rows:,} rows across splits")

                # File completion logging
                if i % 5 == 0 or i == len(file_list):  # Every 5 files or at completion
                    self.logger.info(
                        f"Streamed {i}/{len(file_list)} files: {processed_rows:,} rows processed, {failed_files} failed"
                    )

            except Exception as e:
                failed_files += 1
                self.logger.warning(f"Failed to process {os.path.basename(file_path)}: {str(e)[:100]}...")
                continue

        self.logger.info(
            f"Streaming complete: {processed_rows:,} rows processed from {len(file_list) - failed_files}/{len(file_list)} files"
        )


class StreamingState:
    """Manages state for streaming distribution strategy."""

    def __init__(
        self, target_sizes: List[int], output_dirs: List[str], config: DistributionConfig, logger: logging.Logger
    ):
        self.target_sizes = target_sizes
        self.output_dirs = output_dirs
        self.config = config
        self.logger = logger

        self.current_split = 0
        self.last_logged_split = -1
        self.row_counts = [0] * len(target_sizes)
        self.buffers = [[] for _ in target_sizes]
        self.file_counts = [0] * len(target_sizes)

    def add_row(self, row: pd.Series, file_processor: FileProcessor) -> bool:
        """Add a row to the current split. Returns False if all splits are full."""
        # Find next available split
        while (
            self.current_split < len(self.target_sizes)
            and self.row_counts[self.current_split] >= self.target_sizes[self.current_split]
        ):
            self.current_split += 1

        if self.current_split >= len(self.target_sizes):
            return False  # All splits are full

        # Log when entering a new split
        if self.current_split != self.last_logged_split:
            split_name = self.config.dir_names[self.current_split]
            self.logger.info(
                f"Streaming to split {self.current_split} ({split_name}): {self.output_dirs[self.current_split]}"
            )
            self.last_logged_split = self.current_split

        # Add row to current split
        self.buffers[self.current_split].append(row)
        self.row_counts[self.current_split] += 1

        # Flush buffer if it's full
        if self.config.max_lines_per_file and len(self.buffers[self.current_split]) >= self.config.max_lines_per_file:
            self._flush_buffer(self.current_split, file_processor)

        return True

    def _flush_buffer(self, split_idx: int, file_processor: FileProcessor) -> None:
        """Flush buffer for a specific split."""
        if not self.buffers[split_idx]:
            return

        df_chunk = pd.DataFrame(self.buffers[split_idx])
        filename = f"split_{split_idx}_part{self.file_counts[split_idx]}.{self.config.file_format}"
        output_path = os.path.join(self.output_dirs[split_idx], filename)

        file_processor.save_file(df_chunk, output_path)

        self.file_counts[split_idx] += 1
        self.buffers[split_idx] = []

    def flush_all(self, file_processor: FileProcessor) -> None:
        """Flush all remaining buffers."""
        for i in range(len(self.buffers)):
            if self.buffers[i]:
                if self.config.max_lines_per_file:
                    self._flush_buffer(i, file_processor)
                else:
                    # Create single file for split
                    df_chunk = pd.DataFrame(self.buffers[i])
                    filename = f"split_{i}.{self.config.file_format}"
                    output_path = os.path.join(self.output_dirs[i], filename)
                    file_processor.save_file(df_chunk, output_path)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with improved organization."""
    parser = argparse.ArgumentParser(
        description="Distribute dataset into multiple parts with configurable strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - split CSV files 70/20/10
  python -m asset_embeddings.scripts.data.distributor -i data/input -o data/output

  # Use streaming for large datasets
  python -m asset_embeddings.scripts.data.distributor -i data/input -o data/output --strategy streaming

  # Portfolio data with custom splits
  python -m asset_embeddings.scripts.data.distributor -i data/portfolio -o data/splits \
    --use_portfolio_encoder true --split_ratios 0.6 0.2 0.2
""",
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input_path", "-i", type=str, required=True, help="Path to input dataset (file or directory)"
    )
    required.add_argument("--output_path", "-o", type=str, required=True, help="Path for output splits")

    # Data format arguments
    data_group = parser.add_argument_group("data format options")
    data_group.add_argument(
        "--file_format", "-f", choices=["csv", "json"], default="csv", help="Input file format (default: csv)"
    )
    data_group.add_argument(
        "--use_portfolio_encoder",
        type=utils.str2bool,
        default=False,
        help="Use portfolio data encoder (default: false)",
    )
    data_group.add_argument(
        "--include_proportion",
        type=utils.str2bool,
        default=False,
        help="Include proportion data (portfolio encoder only, default: false)",
    )

    # Portfolio data keys
    portfolio_group = parser.add_argument_group("portfolio data keys")
    portfolio_group.add_argument("--id_key", default="InvestorID", help="Investor ID column name")
    portfolio_group.add_argument("--portfolio_key", default="Portfolio", help="Portfolio column name")
    portfolio_group.add_argument("--proportion1_key", default="Proportion1", help="Proportion1 column name")
    portfolio_group.add_argument("--proportion2_key", default="Proportion2", help="Proportion2 column name")

    # Split configuration
    split_group = parser.add_argument_group("split configuration")
    split_group.add_argument(
        "--split_ratios", "-r", type=float, nargs="+", default=[0.8, 0.2], help="Split ratios (default: 0.8 0.2)"
    )
    split_group.add_argument(
        "--dir_names",
        nargs="+",
        default=None,
        help="Custom directory names for splits (default: split_0, split_1, ...)",
    )

    # Distribution strategy
    strategy_group = parser.add_argument_group("distribution strategy")
    strategy_group.add_argument(
        "--keep_file_integrity",
        type=utils.str2bool,
        default=True,
        help="Maintain file boundaries during split (default: true)",
    )
    strategy_group.add_argument(
        "--strategy",
        choices=["concat", "streaming"],
        default="concat",
        help="Row-level split strategy: concat (fast, high memory) or streaming (slow, low memory)",
    )
    strategy_group.add_argument(
        "--split_shuffle",
        type=utils.str2bool,
        default=True,
        help="Shuffle data before row-level splitting (default: true)",
    )
    strategy_group.add_argument(
        "--max_lines_per_file",
        type=int,
        default=None,
        help="Maximum lines per output file (creates multiple files if exceeded). If not set, saves as single file per split. (default: None)",
    )

    # General options
    general_group = parser.add_argument_group("general options")
    general_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    general_group.add_argument("--verbose", "-v", type=utils.str2bool, default=False, help="Verbose logging")
    general_group.add_argument("--log", "-l", help="Log file path (optional)")

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configuration using the Config framework's from_args method
    config = DistributionConfig().from_args(args)
    config.validate()

    # Setup logger
    logger = (
        Logger_Preparer()
        .set_config(
            LoggerConfig(
                log_name="Dataset Distributor",
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

    logger.debug(f"Commands: {' '.join(sys.argv)}")

    # Create and run distributor
    distributor = DatasetDistributor(config, logger)
    distributor.distribute()
