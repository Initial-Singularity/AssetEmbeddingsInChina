"""Relative-valuation benchmark: ridge fit of the valuation residual on the period's embedding, one row per period into ``eval_rv_results``.

Run as ``python -m asset_embeddings.scripts.eval.rv -c <config> -r <db> -e <experiment>``.
"""

import argparse
import textwrap

from asset_embeddings.evaluators.rv import RVBenchmarkEvaluator
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Relative Valuations Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model_type": "AssetBERT",
            "embedding_path": [
                "model/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
                "model/finetune/AssetBERT/d4/2023Q3/AssetBERT_d4_2023Q3_best_embedding.csv",
                "model/finetune/AssetBERT/d4/2023Q4/AssetBERT_d4_2023Q4_best_embedding.csv",
                "model/finetune/AssetBERT/d4/2024Q1/AssetBERT_d4_2024Q1_best_embedding.csv",
                "model/finetune/AssetBERT/d4/2024Q2/AssetBERT_d4_2024Q2_best_embedding.csv",
                "model/finetune/AssetBERT/d4/2024Q3/AssetBERT_d4_2024Q3_best_embedding.csv"
            ],
            "benchmark_path": [
                "data/processed/Benchmark/2023Q2.csv",
                "data/processed/Benchmark/2023Q3.csv",
                "data/processed/Benchmark/2023Q4.csv",
                "data/processed/Benchmark/2024Q1.csv",
                "data/processed/Benchmark/2024Q2.csv",
                "data/processed/Benchmark/2024Q3.csv"
            ],
            "periods": [
                "2023Q2",
                "2023Q3",
                "2023Q4",
                "2024Q1",
                "2024Q2",
                "2024Q3"
            ],
            "embedding_token_col": "Token",
            "embedding_value_pattern": "Emb_*",
            "benchmark_stock_col": "Stkcd",
            "normalization": "standard",
            "features": ["embedding"],
            "align_chars": true,
            "ridge_alpha": 0.5,
            "seed": 42
        }

        Configuration File Descriptions:
        -------------------------------
        model_type         : Label of the model under evaluation, recorded with every result.
        periods            : Period labels (required), in the order of the path lists.
        embedding_path     : One embedding file per period (a list as long as `periods`); set exactly when "embedding" is in `features`.
        benchmark_path     : One benchmark file per period: the target columns, and the characteristics and IndustryCode unless `char_benchmark_path` is set.
        char_benchmark_path: Optional, one file per period supplying the characteristics and IndustryCode instead of `benchmark_path`.
        features           : Feature blocks in column order, any combination of "embedding", "placebo" (four uniform random columns), "chars" (the characteristics) and "industry" (industry dummies). Default: ["embedding"].
        align_chars        : Drop firms missing a characteristic even when "chars" is not a feature, so the sample matches the one with "chars" added. Default: true.
        ridge_alpha        : Ridge penalty. Set to `null` to select it by 10-fold cross-validation.
        scoring            : "cross_section" (fit and score on every firm of the period, the default) or "holdout" (score on held-out firms; see `holdout_test_size`, `holdout_n_splits`, `holdout_seed`, and `holdout_split_aggregate`, the median (default) or mean over the splits within a period).
        seed               : Seed of the placebo draw and the cross-validation folds.
        embedding_token_col: Column name for token/stock ID in embedding files. Default: `"Token"`.
        embedding_value_pattern: Wildcard pattern to match embedding value columns (e.g., `"Embed*"`, `"Emb_*"`). Default: `"Embed*"`.
        benchmark_stock_col: Column name for stock ID in benchmark files. Default: `"Stkcd"`.
        normalization      : Feature scaling: `"standard"` (StandardScaler), `"minmax"` (MinMaxScaler) or `"none"`. Default: `"standard"`.
        """),
    )
    utils.add_run_args(parser, override_example="ridge_alpha=null scoring=holdout")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the RV evaluation of ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        evaluator = RVBenchmarkEvaluator()
        evaluator.attach(run)
        evaluator.load_configs(config_file=args.config)
        if args.override:
            evaluator.load_overrides(utils.parse_overrides(args.override))
        run.set_config(evaluator.task_name, evaluator.config.to_dict())
        evaluator.prepare()
        evaluator.eval()


if __name__ == "__main__":
    main(get_parser().parse_args())
