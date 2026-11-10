"""Temporal return comovement: the RC slope fitted on the previous period and applied unchanged to the next.

One row per scored period into ``eval_rc_results`` (task ``RC_temporal``); the first period is not scored. Run as
``python -m asset_embeddings.scripts.eval.rc_temporal -c <config> -r <db> -e <experiment>``.
"""

import argparse
import textwrap

from asset_embeddings.evaluators.rc_temporal import TemporalRCEvaluator
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Temporal Return Comovement Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model_type": "AssetBERT",
            "embedding_path": [
                "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d4/2023Q3/AssetBERT_d4_2023Q3_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d4/2023Q4/AssetBERT_d4_2023Q4_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d4/2024Q1/AssetBERT_d4_2024Q1_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d4/2024Q2/AssetBERT_d4_2024Q2_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d4/2024Q3/AssetBERT_d4_2024Q3_best_embedding.csv"
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
            "embedding_value_pattern": "Embed_*",
            "benchmark_stock_col": "Stkcd",
            "normalization": "standard",
            "features": ["embedding"],
            "align_chars": true,
            "ridge_alpha": 0.5,
            "scoring": "cross_section",
            "align": "procrustes",
            "seed": 42
        }

        Configuration File Descriptions:
        -------------------------------
        model_type         : Label of the model under evaluation, recorded with every result.
        periods            : Period labels (required): consecutive quarters "YYYYQn" in increasing order, in the order of the path lists. The slope of each period is fitted on the one before; the first period is not scored.
        embedding_path     : One embedding file per period (a list as long as `periods`); set exactly when `features` is ["embedding"].
        benchmark_path     : One benchmark file per period: the target `qret_next`, and the characteristics unless `char_benchmark_path` is set.
        char_benchmark_path: Optional, one file per period supplying the characteristics instead of `benchmark_path`.
        features           : ["embedding"] (default) or ["chars"]; other blocks are refused (industry dummies can differ between periods, placebo columns are drawn afresh every period).
        align              : "none" (default): period T's embedding enters in its own coordinates. "procrustes": period T's raw embedding is rotated onto period T-1's by the orthogonal Procrustes rotation fitted on the common firms, before each period is normalized. Requires features ["embedding"].
        align_chars        : Drop firms missing a characteristic even when "chars" is not a feature. Default: true.
        ridge_alpha        : Ridge penalty of the fit on period T-1. Set to `null` to select it by 10-fold cross-validation inside T-1.
        scoring            : Only "cross_section" (the default).
        seed               : Seed of the cross-validation folds and the ridge.
        embedding_token_col: Column name for token/stock ID in embedding files. Default: `"Token"`.
        embedding_value_pattern: Wildcard pattern to match embedding value columns (e.g., `"Embed*"`, `"Emb_*"`). Default: `"Embed*"`.
        benchmark_stock_col: Column name for stock ID in benchmark files. Default: `"Stkcd"`.
        normalization      : Feature scaling per period: `"standard"` (StandardScaler), `"minmax"` (MinMaxScaler) or `"none"`. Default: `"standard"`.
        """),
    )
    utils.add_run_args(parser, override_example="ridge_alpha=null align=procrustes")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the temporal RC evaluation of ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        evaluator = TemporalRCEvaluator()
        evaluator.attach(run)
        evaluator.load_configs(config_file=args.config)
        if args.override:
            evaluator.load_overrides(utils.parse_overrides(args.override))
        run.set_config(evaluator.task_name, evaluator.config.to_dict())
        evaluator.prepare()
        evaluator.eval()


if __name__ == "__main__":
    main(get_parser().parse_args())
