"""Pairwise similarity: Fama-MacBeth regression of firm-pair embedding similarity on same-value indicators; rows into ``eval_pairwise_*``.

Run as ``python -m asset_embeddings.scripts.eval.pairwise_similarity -c <config> -r <db> -e <experiment>``.
"""

import argparse
import textwrap

from asset_embeddings.evaluators.pairwise_similarity import PairwiseSimilarityEvaluator
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Pairwise Similarity Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model_type": "AssetBERT",
            "embedding_path": [
                "checkpoints/finetune/AssetBERT/d128/2023Q2/AssetBERT_d128_2023Q2_best_embedding.csv",
                "checkpoints/finetune/AssetBERT/d128/2023Q3/AssetBERT_d128_2023Q3_best_embedding.csv"
            ],
            "benchmark_path": [
                "data/processed/Benchmark/2023Q2.csv",
                "data/processed/Benchmark/2023Q3.csv"
            ],
            "label_path": [
                "data/processed/FirmLabels/SOE/2023Q2.csv",
                "data/processed/FirmLabels/SOE/2023Q3.csv"
            ],
            "label_col": "SOE",
            "match_on": ["SOE", "IndustryCode"],
            "interaction": false,
            "periods": ["2023Q2", "2023Q3"],
            "embedding_value_pattern": "Embed_*"
        }

        Configuration File Descriptions:
        -------------------------------
        model_type         : Label of the model under evaluation, recorded with every result.
        periods            : Period labels (required), in the order of the path lists.
        embedding_path     : One embedding file per period (required); the raw vectors enter the cosine similarity.
        benchmark_path     : One benchmark file per period: the match_on columns other than the label (e.g. IndustryCode).
        label_path         : One label file per period: the stock ID and the label column, one row per stock.
        label_col          : The label column. Firms without a label leave the sample.
        label_stock_col    : Stock ID column of the label files. Default: `"Stkcd"`.
        match_on           : 1 to 4 columns; a pair's indicator is 1 when the two firms share the column's value.
        interaction        : Add the products of every subset of two or more indicators. Default: false.
        """),
    )
    utils.add_run_args(parser, override_example="interaction=true")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the pairwise-similarity regression of ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        evaluator = PairwiseSimilarityEvaluator()
        evaluator.attach(run)
        evaluator.load_configs(config_file=args.config)
        if args.override:
            evaluator.load_overrides(utils.parse_overrides(args.override))
        run.set_config(evaluator.task_name, evaluator.config.to_dict())
        evaluator.prepare()
        evaluator.eval()


if __name__ == "__main__":
    main(get_parser().parse_args())
