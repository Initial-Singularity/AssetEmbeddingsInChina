"""Label probe: cross-validated logit AUC of a binary firm label on the period's features, one row per period into ``eval_probe_results``.

Run as ``python -m asset_embeddings.scripts.eval.label_probe -c <config> -r <db> -e <experiment>``.
"""

import argparse
import textwrap

from asset_embeddings.evaluators.label_probe import LabelProbeEvaluator
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Label Probe Evaluation",
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
            "periods": ["2023Q2", "2023Q3"],
            "embedding_value_pattern": "Embed_*",
            "features": ["embedding"],
            "align_chars": false,
            "normalization": "standard",
            "n_folds": 5,
            "seed": 42
        }

        Configuration File Descriptions:
        -------------------------------
        model_type         : Label of the model under evaluation, recorded with every result.
        periods            : Period labels (required), in the order of the path lists.
        embedding_path     : One embedding file per period; set exactly when "embedding" is in `features`.
        benchmark_path     : One benchmark file per period: the firm universe, and the characteristics unless `char_benchmark_path` is set.
        label_path         : One label file per period: the stock ID and the label column, one row per stock.
        label_col          : The label column; it must be binary 0/1. Firms without a label leave the sample.
        label_stock_col    : Stock ID column of the label files. Default: `"Stkcd"`.
        features           : Feature blocks: "embedding", "placebo", "chars" (the RV characteristics) and "industry". Default: ["embedding"].
        align_chars        : Drop firms missing a characteristic even when "chars" is not a feature. Default: true.
        normalization      : Feature scaling on the whole cross section: `"standard"`, `"minmax"` or `"none"`. Default: `"standard"`.
        n_folds            : Folds of the stratified split (2 to 20). Default: 5.
        seed               : Seed of the fold split and the placebo draw. Default: 42.
        """),
    )
    utils.add_run_args(parser, override_example="n_folds=10 seed=7")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the label probe of ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        evaluator = LabelProbeEvaluator()
        evaluator.attach(run)
        evaluator.load_configs(config_file=args.config)
        if args.override:
            evaluator.load_overrides(utils.parse_overrides(args.override))
        run.set_config(evaluator.task_name, evaluator.config.to_dict())
        evaluator.prepare()
        evaluator.eval()


if __name__ == "__main__":
    main(get_parser().parse_args())
