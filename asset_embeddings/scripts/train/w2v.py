"""AssetW2V training from a config file: a gensim ``<save_name>.model`` under ``save_folder``, one ``train_w2v_summary`` row per run.

Run as ``python -m asset_embeddings.scripts.train.w2v -c <config> [-r <db> -e <experiment>]``.
"""

import argparse
import textwrap

from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils
from asset_embeddings.trainers.w2v import Word2Vec_Trainer


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Train a Stock Embedding Model using Word2Vec architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "pretrained_model": null,
            "embedding_dim": 4,
            "epochs": 10,
            "window": 5,
            "min_count": 1,
            "sg": 1,
            "sample": 0.001,
            "negative_sample": 10,
            "seed": 42,
            "data_path": "data/processed/ShareHoldingSplit/base",
            "data_format": "csv",
            "workers": 1,
            "save_folder": "model/base/AssetW2V/d4",
            "save_name": "AssetW2V_d4_base"
        }

        Configuration File Descriptions:
        -------------------------------
        pretrained_model   : Optional pretrained model path (set to null for training from scratch).
        embedding_dim      : Word embedding dimension (vector size for each stock).
        epochs             : Number of training epochs.
        window             : Context window size for co-occurrence.
        min_count          : Minimum frequency threshold for stocks (1 recommended).
        sg                 : Training mode: 1 for Skip-Gram, 0 for CBOW.
        sample             : Threshold for downsampling frequent stocks (e.g., 0.001).
        negative_sample    : Number of negative samples per positive.
        seed               : Random seed for reproducibility.
        data_path          : Path to input data (CSV, JSON, or binary).
        data_format        : Format of input data: csv (the only one).
        id_key             : Column name for investor ID (default: "InvestorID").
        portfolio_key      : Column name for portfolio data (default: "Portfolio").
        workers            : Number of parallel workers. (1 recommended for reproducibility)
        save_folder        : Directory to save the trained model.
        save_name          : Filename of the saved model (without extension).
        """),
    )
    utils.add_run_args(parser, override_example="window=10 seed=123")
    return parser


def main(args: argparse.Namespace) -> None:
    """Train AssetW2V from ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        w2v_trainer = Word2Vec_Trainer()
        w2v_trainer.attach(run)
        w2v_trainer.load_configs(args.config)
        if args.override:
            w2v_trainer.load_overrides(utils.parse_overrides(args.override))
        run.set_config(w2v_trainer.task_name, w2v_trainer.config.to_dict())
        w2v_trainer.prepare()
        w2v_trainer.train()


if __name__ == "__main__":
    main(get_parser().parse_args())
