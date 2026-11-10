"""AssetRS fit from a config file: a PCA of the investor-asset matrix, saved as ``<save_name>_model.pkl`` and ``<save_name>_embedding.csv`` under ``save_folder``, one ``train_rs_summary`` row per run.

Run as ``python -m asset_embeddings.scripts.train.rs -c <config> [-r <db> -e <experiment>]``.
"""

import textwrap
import argparse

from asset_embeddings import utils
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings.trainers.rs import AssetRS_Trainer


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Train an asset recommender system embedding model using dimensionality reduction techniques (PCA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model_type": "RS_Binary",
            "n_components": 4,
            "whiten": false,
            "data_path": "data/processed/ShareHoldingIntermediate/2023Q2/2023Q2.csv",
            "save_folder": "model/finetune/AssetRS/RS_Binary/d4/2023Q2",
            "save_name": "AssetRS_Binary_d4_2023Q2",
            "save_format": ".pkl",
            "seed": 42,
            "id_key": "InvestorID",
            "portfolio_key": "Portfolio",
            "proportion1_key": "Proportion1",
            "proportion2_key": "Proportion2"
        }

        Configuration File Descriptions:
        --------------------------------
        model_type     : Recommender system model type. Supported values: "RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin".
        n_components   : Target embedding dimension (number of components after dimensionality reduction).
        whiten         : Whether to whiten the reduced features (normalize variance, e.g., in PCA).
        data_path      : Path to the input portfolio data file or directory (CSV format only).
        save_folder    : Output directory to save the model.
        save_name      : File name (without extension) of the saved model.
        save_format    : File format to save the model (e.g., ".pkl").
        seed           : Random seed for reproducibility.
        id_key         : Column name for investor ID (default: "InvestorID").
        portfolio_key  : Column name for portfolio data (default: "Portfolio").
        proportion1_key  : Column name for proportion1 data (default: "Proportion1").
        proportion2_key  : Column name for proportion2 data (default: "Proportion2").
        """),
    )

    utils.add_run_args(parser, override_example="n_components=10 seed=123")
    return parser


def main(args: argparse.Namespace) -> None:
    """Fit AssetRS from ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        rs_trainer = AssetRS_Trainer()
        rs_trainer.attach(run)
        rs_trainer.load_configs(args.config)
        if args.override:
            rs_trainer.load_overrides(utils.parse_overrides(args.override))
        run.set_config(rs_trainer.task_name, rs_trainer.config.to_dict())
        rs_trainer.prepare()
        rs_trainer.train()


if __name__ == "__main__":
    main(get_parser().parse_args())
