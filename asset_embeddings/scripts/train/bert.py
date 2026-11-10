"""AssetBERT masked-LM training from a config file: checkpoints and embedding CSVs under ``train.save_folder``, one ``train_summary`` row per run.

Run as ``python -m asset_embeddings.scripts.train.bert -c <config> [-r <db> -e <experiment>]``.
"""

import os
import argparse
import textwrap

from asset_embeddings import utils
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings.trainers.bert import AssetBERTTrainer


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags and the config layout in the ``--help`` epilog."""
    parser = argparse.ArgumentParser(
        description="Train BERT model with Masked Language Modeling (MLM) on stock data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model": {
                "model_type": "BERT",
                "model_checkpoint": null,
                "w2v_model": "model/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
                "embedding_file": null,
                "vocab_size": 5236,
                "hidden_size": 4,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "intermediate_size": 512,
                "max_position_embeddings": 64,
                "type_vocab_size": 1,
                "freeze_embedding": false,
                "freeze_encoder": false
            },
            "tokenizer": {
                "w2v_model": "model/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
                "vocab_file": null,
                "pretrained_tokenizer_file": null,
                "alias_file": null
            },
            "dataset": {
                "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
                "data_format": "csv",
                "id_key": "InvestorID",
                "portfolio_key": "Portfolio",
                "proportion1_key": "Proportion1",
                "proportion2_key": "Proportion2",
                "include_proportion": true,
                "max_length": 64,
                "num_repeats": 1,
                "mask_prob": 0.15,
                "mask_indices": null
            },
            "dataloader": {
                "batch_size": 128,
                "shuffle": true,
                "num_workers": 4,
                "pin_memory": false
            },
            "optimizer": {
                "optimizer_type": "PagedAdam8bit",
                "optimizer_kwargs": null,
                "learning_rate": 0.001,
                "lr_scheduler_type": "cosine",
                "lr_scheduler_warmup_steps": 0,
                "lr_scheduler_train_steps": null,
                "lr_scheduler_num_cycles": 0.5,
                "lr_scheduler_power": 0
            },
            "accelerator": {
                "mixed_precision": "auto",
                "gradient_accumulation_steps": 1,
                "accelerator_checkpoint": null,
                "use_compile": false,
                "compile_backend": "inductor",
                "compile_mode": "reduce-overhead",
                "compile_fullgraph": false,
                "compile_dynamic": false,
                "compile_cache_dir": null,
                "allow_tf32": false
            },
            "train": {
                "max_epochs": 5,
                "clip_grad_norm": null,
                "clip_grad_value": null,
                "detect_anomaly": false,
                "log_every_steps": 200,
                "report_every_epochs": 1,
                "save_every_epochs": 1,
                "keep_best_only": false,
                "calculate_contextualized_embeddings": true,
                "save_folder": "model/finetune/AssetBERT/d4/2023Q2",
                "save_name": "AssetBERT_d4_2023Q2",
                "save_format": ".safetensors",
                "seed": 42,
                "early_stop_enabled": false,
                "early_stop_patience": 5,
                "early_stop_min_delta": 0.0,
                "early_stop_mode": "min",
                "early_stop_relative": true,
                "early_stop_metric": "train_loss",
                "validation_split": null,
                "split_method": "random",
                "best_metric": "train_loss"
            }
        }

        Configuration File Descriptions:
        --------------------------------
        model.model_type               : Model type, fixed as "BERT".
        model.model_checkpoint         : Pretrained model path; null means training from scratch.
        model.w2v_model                : Path to Word2Vec model for initializing embeddings.
        model.embedding_file           : Optional CSV with embedding weights; null if unused.
        model.vocab_size               : Number of asset tokens in the vocabulary.
        model.hidden_size              : BERT hidden dimension (embedding size).
        model.num_hidden_layers        : Number of transformer encoder layers.
        model.num_attention_heads      : Number of attention heads (must divide hidden size).
        model.intermediate_size        : Size of feedforward layers.
        model.max_position_embeddings  : Max number of assets (i.e., sequence length).
        model.type_vocab_size          : Usually 1 for asset sequences.
        model.freeze_embedding         : If true, freeze embedding layer weights.
        model.freeze_encoder           : If true, freeze encoder parameters.

        tokenizer.w2v_model            : Path to Word2Vec model (for tokenizer).
        tokenizer.vocab_file           : Optional vocab file from transformers.
        tokenizer.pretrained_tokenizer_file : Optional saved tokenizer path.
        tokenizer.alias_file           : Optional JSON/CSV for stock alias mapping.

        dataset.data_path              : Input data directory or file.
        dataset.data_format            : Format: "csv" (the only one).
        dataset.id_key                 : Column name for investor ID (default: "InvestorID").
        dataset.portfolio_key          : Column name for portfolio data (default: "Portfolio").
        dataset.proportion1_key        : Column name for shareholding proportion relative to total shares.
        dataset.proportion2_key        : Column name for shareholding proportion relative to tradable A-shares.
        dataset.include_proportion     : Whether to include proportion information.
        dataset.max_length             : Sequence length (should match model.max_position_embeddings).
        dataset.num_repeats            : Times the training split is replayed per epoch.
        dataset.mask_prob              : Masked LM probability.
        dataset.mask_indices           : Fixed mask positions; mutually exclusive with mask_prob.

        dataloader.batch_size          : Number of samples per batch.
        dataloader.shuffle             : Whether to shuffle training data.
        dataloader.num_workers         : Number of dataloader workers.
        dataloader.persistent_workers  : Whether to use persistent workers.
        dataloader.pin_memory          : Pin memory (if using CUDA).

        optimizer.optimizer_type       : Optimizer name. Supports Adam, AdamW, Lion, etc.
        optimizer.optimizer_kwargs     : Optional dictionary for optimizer arguments.
        optimizer.learning_rate        : Base learning rate.
        optimizer.lr_scheduler_type    : Scheduler: constant, linear, cosine, polynomial, etc.
        optimizer.lr_scheduler_warmup_steps : Number of warmup steps.
        optimizer.lr_scheduler_train_steps  : Total steps (null for auto).
        optimizer.lr_scheduler_num_cycles   : For cosine/cosine_with_restarts.
        optimizer.lr_scheduler_power   : Power for polynomial scheduler.

        accelerator.mixed_precision    : "no", "fp16", "bf16", or "auto".
        accelerator.gradient_accumulation_steps : Grad accumulation steps.
        accelerator.accelerator_checkpoint : Checkpoint path for accelerator framework.
        accelerator.use_compile        : Enable torch.compile. Default: false.
        accelerator.compile_backend    : Compile backend: "inductor", "eager", "aot_eager".
        accelerator.compile_mode       : Compile mode: "default", "reduce-overhead", "max-autotune".
        accelerator.compile_fullgraph  : Require full graph compilation (no graph breaks).
        accelerator.compile_dynamic    : Enable dynamic shapes (for variable batch size).
        accelerator.compile_cache_dir  : Persistent compile cache directory. null = PyTorch default.
        accelerator.allow_tf32         : Enable TF32 on Ampere/Ada GPUs. Default: false.

        train.max_epochs               : Maximum number of training epochs.
        train.clip_grad_norm           : Max gradient norm (L2).
        train.clip_grad_value          : Max gradient value (absolute).
        train.detect_anomaly           : Enable autograd anomaly detection.
        train.log_every_steps          : Steps between training-loss log lines.
        train.report_every_epochs      : Epochs between epoch-loss reports.
        train.save_every_epochs        : Epochs between saved checkpoints.
        train.calculate_contextualized_embeddings : Whether to compute contextualized embeddings after training.
        train.save_folder              : Folder to save model.
        train.save_name                : Name of saved model.
        train.save_format              : Format of model file (e.g., ".safetensors").
        train.seed                     : Random seed.
        train.early_stop_enabled       : Whether to enable early stopping.
        train.early_stop_patience      : Number of epochs without improvement before stopping.
        train.early_stop_min_delta     : Minimum improvement threshold.
        train.early_stop_mode          : Optimization direction ("min" or "max").
        train.early_stop_relative      : Whether to use relative improvement.
        train.early_stop_metric        : Metric to monitor ("train_loss" or "val_loss").
        train.validation_split         : Validation split ratio (null for no split, e.g., 0.1 for 10%).
        train.split_method             : Split method ("random" or "sequential").
        train.best_metric              : Metric for best epoch selection ("train_loss" or "val_loss").
        """),
    )

    utils.add_run_args(parser, override_example="train.max_epochs=10 optimizer.learning_rate=0.001")
    return parser


def main(args: argparse.Namespace) -> None:
    """Train AssetBERT from ``args.config`` inside one recorded run."""
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        trainer = AssetBERTTrainer()
        trainer.attach(run)
        trainer.load_configs(config_file=args.config)
        if args.override:
            trainer.load_overrides(utils.parse_overrides(args.override))
        run.set_config(trainer.task_name, trainer.config.to_dict())
        trainer.prepare()
        trainer.train()


if __name__ == "__main__":
    main(get_parser().parse_args())
