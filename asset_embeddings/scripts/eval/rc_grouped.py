"""Return-comovement benchmark within coverage groups: the RC ridge fit per (period, group) cell with a bootstrap interval, and permutation tests between groups; rows into ``eval_grouped_rc_*``.

Run as ``python -m asset_embeddings.scripts.eval.rc_grouped -c <config> -r <db> -e <experiment>``.
"""

import argparse

from asset_embeddings.evaluators.grouped import GroupedRCEvaluator
from asset_embeddings.logger import configure_logging
from asset_embeddings.records import Run, module_name
from asset_embeddings import utils


def get_parser() -> argparse.ArgumentParser:
    """Build the parser: the six shared flags; the config fields are documented in ``GroupedRCEvalConfig``."""
    parser = argparse.ArgumentParser(description="Grouped Return Comovement Benchmark Evaluation")
    utils.add_run_args(parser, override_example="grouping=fixed")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the grouped RC evaluation of ``args.config`` inside one recorded run."""
    configure_logging(log_files=args.log, console_level="DEBUG" if args.verbose else "INFO")
    with Run.from_args(args, script=module_name(globals())) as run:
        evaluator = GroupedRCEvaluator()
        evaluator.attach(run)
        evaluator.load_configs(config_file=args.config)
        if args.override:
            evaluator.load_overrides(utils.parse_overrides(args.override))
        run.set_config(evaluator.task_name, evaluator.config.to_dict())
        evaluator.prepare()
        evaluator.eval()


if __name__ == "__main__":
    main(get_parser().parse_args())
