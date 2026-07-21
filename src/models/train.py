"""Command-line orchestration for classical fraud classifier training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from .data import (
    ModelDataPaths,
    build_model_dataset,
    ensure_feature_streams,
    load_model_dataset,
)
from .search import (
    SUPPORTED_MODELS,
    build_evaluation_matrix,
    load_model_report,
    run_model_search,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_ITERATIONS = {
    "logistic_regression": 5,
    "random_forest": 4,
    "xgboost": 6,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare sparse data and optimize classical fraud classifiers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="FraudShield repository root",
    )
    parser.add_argument(
        "--chunksize", type=int, default=100_000, help="CSV transformation rows per chunk"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="replace registered outputs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare-data",
        help="generate causal feature streams and sparse chronological partitions",
    )

    fit = subparsers.add_parser(
        "fit", help="optimize and evaluate one registered classifier"
    )
    fit.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    fit.add_argument("--iterations", type=int)

    run_all = subparsers.add_parser(
        "run-all", help="optimize and evaluate all registered classifiers"
    )
    run_all.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS)
    )
    run_all.add_argument(
        "--iterations",
        type=int,
        help="override the registered search count for every selected classifier",
    )
    summarize = subparsers.add_parser(
        "summarize", help="verify model reports and create the evaluation matrix"
    )
    summarize.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS)
    )
    return parser


def _prepare(paths: ModelDataPaths, args: argparse.Namespace) -> None:
    ensure_feature_streams(paths, chunksize=args.chunksize)
    build_model_dataset(
        paths,
        chunksize=args.chunksize,
        overwrite=args.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run model-data preparation or classifier optimization."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = ModelDataPaths.project_defaults(args.project_root.resolve())
    if args.command == "prepare-data":
        _prepare(paths, args)
        LOGGER.info("model data written to %s", paths.output_dir)
        return 0

    model_output = args.project_root.resolve() / "artifacts" / "models"
    if args.command == "summarize":
        reports = {
            model_name: load_model_report(
                model_output / f"{model_name}_report.json"
            )
            for model_name in args.models
        }
        build_evaluation_matrix(
            reports,
            model_output / "evaluation_matrix.json",
            overwrite=args.force,
        )
        return 0

    data = load_model_dataset(paths.output_dir)
    if args.command == "fit":
        iterations = args.iterations or DEFAULT_ITERATIONS[args.model]
        report = run_model_search(
            args.model,
            data,
            model_output,
            n_iter=iterations,
            random_state=args.random_state,
            overwrite=args.force,
        )
        LOGGER.info(
            "%s holdout AP=%.6f recall=%.6f precision=%.6f",
            args.model,
            report["holdout_metrics"]["ranking"]["average_precision"],
            report["holdout_metrics"]["operating_point"]["recall"],
            report["holdout_metrics"]["operating_point"]["precision"],
        )
        return 0

    reports: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        iterations = args.iterations or DEFAULT_ITERATIONS[model_name]
        reports[model_name] = run_model_search(
            model_name,
            data,
            model_output,
            n_iter=iterations,
            random_state=args.random_state,
            overwrite=args.force,
        )
    build_evaluation_matrix(
        reports,
        model_output / "evaluation_matrix.json",
        overwrite=args.force,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
