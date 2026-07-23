"""Command-line orchestration for static and sequential neural classifiers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from .data import ModelDataPaths, load_model_dataset
from .fnn import run_fnn_search
from .lstm import run_lstm_search
from .search import build_evaluation_matrix, load_model_report
from .sequences import build_sequence_index, load_sequence_index

LOGGER = logging.getLogger(__name__)
DEEP_MODELS = ("fnn", "lstm")
ALL_REPORTS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "fnn",
    "lstm",
)
DEFAULT_ITERATIONS = {"fnn": 3, "lstm": 3}
DEFAULT_EPOCHS = {"fnn": 6, "lstm": 5}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare sequences and optimize neural fraud classifiers."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="FraudShield repository root"
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace registered outputs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-sequences", help="create strictly prior per-card sequence pointers"
    )

    fit = subparsers.add_parser("fit", help="optimize one neural classifier")
    fit.add_argument("--model", choices=DEEP_MODELS, required=True)
    fit.add_argument("--iterations", type=int)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--patience", type=int, default=2)

    run_all = subparsers.add_parser(
        "run-all", help="optimize both registered neural classifiers"
    )
    run_all.add_argument("--iterations", type=int)
    run_all.add_argument("--epochs", type=int)
    run_all.add_argument("--patience", type=int, default=2)

    subparsers.add_parser(
        "summarize", help="verify all classifier reports and create a combined registry"
    )
    return parser


def _fit_model(
    model_name: str,
    data,
    paths: ModelDataPaths,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    iterations = args.iterations or DEFAULT_ITERATIONS[model_name]
    epochs = args.epochs or DEFAULT_EPOCHS[model_name]
    common = {
        "n_iter": iterations,
        "max_epochs": epochs,
        "patience": args.patience,
        "random_state": args.random_state,
        "device_name": args.device,
        "overwrite": args.force,
    }
    if model_name == "fnn":
        return run_fnn_search(data, output_dir, **common)
    sequence_index = load_sequence_index(paths.output_dir)
    return run_lstm_search(data, sequence_index, output_dir, **common)


def main(argv: Sequence[str] | None = None) -> int:
    """Run sequence preparation, neural optimization, or report consolidation."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_root = args.project_root.resolve()
    paths = ModelDataPaths.project_defaults(project_root)
    output_dir = project_root / "artifacts" / "models"
    if args.command == "prepare-sequences":
        report = build_sequence_index(
            paths, chunksize=args.chunksize, overwrite=args.force
        )
        LOGGER.info(
            "sequence index rows=%s cards=%s",
            f"{report['rows']:,}",
            f"{report['cards']:,}",
        )
        return 0
    if args.command == "summarize":
        reports = {
            model_name: load_model_report(
                output_dir / f"{model_name}_report.json"
            )
            for model_name in ALL_REPORTS
        }
        build_evaluation_matrix(
            reports,
            output_dir / "deep_evaluation_matrix.json",
            overwrite=args.force,
        )
        return 0

    data = load_model_dataset(paths.output_dir)
    if args.command == "fit":
        report = _fit_model(args.model, data, paths, output_dir, args)
        LOGGER.info(
            "%s holdout AP=%.6f recall=%.6f precision=%.6f",
            args.model,
            report["holdout_metrics"]["ranking"]["average_precision"],
            report["holdout_metrics"]["operating_point"]["recall"],
            report["holdout_metrics"]["operating_point"]["precision"],
        )
        return 0

    sequence_manifest = paths.output_dir / "sequence_index_manifest.json"
    if not sequence_manifest.is_file():
        build_sequence_index(paths, chunksize=args.chunksize)
    reports = {
        model_name: _fit_model(model_name, data, paths, output_dir, args)
        for model_name in DEEP_MODELS
    }
    classical = {
        model_name: load_model_report(output_dir / f"{model_name}_report.json")
        for model_name in ALL_REPORTS[:3]
    }
    build_evaluation_matrix(
        {**classical, **reports},
        output_dir / "deep_evaluation_matrix.json",
        overwrite=args.force,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
