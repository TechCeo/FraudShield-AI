"""Command-line orchestration for fusion, latency, and drift controls."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..utils import atomic_write_json, json_digest, sha256_file
from .data import ModelDataPaths, load_model_dataset
from .drift import DriftConfig, DriftDetector, run_drift_simulation
from .hybrid import (
    HybridInferenceEngine,
    benchmark_model_suite,
    build_operational_tradeoff_matrix,
    run_hybrid_search,
)
from .search import build_evaluation_matrix, load_model_report
from .sequences import (
    GlobalSparseAccessor,
    load_sequence_index,
    sequence_tensor_batch,
)

LOGGER = logging.getLogger(__name__)
ALL_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "fnn",
    "lstm",
    "hybrid",
)


def _load_digest_document(path: Path, artifact_type: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"artifact root must be an object: {path}")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError(f"artifact digest does not match its content: {path}")
    if content.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected artifact type: {path}")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize hybrid fusion and evaluate operational controls."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="FraudShield repository root",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace registered outputs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="optimize and evaluate hybrid fusion")
    fit.add_argument("--weight-samples", type=int, default=32)
    fit.add_argument("--minimum-weight", type=float, default=0.05)

    benchmark = subparsers.add_parser(
        "benchmark", help="measure warm prepared-feature model latency"
    )
    benchmark.add_argument("--warmups", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=20)

    drift = subparsers.add_parser(
        "simulate-drift", help="fit drift references and evaluate controlled windows"
    )
    drift.add_argument("--feature-sample-rows", type=int, default=100_000)
    drift.add_argument("--prediction-sample-rows", type=int, default=10_000)

    subparsers.add_parser(
        "summarize", help="build six-model quality and operational registries"
    )

    run_all = subparsers.add_parser(
        "run-all", help="run fusion, latency, drift, and report consolidation"
    )
    run_all.add_argument("--weight-samples", type=int, default=32)
    run_all.add_argument("--minimum-weight", type=float, default=0.05)
    run_all.add_argument("--warmups", type=int, default=3)
    run_all.add_argument("--repeats", type=int, default=20)
    run_all.add_argument("--feature-sample-rows", type=int, default=100_000)
    run_all.add_argument("--prediction-sample-rows", type=int, default=10_000)
    return parser


def _fit(args, data, sequence_index, output_dir: Path) -> dict[str, Any]:
    report = run_hybrid_search(
        data,
        sequence_index,
        output_dir,
        n_weight_samples=args.weight_samples,
        minimum_component_weight=args.minimum_weight,
        random_state=args.random_state,
        device_name=args.device,
        overwrite=args.force,
    )
    LOGGER.info(
        "hybrid holdout AP=%.6f recall=%.6f precision=%.6f",
        report["holdout_metrics"]["ranking"]["average_precision"],
        report["holdout_metrics"]["operating_point"]["recall"],
        report["holdout_metrics"]["operating_point"]["precision"],
    )
    return report


def _prepared_benchmark_rows(
    data, sequence_index, sequence_length: int, rows: int = 256
):
    accessor = GlobalSparseAccessor(data)
    validation_start, validation_stop = sequence_index.offsets["validation"]
    if validation_stop - validation_start < rows:
        raise ValueError("validation partition is smaller than benchmark request")
    endpoints = np.arange(validation_start, validation_start + rows, dtype=np.int64)
    static = data.validation_features[:rows].toarray().astype(np.float32)
    features, lengths = sequence_tensor_batch(
        accessor,
        endpoints,
        sequence_index.previous,
        sequence_length,
        torch.device("cpu"),
    )
    return static, features.numpy(), lengths.numpy()


def _benchmark(
    args, data, sequence_index, output_dir: Path
) -> dict[str, Any]:
    engine = HybridInferenceEngine.load(output_dir, device_name=args.device)
    static, sequences, lengths = _prepared_benchmark_rows(
        data, sequence_index, engine.sequence_length
    )
    content = benchmark_model_suite(
        engine,
        output_dir,
        static,
        sequences,
        lengths,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    content["hybrid_config_sha256"] = sha256_file(
        output_dir / "hybrid_config.json"
    )
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(
        document,
        output_dir / "latency_benchmark.json",
        overwrite=args.force,
    )
    return document


def _simulate_drift(
    args, data, output_dir: Path
) -> dict[str, Any]:
    hybrid_report = load_model_report(output_dir / "hybrid_report.json")
    prediction_path = output_dir / hybrid_report["prediction_file"]
    if sha256_file(prediction_path) != hybrid_report["prediction_sha256"]:
        raise ValueError("hybrid prediction cache differs from its report")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        validation_predictions = arrays["validation_hybrid"].astype(np.float64)
        holdout_predictions = arrays["holdout_hybrid"].astype(np.float64)
    config = DriftConfig(
        feature_sample_rows=args.feature_sample_rows,
        prediction_sample_rows=args.prediction_sample_rows,
        random_state=args.random_state,
    )
    detector = DriftDetector(config).fit(
        data.validation_features,
        data.metadata["feature_names"],
        validation_predictions,
        reference_context={
            "feature_reference_partition": "chronological_validation",
            "prediction_reference_partition": "chronological_validation",
            "reference_policy": "shared_recent_labeled_window",
            "model_data_manifest_sha256": data.metadata["payload_sha256"],
            "hybrid_report_sha256": hybrid_report["payload_sha256"],
        },
    )
    detector_path = output_dir / "drift_detector.json"
    detector.save(detector_path, overwrite=args.force)
    report = run_drift_simulation(
        detector,
        data.validation_features,
        data.holdout_features,
        validation_predictions,
        holdout_predictions,
        output_dir / "drift_simulation_report.json",
        overwrite=args.force,
    )
    report["detector_file"] = detector_path.name
    report["detector_sha256"] = sha256_file(detector_path)
    content = {key: value for key, value in report.items() if key != "payload_sha256"}
    report = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(
        report,
        output_dir / "drift_simulation_report.json",
        overwrite=True,
    )
    return report


def _summarize(args, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reports = {
        name: load_model_report(output_dir / f"{name}_report.json")
        for name in ALL_MODELS
    }
    quality = build_evaluation_matrix(
        reports,
        output_dir / "hybrid_evaluation_matrix.json",
        overwrite=args.force,
    )
    latency = _load_digest_document(
        output_dir / "latency_benchmark.json",
        "fraud_model_latency_benchmark",
    )
    tradeoffs = build_operational_tradeoff_matrix(
        reports,
        latency,
        output_dir,
        output_dir / "operational_tradeoff_matrix.json",
        overwrite=args.force,
    )
    return quality, tradeoffs


def main(argv: Sequence[str] | None = None) -> int:
    """Run hybrid fitting, benchmarking, drift simulation, or consolidation."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_root = args.project_root.resolve()
    paths = ModelDataPaths.project_defaults(project_root)
    output_dir = project_root / "artifacts" / "models"
    data = load_model_dataset(paths.output_dir)
    sequence_index = load_sequence_index(paths.output_dir)
    if args.command == "fit":
        _fit(args, data, sequence_index, output_dir)
        return 0
    if args.command == "benchmark":
        _benchmark(args, data, sequence_index, output_dir)
        return 0
    if args.command == "simulate-drift":
        report = _simulate_drift(args, data, output_dir)
        LOGGER.info(
            "drift validation=%s holdout=%s injected=%s",
            report["scenarios"]["validation"]["overall_status"],
            report["scenarios"]["holdout"]["overall_status"],
            report["scenarios"]["injected"]["overall_status"],
        )
        return 0
    if args.command == "summarize":
        _summarize(args, output_dir)
        return 0

    _fit(args, data, sequence_index, output_dir)
    _benchmark(args, data, sequence_index, output_dir)
    _simulate_drift(args, data, output_dir)
    _summarize(args, output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
