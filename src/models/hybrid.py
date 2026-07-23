"""Validation-optimized hybrid fusion and warm low-latency inference."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score

from ..utils import atomic_write_json, json_digest, runtime_dependencies, sha256_file
from .data import ModelDataset
from .deep_common import resolve_device
from .evaluation import (
    evaluate_probabilities,
    select_fbeta_threshold,
    threshold_selection_dict,
)
from .fnn import load_fnn, predict_fnn
from .lstm import load_lstm, predict_lstm
from .search import load_model_report
from .sequences import GlobalSparseAccessor, SequenceIndex

LOGGER = logging.getLogger(__name__)
COMPONENT_NAMES = ("xgboost", "fnn", "lstm")
HYBRID_CONFIG_SCHEMA_VERSION = 1


def _probability_array(values: Sequence[float] | np.ndarray, rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (rows,):
        raise ValueError("component probabilities must have equal one-dimensional shapes")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("component probabilities must be finite and in [0, 1]")
    return probabilities


def _normalized_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != set(COMPONENT_NAMES):
        raise ValueError(f"weights must contain exactly {COMPONENT_NAMES}")
    normalized = {name: float(weights[name]) for name in COMPONENT_NAMES}
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError("component weights must be finite and nonnegative")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("at least one component weight must be positive")
    return {name: value / total for name, value in normalized.items()}


def blend_probabilities(
    components: Mapping[str, Sequence[float] | np.ndarray],
    weights: Mapping[str, float],
    *,
    blend_space: str,
) -> np.ndarray:
    """Blend aligned component probabilities in probability or log-odds space."""

    if set(components) != set(COMPONENT_NAMES):
        raise ValueError(f"components must contain exactly {COMPONENT_NAMES}")
    rows = len(np.asarray(components[COMPONENT_NAMES[0]]))
    arrays = {
        name: _probability_array(components[name], rows) for name in COMPONENT_NAMES
    }
    resolved = _normalized_weights(weights)
    if blend_space == "probability":
        output = sum(resolved[name] * arrays[name] for name in COMPONENT_NAMES)
    elif blend_space == "logit":
        epsilon = 1e-7
        logit = sum(
            resolved[name]
            * np.log(
                np.clip(arrays[name], epsilon, 1.0 - epsilon)
                / (1.0 - np.clip(arrays[name], epsilon, 1.0 - epsilon))
            )
            for name in COMPONENT_NAMES
        )
        output = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    else:
        raise ValueError("blend_space must be 'probability' or 'logit'")
    return np.asarray(output, dtype=np.float64)


def sample_hybrid_candidates(
    *,
    n_weight_samples: int = 32,
    minimum_component_weight: float = 0.05,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Return deterministic constrained simplex weights for both blend spaces."""

    if n_weight_samples <= 0:
        raise ValueError("n_weight_samples must be positive")
    minimum = float(minimum_component_weight)
    if not 0.0 < minimum < 1.0 / len(COMPONENT_NAMES):
        raise ValueError("minimum_component_weight must be in (0, 1/3)")
    anchors = [
        np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64),
        np.array([0.70, 0.05, 0.25], dtype=np.float64),
        np.array([0.65, 0.10, 0.25], dtype=np.float64),
        np.array([0.60, 0.05, 0.35], dtype=np.float64),
        np.array([0.55, 0.10, 0.35], dtype=np.float64),
        np.array([0.50, 0.05, 0.45], dtype=np.float64),
    ]
    generator = np.random.default_rng(int(random_state))
    random_count = max(0, n_weight_samples - len(anchors))
    random_weights = generator.dirichlet(
        np.array([4.0, 1.25, 2.5]), size=random_count
    )
    scale = 1.0 - len(COMPONENT_NAMES) * minimum
    constrained = [
        minimum + scale * weights for weights in [*anchors, *random_weights]
    ][:n_weight_samples]
    candidates: list[dict[str, Any]] = []
    for blend_space in ("probability", "logit"):
        for weights in constrained:
            resolved = _normalized_weights(
                dict(zip(COMPONENT_NAMES, weights.tolist(), strict=True))
            )
            candidates.append(
                {"blend_space": blend_space, "weights": resolved}
            )
    return candidates


def optimize_hybrid_blend(
    target: np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    n_weight_samples: int = 32,
    minimum_component_weight: float = 0.05,
    random_state: int = 42,
    threshold_beta: float = 2.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Select fusion weights and a threshold using validation rows only."""

    target_array = np.asarray(target, dtype=np.int8)
    candidates = sample_hybrid_candidates(
        n_weight_samples=n_weight_samples,
        minimum_component_weight=minimum_component_weight,
        random_state=random_state,
    )
    results: list[dict[str, Any]] = []
    best_index = -1
    best_probabilities: np.ndarray | None = None
    best_rank: tuple[float, float, float, float] | None = None
    for index, candidate in enumerate(candidates):
        probabilities = blend_probabilities(
            components,
            candidate["weights"],
            blend_space=candidate["blend_space"],
        )
        selection = select_fbeta_threshold(
            target_array, probabilities, beta=threshold_beta
        )
        average_precision = float(
            average_precision_score(target_array, probabilities)
        )
        result = {
            "candidate_index": index,
            "blend_space": candidate["blend_space"],
            "weights": candidate["weights"],
            "validation_average_precision": average_precision,
            "threshold": selection.threshold,
            "validation_fbeta": selection.fbeta,
            "validation_recall": selection.recall,
            "validation_precision": selection.precision,
        }
        results.append(result)
        rank = (
            average_precision,
            selection.fbeta,
            selection.recall,
            selection.precision,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_index = index
            best_probabilities = probabilities
    if best_index < 0 or best_probabilities is None:
        raise RuntimeError("hybrid search did not select a candidate")
    return results[best_index], results, best_probabilities


def _load_xgboost(path: Path):
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(path)
    return model


def _component_reports(output_dir: Path) -> dict[str, dict[str, Any]]:
    reports = {
        name: load_model_report(output_dir / f"{name}_report.json")
        for name in COMPONENT_NAMES
    }
    data_digests = {report["model_data_manifest_sha256"] for report in reports.values()}
    schema_digests = {report["feature_schema_sha256"] for report in reports.values()}
    if len(data_digests) != 1 or len(schema_digests) != 1:
        raise ValueError("hybrid components do not share model-data lineage")
    return reports


def collect_component_probabilities(
    data: ModelDataset,
    sequence_index: SequenceIndex,
    output_dir: str | Path,
    partition: str,
    *,
    device_name: str = "auto",
) -> dict[str, np.ndarray]:
    """Score one registered partition with all three frozen component models."""

    if partition not in {"validation", "holdout"}:
        raise ValueError("partition must be 'validation' or 'holdout'")
    output_root = Path(output_dir)
    reports = _component_reports(output_root)
    if reports["lstm"]["sequence_index_manifest_sha256"] != sequence_index.metadata[
        "payload_sha256"
    ]:
        raise ValueError("LSTM and sequence index have different lineage")
    if next(iter({r["model_data_manifest_sha256"] for r in reports.values()})) != data.metadata[
        "payload_sha256"
    ]:
        raise ValueError("hybrid components and loaded model data differ")
    device = resolve_device(device_name)
    static_matrix = (
        data.validation_features if partition == "validation" else data.holdout_features
    )
    xgboost = _load_xgboost(output_root / reports["xgboost"]["model_file"])
    fnn = load_fnn(output_root / reports["fnn"]["model_file"]).to(device)
    lstm = load_lstm(output_root / reports["lstm"]["model_file"]).to(device)
    accessor = GlobalSparseAccessor(data)
    start, stop = sequence_index.offsets[partition]
    endpoints = np.arange(start, stop, dtype=np.int64)
    xgb_probability = np.asarray(
        xgboost.get_booster().inplace_predict(static_matrix), dtype=np.float64
    )
    fnn_probability = predict_fnn(fnn, static_matrix, device=device)
    lstm_probability = predict_lstm(
        lstm,
        accessor,
        sequence_index,
        endpoints,
        sequence_length=int(reports["lstm"]["sequence_length"]),
        device=device,
    )
    return {
        "xgboost": xgb_probability,
        "fnn": fnn_probability,
        "lstm": lstm_probability,
    }


def _save_probabilities_atomic(
    arrays: Mapping[str, np.ndarray], destination: Path, *, overwrite: bool
) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"prediction output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}{destination.suffix}"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _hybrid_config_content(
    best: Mapping[str, Any],
    selection,
    reports: Mapping[str, Mapping[str, Any]],
    data: ModelDataset,
    sequence_index: SequenceIndex,
) -> dict[str, Any]:
    return {
        "artifact_type": "fraud_hybrid_config",
        "schema_version": HYBRID_CONFIG_SCHEMA_VERSION,
        "component_names": list(COMPONENT_NAMES),
        "blend_space": best["blend_space"],
        "weights": best["weights"],
        "decision_threshold": float(selection.threshold),
        "threshold_objective": f"F{selection.beta:g}",
        "model_data_manifest_sha256": data.metadata["payload_sha256"],
        "feature_schema_sha256": data.metadata["feature_schema_sha256"],
        "sequence_index_manifest_sha256": sequence_index.metadata["payload_sha256"],
        "component_reports": {
            name: {
                "payload_sha256": reports[name]["payload_sha256"],
                "model_sha256": reports[name]["model_sha256"],
                "model_file": reports[name]["model_file"],
            }
            for name in COMPONENT_NAMES
        },
    }


def run_hybrid_search(
    data: ModelDataset,
    sequence_index: SequenceIndex,
    output_dir: str | Path,
    *,
    n_weight_samples: int = 32,
    minimum_component_weight: float = 0.05,
    random_state: int = 42,
    threshold_beta: float = 2.0,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Optimize hybrid fusion on validation and score holdout exactly once."""

    output_root = Path(output_dir)
    config_path = output_root / "hybrid_config.json"
    report_path = output_root / "hybrid_report.json"
    probabilities_path = output_root / "hybrid_probabilities.npz"
    if not overwrite:
        existing = [
            str(path)
            for path in (config_path, report_path, probabilities_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"hybrid outputs already exist: {', '.join(existing)}")
    reports = _component_reports(output_root)
    started = time.perf_counter()
    validation_components = collect_component_probabilities(
        data,
        sequence_index,
        output_root,
        "validation",
        device_name=device_name,
    )
    best, candidates, validation_probability = optimize_hybrid_blend(
        data.validation_target,
        validation_components,
        n_weight_samples=n_weight_samples,
        minimum_component_weight=minimum_component_weight,
        random_state=random_state,
        threshold_beta=threshold_beta,
    )
    selection = select_fbeta_threshold(
        data.validation_target, validation_probability, beta=threshold_beta
    )
    holdout_components = collect_component_probabilities(
        data,
        sequence_index,
        output_root,
        "holdout",
        device_name=device_name,
    )
    holdout_probability = blend_probabilities(
        holdout_components,
        best["weights"],
        blend_space=best["blend_space"],
    )
    config_content = _hybrid_config_content(
        best, selection, reports, data, sequence_index
    )
    config_document = {
        **config_content,
        "payload_sha256": json_digest(config_content),
    }
    atomic_write_json(config_document, config_path, overwrite=overwrite)
    prediction_arrays: dict[str, np.ndarray] = {
        **{
            f"validation_{name}": values.astype(np.float32)
            for name, values in validation_components.items()
        },
        "validation_hybrid": validation_probability.astype(np.float32),
        **{
            f"holdout_{name}": values.astype(np.float32)
            for name, values in holdout_components.items()
        },
        "holdout_hybrid": holdout_probability.astype(np.float32),
    }
    _save_probabilities_atomic(
        prediction_arrays, probabilities_path, overwrite=overwrite
    )
    content: dict[str, Any] = {
        "artifact_type": "fraud_classifier_report",
        "schema_version": 1,
        "model_name": "hybrid",
        "architecture": "constrained_three_component_probability_fusion",
        "selection_metric": "validation_average_precision",
        "random_state": int(random_state),
        "search_iterations": len(candidates),
        "minimum_component_weight": float(minimum_component_weight),
        "threshold_source": "chronological_validation",
        "threshold_objective": f"F{threshold_beta:g}",
        "holdout_usage": "winner_only_with_frozen_validation_threshold",
        "model_file": config_path.name,
        "model_sha256": sha256_file(config_path),
        "prediction_file": probabilities_path.name,
        "prediction_sha256": sha256_file(probabilities_path),
        "model_data_manifest_sha256": data.metadata["payload_sha256"],
        "feature_schema_sha256": data.metadata["feature_schema_sha256"],
        "sequence_index_manifest_sha256": sequence_index.metadata["payload_sha256"],
        "component_report_sha256": {
            name: reports[name]["payload_sha256"] for name in COMPONENT_NAMES
        },
        "dependencies": runtime_dependencies(
            "numpy", "scipy", "scikit-learn", "torch", "xgboost"
        ),
        "device": str(resolve_device(device_name)),
        "fit_seconds": time.perf_counter() - started,
        "best_candidate_index": int(best["candidate_index"]),
        "best_parameters": {
            "blend_space": best["blend_space"],
            "weights": best["weights"],
        },
        "threshold_selection": threshold_selection_dict(selection),
        "validation_metrics": evaluate_probabilities(
            data.validation_target,
            validation_probability,
            threshold=selection.threshold,
        ),
        "holdout_metrics": evaluate_probabilities(
            data.holdout_target,
            holdout_probability,
            threshold=selection.threshold,
        ),
        "candidate_results": candidates,
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, report_path, overwrite=overwrite)
    return document


def load_hybrid_config(path: str | Path) -> dict[str, Any]:
    """Load and verify a hybrid fusion configuration."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("hybrid config root must be an object")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError("hybrid config digest does not match its content")
    if content.get("artifact_type") != "fraud_hybrid_config":
        raise ValueError("unexpected hybrid config artifact type")
    if content.get("schema_version") != HYBRID_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid config schema version")
    _normalized_weights(content["weights"])
    return document


class HybridInferenceEngine:
    """Warm component bundle for prepared static and sequential feature scoring."""

    def __init__(
        self,
        xgboost,
        fnn,
        lstm,
        config: Mapping[str, Any],
        *,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        threshold = float(config["decision_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        _normalized_weights(config["weights"])
        if config["blend_space"] not in {"probability", "logit"}:
            raise ValueError("unsupported hybrid blend space")
        self.xgboost = xgboost
        self.fnn = fnn.to(device).eval()
        self.lstm = lstm.to(device).eval()
        self.config = dict(config)
        self.sequence_length = int(sequence_length)
        self.decision_threshold = threshold
        self.device = device

    @classmethod
    def load(
        cls, output_dir: str | Path, *, device_name: str = "auto"
    ) -> "HybridInferenceEngine":
        root = Path(output_dir)
        config = load_hybrid_config(root / "hybrid_config.json")
        reports = _component_reports(root)
        for name in COMPONENT_NAMES:
            registered = config["component_reports"][name]
            if (
                registered["payload_sha256"] != reports[name]["payload_sha256"]
                or registered["model_sha256"] != reports[name]["model_sha256"]
            ):
                raise ValueError(f"{name} component differs from hybrid config")
        device = resolve_device(device_name)
        return cls(
            _load_xgboost(root / reports["xgboost"]["model_file"]),
            load_fnn(root / reports["fnn"]["model_file"]),
            load_lstm(root / reports["lstm"]["model_file"]),
            config,
            sequence_length=int(reports["lstm"]["sequence_length"]),
            device=device,
        )

    def score_prepared(
        self,
        static_features: sparse.spmatrix | np.ndarray,
        sequence_features: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Score already transformed rows with all components and their fusion."""

        if sparse.issparse(static_features):
            rows, columns = static_features.shape
            static_dense = static_features.toarray().astype(np.float32, copy=False)
            static_xgb = static_features
        else:
            static_dense = np.asarray(static_features, dtype=np.float32)
            if static_dense.ndim != 2:
                raise ValueError("static_features must be two-dimensional")
            rows, columns = static_dense.shape
            static_xgb = static_dense
        sequences = np.asarray(sequence_features, dtype=np.float32)
        lengths = np.asarray(sequence_lengths, dtype=np.int64)
        if (
            sequences.shape != (rows, self.sequence_length, columns)
            or lengths.shape != (rows,)
            or (lengths <= 0).any()
            or (lengths > self.sequence_length).any()
        ):
            raise ValueError("prepared sequence shapes or lengths are invalid")
        xgb_probability = np.asarray(
            self.xgboost.get_booster().inplace_predict(static_xgb),
            dtype=np.float64,
        )
        static_tensor = torch.from_numpy(
            np.ascontiguousarray(static_dense, dtype=np.float32)
        ).to(self.device)
        sequence_tensor = torch.from_numpy(
            np.ascontiguousarray(sequences, dtype=np.float32)
        ).to(self.device)
        length_tensor = torch.from_numpy(lengths)
        with torch.inference_mode():
            fnn_probability = (
                torch.sigmoid(self.fnn(static_tensor)).cpu().numpy().astype(np.float64)
            )
            lstm_probability = (
                torch.sigmoid(self.lstm(sequence_tensor, length_tensor))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        components = {
            "xgboost": xgb_probability,
            "fnn": fnn_probability,
            "lstm": lstm_probability,
        }
        hybrid_probability = blend_probabilities(
            components,
            self.config["weights"],
            blend_space=self.config["blend_space"],
        )
        return {**components, "hybrid": hybrid_probability}

    def predict_prepared(
        self,
        static_features: sparse.spmatrix | np.ndarray,
        sequence_features: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return component scores, fused risk, and frozen-threshold decisions."""

        scores = self.score_prepared(
            static_features,
            sequence_features,
            sequence_lengths,
        )
        return {
            "xgboost_probability": scores["xgboost"],
            "fnn_probability": scores["fnn"],
            "lstm_probability": scores["lstm"],
            "fraud_probability": scores["hybrid"],
            "fraud_flag": scores["hybrid"] >= self.decision_threshold,
        }


def benchmark_hybrid_engine(
    engine: HybridInferenceEngine,
    static_features: np.ndarray,
    sequence_features: np.ndarray,
    sequence_lengths: np.ndarray,
    *,
    batch_sizes: Sequence[int] = (1, 32, 256),
    warmups: int = 3,
    repeats: int = 20,
) -> dict[str, Any]:
    """Measure warm prepared-feature latency without model loading or disk I/O."""

    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0 or batch_size > len(static_features):
            raise ValueError("benchmark batch size is out of range")
        static = static_features[:batch_size]
        sequences = sequence_features[:batch_size]
        lengths = sequence_lengths[:batch_size]
        for _ in range(warmups):
            engine.score_prepared(static, sequences, lengths)
        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            engine.score_prepared(static, sequences, lengths)
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
        values = np.asarray(durations, dtype=np.float64)
        p50 = float(np.quantile(values, 0.50))
        rows.append(
            {
                "batch_size": int(batch_size),
                "repeats": int(repeats),
                "latency_ms_p50": p50,
                "latency_ms_p95": float(np.quantile(values, 0.95)),
                "latency_ms_p99": float(np.quantile(values, 0.99)),
                "per_transaction_ms_p50": p50 / batch_size,
                "throughput_transactions_per_second_p50": float(
                    batch_size / (p50 / 1000.0)
                ),
            }
        )
    return {
        "measurement_scope": "warm_models_pretransformed_static_and_sequence_features",
        "device": str(engine.device),
        "torch_inference_mode": True,
        "model_loading_included": False,
        "feature_engineering_included": False,
        "batches": rows,
    }


def _latency_rows(
    scorer,
    *,
    batch_sizes: Sequence[int],
    available_rows: int,
    warmups: int,
    repeats: int,
    synchronize,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0 or batch_size > available_rows:
            raise ValueError("benchmark batch size is out of range")
        for _ in range(warmups):
            scorer(batch_size)
        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            scorer(batch_size)
            synchronize()
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
        values = np.asarray(durations, dtype=np.float64)
        p50 = float(np.quantile(values, 0.50))
        rows.append(
            {
                "batch_size": int(batch_size),
                "repeats": int(repeats),
                "latency_ms_p50": p50,
                "latency_ms_p95": float(np.quantile(values, 0.95)),
                "latency_ms_p99": float(np.quantile(values, 0.99)),
                "per_transaction_ms_p50": p50 / batch_size,
                "throughput_transactions_per_second_p50": float(
                    batch_size / (p50 / 1000.0)
                ),
            }
        )
    return rows


def benchmark_model_suite(
    engine: HybridInferenceEngine,
    output_dir: str | Path,
    static_features: np.ndarray,
    sequence_features: np.ndarray,
    sequence_lengths: np.ndarray,
    *,
    batch_sizes: Sequence[int] = (1, 32, 256),
    warmups: int = 3,
    repeats: int = 20,
) -> dict[str, Any]:
    """Measure every registered model with warm, aligned prepared features."""

    import joblib

    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    root = Path(output_dir)
    logistic_report = load_model_report(root / "logistic_regression_report.json")
    forest_report = load_model_report(root / "random_forest_report.json")
    logistic = joblib.load(root / logistic_report["model_file"])
    forest = joblib.load(root / forest_report["model_file"])
    static = np.asarray(static_features, dtype=np.float32)
    sequences = np.asarray(sequence_features, dtype=np.float32)
    lengths = np.asarray(sequence_lengths, dtype=np.int64)
    if (
        static.ndim != 2
        or sequences.shape
        != (len(static), engine.sequence_length, static.shape[1])
        or lengths.shape != (len(static),)
    ):
        raise ValueError("benchmark inputs are not aligned")

    def synchronize() -> None:
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)

    def fnn_score(size: int) -> np.ndarray:
        values = torch.from_numpy(np.ascontiguousarray(static[:size])).to(engine.device)
        with torch.inference_mode():
            return torch.sigmoid(engine.fnn(values)).cpu().numpy()

    def lstm_score(size: int) -> np.ndarray:
        values = torch.from_numpy(
            np.ascontiguousarray(sequences[:size])
        ).to(engine.device)
        length_values = torch.from_numpy(lengths[:size])
        with torch.inference_mode():
            return torch.sigmoid(engine.lstm(values, length_values)).cpu().numpy()

    scorers = {
        "logistic_regression": lambda size: logistic.predict_proba(static[:size])[:, 1],
        "random_forest": lambda size: forest.predict_proba(static[:size])[:, 1],
        "xgboost": lambda size: engine.xgboost.get_booster().inplace_predict(
            static[:size]
        ),
        "fnn": fnn_score,
        "lstm": lstm_score,
        "hybrid": lambda size: engine.score_prepared(
            static[:size], sequences[:size], lengths[:size]
        )["hybrid"],
    }
    return {
        "artifact_type": "fraud_model_latency_benchmark",
        "schema_version": 1,
        "measurement_scope": "warm_models_pretransformed_static_and_sequence_features",
        "device": str(engine.device),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_inference_mode": True,
        "model_loading_included": False,
        "feature_engineering_included": False,
        "warmups": int(warmups),
        "dependencies": runtime_dependencies(
            "numpy", "scikit-learn", "torch", "xgboost"
        ),
        "models": {
            name: _latency_rows(
                scorer,
                batch_sizes=batch_sizes,
                available_rows=len(static),
                warmups=warmups,
                repeats=repeats,
                synchronize=synchronize,
            )
            for name, scorer in scorers.items()
        },
    }


def build_operational_tradeoff_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    latency_report: Mapping[str, Any],
    output_dir: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Combine model quality, context, latency, and artifact footprint."""

    root = Path(output_dir)
    context = {
        "logistic_regression": "static",
        "random_forest": "static",
        "xgboost": "static",
        "fnn": "static_neural",
        "lstm": "causal_sequence",
        "hybrid": "static_and_causal_sequence",
    }
    rows: list[dict[str, Any]] = []
    component_bytes = sum(
        (root / reports[name]["model_file"]).stat().st_size
        for name in COMPONENT_NAMES
    )
    for name, report in reports.items():
        latency_by_batch = {
            int(item["batch_size"]): item
            for item in latency_report["models"][name]
        }
        model_path = root / report["model_file"]
        artifact_bytes = int(model_path.stat().st_size)
        effective_bytes = (
            artifact_bytes + component_bytes if name == "hybrid" else artifact_bytes
        )
        holdout = report["holdout_metrics"]
        rows.append(
            {
                "model_name": name,
                "input_context": context[name],
                "holdout_average_precision": holdout["ranking"][
                    "average_precision"
                ],
                "holdout_pr_auc": holdout["ranking"]["pr_auc_trapezoidal"],
                "holdout_recall": holdout["operating_point"]["recall"],
                "holdout_precision": holdout["operating_point"]["precision"],
                "holdout_false_positive_rate": holdout["operating_point"][
                    "false_positive_rate"
                ],
                "holdout_alert_rate": holdout["operating_point"]["alert_rate"],
                "threshold": report["threshold_selection"]["threshold"],
                "artifact_bytes": artifact_bytes,
                "effective_loaded_artifact_bytes": effective_bytes,
                "batch_1_latency_ms_p50": latency_by_batch[1]["latency_ms_p50"],
                "batch_1_latency_ms_p95": latency_by_batch[1]["latency_ms_p95"],
                "batch_256_per_transaction_ms_p50": latency_by_batch[256][
                    "per_transaction_ms_p50"
                ],
                "batch_256_throughput_per_second_p50": latency_by_batch[256][
                    "throughput_transactions_per_second_p50"
                ],
            }
        )
    rows.sort(
        key=lambda item: float(item["holdout_average_precision"]), reverse=True
    )
    content: dict[str, Any] = {
        "artifact_type": "fraud_model_operational_tradeoff_matrix",
        "schema_version": 1,
        "latency_measurement_scope": latency_report["measurement_scope"],
        "device": latency_report["device"],
        "models": rows,
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, destination, overwrite=overwrite)
    return document
