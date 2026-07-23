"""Label-independent feature and prediction drift monitoring."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import ks_2samp, wasserstein_distance

from ..utils import atomic_write_json, json_digest, runtime_dependencies

DRIFT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DriftConfig:
    """Sampling, binning, and configured alert thresholds."""

    feature_bins: int = 10
    feature_sample_rows: int = 100_000
    prediction_sample_rows: int = 10_000
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    epsilon: float = 1e-6
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.feature_bins < 2:
            raise ValueError("feature_bins must be at least 2")
        if self.feature_sample_rows <= 0 or self.prediction_sample_rows <= 0:
            raise ValueError("drift sample sizes must be positive")
        if not 0.0 < self.psi_warning < self.psi_critical:
            raise ValueError("PSI thresholds must be positive and ordered")
        if not 0.0 < self.epsilon < 0.01:
            raise ValueError("epsilon must be in (0, 0.01)")


def _sample_indices(rows: int, limit: int, random_state: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("drift input must contain rows")
    if rows <= limit:
        return np.arange(rows, dtype=np.int64)
    generator = np.random.default_rng(int(random_state))
    return np.sort(generator.choice(rows, size=limit, replace=False))


def _sample_matrix(
    matrix: sparse.spmatrix | np.ndarray, limit: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(matrix.shape) != 2:
        raise ValueError("drift matrix must be two-dimensional")
    indices = _sample_indices(matrix.shape[0], limit, random_state)
    if sparse.issparse(matrix):
        values = matrix[indices].toarray()
    else:
        values = np.asarray(matrix)[indices]
    dense = np.asarray(values, dtype=np.float64)
    if not np.isfinite(dense).all():
        raise ValueError("drift matrix must contain finite values")
    return dense, indices


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    internal = np.unique(np.quantile(values, quantiles))
    return np.concatenate(([-np.inf], internal, [np.inf])).astype(np.float64)


def _proportions(values: np.ndarray, edges: np.ndarray, epsilon: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    proportions = counts.astype(np.float64) / max(1, counts.sum())
    proportions = np.clip(proportions, epsilon, None)
    return proportions / proportions.sum()


def population_stability_index(
    reference_proportions: np.ndarray,
    current_proportions: np.ndarray,
) -> float:
    """Return PSI for aligned, positive discrete probability vectors."""

    reference = np.asarray(reference_proportions, dtype=np.float64)
    current = np.asarray(current_proportions, dtype=np.float64)
    if (
        reference.shape != current.shape
        or reference.ndim != 1
        or (reference <= 0.0).any()
        or (current <= 0.0).any()
    ):
        raise ValueError("PSI inputs must be aligned positive vectors")
    return float(np.sum((current - reference) * np.log(current / reference)))


def _status(value: float, config: DriftConfig) -> str:
    if value >= config.psi_critical:
        return "critical"
    if value >= config.psi_warning:
        return "warning"
    return "stable"


class DriftDetector:
    """Quantile-binned feature PSI plus prediction distribution monitoring."""

    def __init__(self, config: DriftConfig = DriftConfig()) -> None:
        self.config = config
        self.feature_names: list[str] = []
        self.feature_reference: dict[str, dict[str, Any]] = {}
        self.prediction_reference: dict[str, Any] = {}
        self.reference_context: dict[str, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        feature_matrix: sparse.spmatrix | np.ndarray,
        feature_names: Sequence[str],
        prediction_probabilities: Sequence[float] | np.ndarray,
        *,
        reference_context: Mapping[str, Any],
    ) -> "DriftDetector":
        """Fit unlabeled reference distributions from registered source windows."""

        names = [str(name) for name in feature_names]
        if len(names) != feature_matrix.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature names must be unique and align with matrix columns")
        feature_sample, feature_indices = _sample_matrix(
            feature_matrix,
            self.config.feature_sample_rows,
            self.config.random_state,
        )
        probabilities = np.asarray(prediction_probabilities, dtype=np.float64)
        if (
            probabilities.ndim != 1
            or probabilities.size == 0
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise ValueError("reference predictions must be finite probabilities")
        prediction_indices = _sample_indices(
            len(probabilities),
            self.config.prediction_sample_rows,
            self.config.random_state,
        )
        prediction_sample = probabilities[prediction_indices]
        self.feature_names = names
        self.feature_reference = {}
        for position, name in enumerate(names):
            values = feature_sample[:, position]
            edges = _bin_edges(values, self.config.feature_bins)
            self.feature_reference[name] = {
                "edges": edges.tolist(),
                "proportions": _proportions(
                    values, edges, self.config.epsilon
                ).tolist(),
                "mean": float(values.mean()),
                "scale": float(values.std(ddof=0)) or 1.0,
            }
        prediction_edges = _bin_edges(
            prediction_sample, self.config.feature_bins
        )
        self.prediction_reference = {
            "edges": prediction_edges.tolist(),
            "proportions": _proportions(
                prediction_sample, prediction_edges, self.config.epsilon
            ).tolist(),
            "sample": prediction_sample.tolist(),
        }
        self.reference_context = {
            **dict(reference_context),
            "feature_rows_available": int(feature_matrix.shape[0]),
            "feature_rows_sampled": int(len(feature_indices)),
            "prediction_rows_available": int(len(probabilities)),
            "prediction_rows_sampled": int(len(prediction_indices)),
        }
        self.is_fitted = True
        return self

    def evaluate(
        self,
        feature_matrix: sparse.spmatrix | np.ndarray,
        prediction_probabilities: Sequence[float] | np.ndarray,
        *,
        window_name: str,
    ) -> dict[str, Any]:
        """Evaluate one unlabeled window against frozen reference distributions."""

        if not self.is_fitted:
            raise RuntimeError("drift detector is not fitted")
        if feature_matrix.shape[1] != len(self.feature_names):
            raise ValueError("drift feature width differs from the reference")
        feature_sample, feature_indices = _sample_matrix(
            feature_matrix,
            self.config.feature_sample_rows,
            self.config.random_state,
        )
        feature_results: list[dict[str, Any]] = []
        for position, name in enumerate(self.feature_names):
            reference = self.feature_reference[name]
            edges = np.asarray(reference["edges"], dtype=np.float64)
            reference_proportions = np.asarray(
                reference["proportions"], dtype=np.float64
            )
            values = feature_sample[:, position]
            current_proportions = _proportions(
                values, edges, self.config.epsilon
            )
            psi = population_stability_index(
                reference_proportions, current_proportions
            )
            normalized_mean_shift = (
                float(values.mean()) - float(reference["mean"])
            ) / float(reference["scale"])
            feature_results.append(
                {
                    "feature": name,
                    "psi": psi,
                    "status": _status(psi, self.config),
                    "normalized_mean_shift": normalized_mean_shift,
                }
            )
        feature_results.sort(key=lambda value: float(value["psi"]), reverse=True)
        probabilities = np.asarray(prediction_probabilities, dtype=np.float64)
        if (
            probabilities.ndim != 1
            or probabilities.size == 0
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise ValueError("current predictions must be finite probabilities")
        prediction_indices = _sample_indices(
            len(probabilities),
            self.config.prediction_sample_rows,
            self.config.random_state,
        )
        prediction_sample = probabilities[prediction_indices]
        prediction_edges = np.asarray(
            self.prediction_reference["edges"], dtype=np.float64
        )
        prediction_psi = population_stability_index(
            np.asarray(self.prediction_reference["proportions"], dtype=np.float64),
            _proportions(
                prediction_sample, prediction_edges, self.config.epsilon
            ),
        )
        reference_prediction_sample = np.asarray(
            self.prediction_reference["sample"], dtype=np.float64
        )
        ks_result = ks_2samp(
            reference_prediction_sample,
            prediction_sample,
            method="asymp",
        )
        prediction_status = _status(prediction_psi, self.config)
        statuses = [result["status"] for result in feature_results]
        overall_status = (
            "critical"
            if "critical" in statuses or prediction_status == "critical"
            else "warning"
            if "warning" in statuses or prediction_status == "warning"
            else "stable"
        )
        return {
            "window_name": str(window_name),
            "overall_status": overall_status,
            "feature_rows_available": int(feature_matrix.shape[0]),
            "feature_rows_sampled": int(len(feature_indices)),
            "prediction_rows_available": int(len(probabilities)),
            "prediction_rows_sampled": int(len(prediction_indices)),
            "feature_summary": {
                "stable": int(sum(value == "stable" for value in statuses)),
                "warning": int(sum(value == "warning" for value in statuses)),
                "critical": int(sum(value == "critical" for value in statuses)),
                "mean_psi": float(
                    np.mean([result["psi"] for result in feature_results])
                ),
                "max_psi": float(feature_results[0]["psi"]),
            },
            "top_feature_drift": feature_results[:15],
            "prediction_drift": {
                "psi": prediction_psi,
                "status": prediction_status,
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
                "wasserstein_distance": float(
                    wasserstein_distance(
                        reference_prediction_sample, prediction_sample
                    )
                ),
                "reference_mean": float(reference_prediction_sample.mean()),
                "current_mean": float(prediction_sample.mean()),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("drift detector is not fitted")
        content = {
            "artifact_type": "fraud_drift_detector",
            "schema_version": DRIFT_SCHEMA_VERSION,
            "config": asdict(self.config),
            "feature_names": copy.deepcopy(self.feature_names),
            "feature_reference": copy.deepcopy(self.feature_reference),
            "prediction_reference": copy.deepcopy(self.prediction_reference),
            "reference_context": copy.deepcopy(self.reference_context),
            "dependencies": runtime_dependencies("numpy", "scipy"),
        }
        return {**content, "payload_sha256": json_digest(content)}

    def save(
        self, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        return atomic_write_json(self.to_dict(), path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "DriftDetector":
        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("drift detector root must be an object")
        digest = document.get("payload_sha256")
        content = {
            key: value for key, value in document.items() if key != "payload_sha256"
        }
        if digest != json_digest(content):
            raise ValueError("drift detector digest does not match its content")
        if content.get("artifact_type") != "fraud_drift_detector":
            raise ValueError("unexpected drift artifact type")
        if content.get("schema_version") != DRIFT_SCHEMA_VERSION:
            raise ValueError("unsupported drift detector schema version")
        instance = cls(DriftConfig(**content["config"]))
        instance.feature_names = list(content["feature_names"])
        instance.feature_reference = copy.deepcopy(content["feature_reference"])
        instance.prediction_reference = copy.deepcopy(
            content["prediction_reference"]
        )
        instance.reference_context = copy.deepcopy(content["reference_context"])
        instance.is_fitted = True
        return instance


def shifted_probability_scenario(
    probabilities: Sequence[float] | np.ndarray, *, logit_shift: float
) -> np.ndarray:
    """Apply a controlled log-odds shift for prediction-drift simulation."""

    values = np.asarray(probabilities, dtype=np.float64)
    epsilon = 1e-7
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)) + float(logit_shift)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def run_drift_simulation(
    detector: DriftDetector,
    validation_features: sparse.spmatrix,
    holdout_features: sparse.spmatrix,
    validation_predictions: np.ndarray,
    holdout_predictions: np.ndarray,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate observed windows and an injected feature/score shift."""

    validation = detector.evaluate(
        validation_features,
        validation_predictions,
        window_name="chronological_validation",
    )
    holdout = detector.evaluate(
        holdout_features,
        holdout_predictions,
        window_name="out_of_time_holdout",
    )
    sample_rows = min(detector.config.feature_sample_rows, validation_features.shape[0])
    indices = _sample_indices(
        validation_features.shape[0],
        sample_rows,
        detector.config.random_state,
    )
    shifted_features = validation_features[indices].toarray().astype(np.float64)
    shifted_names = (
        "num__log1p_amt",
        "num__log1p_distance_card_merchant_km",
        "num__log1p_cc_txn_count_prev_24h",
        "num__log1p_cc_amt_sum_prev_24h",
    )
    shifted_columns: list[str] = []
    for name in shifted_names:
        if name in detector.feature_names:
            shifted_features[:, detector.feature_names.index(name)] += 1.5
            shifted_columns.append(name)
    shifted_predictions = shifted_probability_scenario(
        validation_predictions[indices], logit_shift=0.75
    )
    injected = detector.evaluate(
        shifted_features,
        shifted_predictions,
        window_name="injected_behavior_and_score_shift",
    )
    content: dict[str, Any] = {
        "artifact_type": "fraud_drift_simulation_report",
        "schema_version": 1,
        "detector_config": asdict(detector.config),
        "injected_feature_shift_standardized_units": 1.5,
        "injected_prediction_logit_shift": 0.75,
        "injected_features": shifted_columns,
        "scenarios": {
            "validation": validation,
            "holdout": holdout,
            "injected": injected,
        },
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, destination, overwrite=overwrite)
    return document
