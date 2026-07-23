"""Causal LSTM classifier over chronological per-card transaction sequences."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from ..utils import atomic_write_json, json_digest, runtime_dependencies, sha256_file
from .data import ModelDataset
from .deep_common import (
    configure_torch,
    cpu_state_dict,
    iter_index_batches,
    load_torch_artifact,
    resolve_device,
    sampled_training_indices,
    save_torch_artifact,
)
from .evaluation import (
    evaluate_probabilities,
    select_fbeta_threshold,
    threshold_selection_dict,
)
from .sequences import GlobalSparseAccessor, SequenceIndex, sequence_tensor_batch

LOGGER = logging.getLogger(__name__)


class CausalFraudLSTM(torch.nn.Module):
    """Unidirectional recurrent classifier returning one logit per endpoint."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 48,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if min(input_dim, projection_dim, hidden_size, num_layers) <= 0:
            raise ValueError("model dimensions and layer count must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(projection_dim)),
            torch.nn.GELU(),
            torch.nn.LayerNorm(int(projection_dim)),
        )
        self.recurrent = torch.nn.LSTM(
            input_size=int(projection_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=False,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_size), 1),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.recurrent(packed)
        return self.output(hidden[-1]).squeeze(-1)


@dataclass
class _CandidateFit:
    state_dict: dict[str, torch.Tensor]
    probabilities: np.ndarray
    best_epoch: int
    fit_seconds: float
    history: list[dict[str, float | int]]
    sampled_rows: int
    sampled_target_counts: dict[str, int]
    pos_weight: float


def sample_lstm_parameters(
    *, n_iter: int, random_state: int
) -> list[dict[str, Any]]:
    """Draw deterministic LSTM configurations from a bounded CPU search space."""

    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    space = {
        "sequence_length": [8, 12, 16],
        "projection_dim": [32, 48, 64],
        "hidden_size": [32, 48, 64],
        "num_layers": [1, 2],
        "dropout": [0.10, 0.20],
        "learning_rate": [3e-4, 7e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [1024, 2048],
        "negative_ratio": [20],
        "pos_weight_multiplier": [0.25, 0.5, 1.0],
    }
    return [
        dict(parameters)
        for parameters in ParameterSampler(
            space, n_iter=n_iter, random_state=int(random_state)
        )
    ]


def predict_lstm(
    model: CausalFraudLSTM,
    accessor: GlobalSparseAccessor,
    sequence_index: SequenceIndex,
    current_indices: np.ndarray,
    *,
    sequence_length: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return probabilities for ordered global transaction endpoints."""

    current = np.asarray(current_indices, dtype=np.int64)
    output = np.empty(len(current), dtype=np.float64)
    positions = np.arange(len(current), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for batch_positions in iter_index_batches(positions, batch_size):
            endpoints = current[batch_positions]
            features, lengths = sequence_tensor_batch(
                accessor,
                endpoints,
                sequence_index.previous,
                sequence_length,
                device,
            )
            probabilities = torch.sigmoid(model(features, lengths)).cpu().numpy()
            output[batch_positions] = probabilities.astype(np.float64, copy=False)
    return output


def _fit_candidate(
    data: ModelDataset,
    sequence_index: SequenceIndex,
    accessor: GlobalSparseAccessor,
    parameters: Mapping[str, Any],
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    model_seed: int,
    sample_seed: int,
) -> _CandidateFit:
    configure_torch(model_seed)
    model = CausalFraudLSTM(
        input_dim=data.train_features.shape[1],
        projection_dim=int(parameters["projection_dim"]),
        hidden_size=int(parameters["hidden_size"]),
        num_layers=int(parameters["num_layers"]),
        dropout=float(parameters["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    initial_indices = sampled_training_indices(
        data.train_target,
        negative_ratio=int(parameters["negative_ratio"]),
        random_state=sample_seed,
    )
    initial_target = data.train_target[initial_indices]
    positive = int(np.count_nonzero(initial_target == 1))
    negative = int(np.count_nonzero(initial_target == 0))
    pos_weight_value = (negative / positive) * float(
        parameters["pos_weight_multiplier"]
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    validation_start, validation_stop = sequence_index.offsets["validation"]
    validation_endpoints = np.arange(
        validation_start, validation_stop, dtype=np.int64
    )
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_probabilities: np.ndarray | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        endpoints = sampled_training_indices(
            data.train_target,
            negative_ratio=int(parameters["negative_ratio"]),
            random_state=sample_seed + epoch - 1,
        )
        model.train()
        loss_sum = 0.0
        rows_seen = 0
        for batch_endpoints in iter_index_batches(
            endpoints, int(parameters["batch_size"])
        ):
            features, lengths = sequence_tensor_batch(
                accessor,
                batch_endpoints,
                sequence_index.previous,
                int(parameters["sequence_length"]),
                device,
            )
            target = torch.from_numpy(
                data.train_target[batch_endpoints].astype(np.float32, copy=False)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, lengths)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_endpoints)
            rows_seen += len(batch_endpoints)
        validation_probabilities = predict_lstm(
            model,
            accessor,
            sequence_index,
            validation_endpoints,
            sequence_length=int(parameters["sequence_length"]),
            device=device,
        )
        validation_ap = float(
            average_precision_score(
                data.validation_target, validation_probabilities
            )
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": loss_sum / rows_seen,
            "validation_average_precision": validation_ap,
        }
        history.append(epoch_record)
        LOGGER.info(
            "lstm epoch %d loss=%.6f validation_AP=%.6f",
            epoch,
            epoch_record["training_loss"],
            validation_ap,
        )
        if validation_ap > best_score + 1e-7:
            best_score = validation_ap
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_probabilities = validation_probabilities.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None or best_probabilities is None:
        raise RuntimeError("LSTM training did not produce a validation result")
    return _CandidateFit(
        state_dict=best_state,
        probabilities=best_probabilities,
        best_epoch=best_epoch,
        fit_seconds=time.perf_counter() - started,
        history=history,
        sampled_rows=len(initial_indices),
        sampled_target_counts={"0": negative, "1": positive},
        pos_weight=pos_weight_value,
    )


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(result["validation_metrics"]["ranking"]["average_precision"]),
        float(result["threshold_selection"]["recall"]),
        float(result["threshold_selection"]["precision"]),
    )


def run_lstm_search(
    data: ModelDataset,
    sequence_index: SequenceIndex,
    output_dir: str | Path,
    *,
    n_iter: int = 3,
    max_epochs: int = 5,
    patience: int = 2,
    random_state: int = 42,
    threshold_beta: float = 2.0,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Optimize, persist, and holdout-score the causal recurrent classifier."""

    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if sequence_index.metadata["model_data_manifest_sha256"] != data.metadata[
        "payload_sha256"
    ]:
        raise ValueError("sequence index and model data have different lineage")
    output_root = Path(output_dir)
    model_path = output_root / "lstm.pt"
    report_path = output_root / "lstm_report.json"
    if not overwrite:
        existing = [str(path) for path in (model_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"neural outputs already exist: {', '.join(existing)}")
    device = resolve_device(device_name)
    accessor = GlobalSparseAccessor(data)
    parameters = sample_lstm_parameters(n_iter=n_iter, random_state=random_state)
    results: list[dict[str, Any]] = []
    winner_state: dict[str, torch.Tensor] | None = None
    winner_parameters: dict[str, Any] | None = None
    winner_index = -1
    winner_fit: _CandidateFit | None = None
    for index, candidate in enumerate(parameters):
        LOGGER.info("fitting lstm candidate %d/%d", index + 1, len(parameters))
        fitted = _fit_candidate(
            data,
            sequence_index,
            accessor,
            candidate,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            model_seed=random_state + 10_000 * index,
            sample_seed=random_state,
        )
        selection = select_fbeta_threshold(
            data.validation_target, fitted.probabilities, beta=threshold_beta
        )
        result: dict[str, Any] = {
            "candidate_index": index,
            "parameters": candidate,
            "fit_seconds": fitted.fit_seconds,
            "best_epoch": fitted.best_epoch,
            "epochs_executed": len(fitted.history),
            "epoch_history": fitted.history,
            "sampled_training_rows": fitted.sampled_rows,
            "sampled_target_counts": fitted.sampled_target_counts,
            "positive_loss_weight": fitted.pos_weight,
            "threshold_selection": threshold_selection_dict(selection),
            "validation_metrics": evaluate_probabilities(
                data.validation_target,
                fitted.probabilities,
                threshold=selection.threshold,
            ),
        }
        results.append(result)
        if winner_index < 0 or _rank_key(result) > _rank_key(results[winner_index]):
            winner_index = index
            winner_state = fitted.state_dict
            winner_parameters = dict(candidate)
            winner_fit = fitted
        LOGGER.info(
            "lstm candidate %d validation AP=%.6f recall=%.6f precision=%.6f",
            index + 1,
            result["validation_metrics"]["ranking"]["average_precision"],
            selection.recall,
            selection.precision,
        )
        if winner_fit is not fitted:
            del fitted
        gc.collect()
    if winner_state is None or winner_parameters is None or winner_fit is None:
        raise RuntimeError("LSTM search did not select a candidate")
    model_config = {
        "input_dim": int(data.train_features.shape[1]),
        "projection_dim": int(winner_parameters["projection_dim"]),
        "hidden_size": int(winner_parameters["hidden_size"]),
        "num_layers": int(winner_parameters["num_layers"]),
        "dropout": float(winner_parameters["dropout"]),
    }
    model = CausalFraudLSTM(**model_config).to(device)
    model.load_state_dict(winner_state)
    validation_start, validation_stop = sequence_index.offsets["validation"]
    holdout_start, holdout_stop = sequence_index.offsets["holdout"]
    validation_probabilities = predict_lstm(
        model,
        accessor,
        sequence_index,
        np.arange(validation_start, validation_stop, dtype=np.int64),
        sequence_length=int(winner_parameters["sequence_length"]),
        device=device,
    )
    selection = select_fbeta_threshold(
        data.validation_target, validation_probabilities, beta=threshold_beta
    )
    holdout_probabilities = predict_lstm(
        model,
        accessor,
        sequence_index,
        np.arange(holdout_start, holdout_stop, dtype=np.int64),
        sequence_length=int(winner_parameters["sequence_length"]),
        device=device,
    )
    save_torch_artifact(
        winner_state, model_config, model_path, overwrite=overwrite
    )
    content: dict[str, Any] = {
        "artifact_type": "fraud_classifier_report",
        "schema_version": 1,
        "model_name": "lstm",
        "architecture": "unidirectional_causal_lstm",
        "selection_metric": "validation_average_precision",
        "random_state": int(random_state),
        "search_iterations": len(parameters),
        "threshold_source": "chronological_validation",
        "threshold_objective": f"F{threshold_beta:g}",
        "holdout_usage": "winner_only_with_frozen_validation_threshold",
        "training_imbalance_control": "all_fraud_plus_epoch_seeded_training_only_negative_endpoint_sampling",
        "sequence_contract": "oldest_to_current_same_card_features_without_future_rows_or_labels",
        "sequence_length": int(winner_parameters["sequence_length"]),
        "sequence_index_manifest_sha256": sequence_index.metadata["payload_sha256"],
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "model_config": model_config,
        "model_data_manifest_sha256": data.metadata["payload_sha256"],
        "feature_schema_sha256": data.metadata["feature_schema_sha256"],
        "dependencies": runtime_dependencies(
            "numpy", "scipy", "scikit-learn", "torch"
        ),
        "device": str(device),
        "best_candidate_index": winner_index,
        "best_parameters": winner_parameters,
        "best_epoch": winner_fit.best_epoch,
        "final_fit_seconds": winner_fit.fit_seconds,
        "threshold_selection": threshold_selection_dict(selection),
        "validation_metrics": evaluate_probabilities(
            data.validation_target,
            validation_probabilities,
            threshold=selection.threshold,
        ),
        "holdout_metrics": evaluate_probabilities(
            data.holdout_target,
            holdout_probabilities,
            threshold=selection.threshold,
        ),
        "candidate_results": results,
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, report_path, overwrite=overwrite)
    return document


def load_lstm(path: str | Path) -> CausalFraudLSTM:
    """Load a local recurrent tensor artifact with restricted deserialization."""

    payload = load_torch_artifact(path)
    model = CausalFraudLSTM(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
