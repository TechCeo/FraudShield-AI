"""Shared deterministic training and persistence utilities for neural classifiers."""

from __future__ import annotations

import copy
import os
import random
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from scipy import sparse


def configure_torch(random_state: int) -> None:
    """Configure repeatable Python, NumPy, and PyTorch random state."""

    seed = int(random_state)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit CPU/CUDA request or select the available accelerator."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(
        "cuda" if (requested == "cuda" or requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def sampled_training_indices(
    target: np.ndarray,
    *,
    negative_ratio: int | None,
    random_state: int,
) -> np.ndarray:
    """Return all fraud endpoints plus a deterministic training-only negative sample."""

    values = np.asarray(target)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("target must be a one-dimensional binary array")
    positive = np.flatnonzero(values == 1)
    negative = np.flatnonzero(values == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("target must contain both classes")
    generator = np.random.default_rng(int(random_state))
    if negative_ratio is None:
        selected_negative = negative
    else:
        if isinstance(negative_ratio, bool) or negative_ratio <= 0:
            raise ValueError("negative_ratio must be a positive integer or None")
        count = min(negative.size, int(negative_ratio) * positive.size)
        selected_negative = generator.choice(negative, size=count, replace=False)
    selected = np.concatenate([positive, selected_negative]).astype(np.int64, copy=False)
    generator.shuffle(selected)
    return selected


def iter_index_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield contiguous views over an already ordered index array."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def csr_tensor(
    matrix: sparse.csr_matrix, indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Materialize one sparse row batch as a contiguous float32 tensor."""

    dense = np.asarray(matrix[indices].toarray(), dtype=np.float32, order="C")
    return torch.from_numpy(dense).to(device=device, non_blocking=device.type == "cuda")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state dictionary onto CPU for stable candidate retention."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_torch_artifact(
    state_dict: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist tensor weights and a primitive model configuration."""

    output = Path(destination)
    if output.exists() and not overwrite:
        raise FileExistsError(f"model output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}{output.suffix}")
    payload = {
        "state_dict": copy.deepcopy(dict(state_dict)),
        "model_config": copy.deepcopy(dict(model_config)),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    """Load a local tensor artifact with PyTorch's restricted weights loader."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict", "model_config"}:
        raise ValueError("unexpected neural model artifact structure")
    if not isinstance(payload["state_dict"], dict) or not isinstance(
        payload["model_config"], dict
    ):
        raise ValueError("neural model artifact fields are invalid")
    return payload
