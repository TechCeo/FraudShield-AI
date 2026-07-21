"""Shared integrity, serialization, and runtime helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return an uppercase SHA-256 digest without loading a file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def json_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable uppercase SHA-256 digest for a JSON-compatible mapping."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def atomic_write_json(
    payload: Mapping[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    """Serialize JSON through an atomic replacement with write-once defaults."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def runtime_dependencies(*packages: str) -> dict[str, str]:
    """Return the Python version and installed versions of named distributions."""

    dependencies = {"python": platform.python_version()}
    for package in packages:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    return dependencies
