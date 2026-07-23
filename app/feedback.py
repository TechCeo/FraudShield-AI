"""Concurrency-safe local reviewer feedback persistence."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

ALLOWED_FEEDBACK = {"confirm_fraud", "false_positive"}


class FeedbackStore:
    """SQLite-backed feedback ledger with idempotent reviewer updates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_feedback (
                    prediction_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    reviewer_label TEXT NOT NULL
                        CHECK (reviewer_label IN ('confirm_fraud', 'false_positive')),
                    fraud_probability REAL NOT NULL
                        CHECK (fraud_probability >= 0.0 AND fraud_probability <= 1.0),
                    decision_threshold REAL NOT NULL
                        CHECK (decision_threshold >= 0.0 AND decision_threshold <= 1.0),
                    fraud_flag INTEGER NOT NULL CHECK (fraud_flag IN (0, 1)),
                    model_config_sha256 TEXT NOT NULL,
                    transaction_json TEXT NOT NULL,
                    engineered_features_json TEXT NOT NULL,
                    component_probabilities_json TEXT NOT NULL,
                    context_depth INTEGER NOT NULL CHECK (context_depth >= 1)
                )
                """
            )

    def record(
        self,
        prediction: Mapping[str, Any],
        reviewer_label: str,
        *,
        updated_at_utc: str,
    ) -> None:
        """Insert or replace the reviewer label for one prediction."""

        label = str(reviewer_label)
        if label not in ALLOWED_FEEDBACK:
            raise ValueError(f"reviewer_label must be one of {sorted(ALLOWED_FEEDBACK)}")
        required = {
            "prediction_id",
            "scored_at_utc",
            "fraud_probability",
            "decision_threshold",
            "fraud_flag",
            "model_config_sha256",
            "transaction",
            "engineered_features",
            "component_probabilities",
            "context_depth",
        }
        missing = sorted(required.difference(prediction))
        if missing:
            raise ValueError(f"prediction is missing feedback fields: {', '.join(missing)}")
        values = (
            str(prediction["prediction_id"]),
            str(prediction["scored_at_utc"]),
            str(updated_at_utc),
            label,
            float(prediction["fraud_probability"]),
            float(prediction["decision_threshold"]),
            int(bool(prediction["fraud_flag"])),
            str(prediction["model_config_sha256"]),
            json.dumps(
                prediction["transaction"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            json.dumps(
                prediction["engineered_features"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            json.dumps(
                prediction["component_probabilities"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            int(prediction["context_depth"]),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_feedback (
                    prediction_id,
                    created_at_utc,
                    updated_at_utc,
                    reviewer_label,
                    fraud_probability,
                    decision_threshold,
                    fraud_flag,
                    model_config_sha256,
                    transaction_json,
                    engineered_features_json,
                    component_probabilities_json,
                    context_depth
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    updated_at_utc = excluded.updated_at_utc,
                    reviewer_label = excluded.reviewer_label
                """,
                values,
            )

    def count(self) -> int:
        """Return the number of distinct reviewed predictions."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM prediction_feedback"
            ).fetchone()
        return int(row["count"])

    def records(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        """Return newest feedback records without expanding stored JSON."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    prediction_id,
                    created_at_utc,
                    updated_at_utc,
                    reviewer_label,
                    fraud_probability,
                    decision_threshold,
                    fraud_flag,
                    model_config_sha256,
                    transaction_json,
                    engineered_features_json,
                    component_probabilities_json,
                    context_depth
                FROM prediction_feedback
                ORDER BY updated_at_utc DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, *, limit: int = 100_000) -> bytes:
        """Serialize feedback rows as UTF-8 CSV for controlled retraining intake."""

        records = self.records(limit=limit)
        if not records:
            return b""
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        return buffer.getvalue().encode("utf-8")
