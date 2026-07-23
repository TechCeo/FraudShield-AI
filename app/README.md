# Application Runtime Contract

Fraud Shield AI exposes a local Streamlit interface for manual transaction assessment, batch review, model evidence inspection, monitoring summaries, and reviewer feedback capture. The application consumes the registered preprocessing, XGBoost, FNN, LSTM, hybrid, sequence, velocity, evaluation, and drift artifacts without refitting or mutating them.

## Runtime Components

| Path | Responsibility |
|---|---|
| `app/main.py` | Streamlit layout, manual form, CSV upload, result visualization, reviewer controls, and system-monitor presentation. |
| `app/scoring.py` | Raw-input validation, causal feature computation, compact sequence-context generation, session-isolated inference state, hybrid scoring, and XGBoost local contributions. |
| `app/feedback.py` | Parameterized SQLite persistence, idempotent reviewer updates, counts, and controlled CSV export. |
| `.streamlit/config.toml` | Dark security-oriented theme, headless server default, upload ceiling, and telemetry setting. |
| `tests/test_app.py` | Input, timestamp, card-precision, sequence-causality, state-commit, and feedback-ledger contracts. |

## Launch

Run commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m app.scoring build-context --project-root .
python -m app.scoring verify-runtime --project-root . --device cpu
python -m streamlit run app/main.py
```

`build-context` is write-once by default. Use `--force` only when a registered upstream model-data or sequence artifact has changed. The application builds the context automatically when it is absent and all required upstream artifacts are available.

The default local address is `http://localhost:8501`.

## Input Contract

Manual and uploaded transactions use the same raw schema:

| Field | Type | Constraint |
|---|---|---|
| `trans_date_trans_time` | `YYYY-MM-DD HH:MM:SS` | Required display timestamp. |
| `cc_num` | String | Required and never accepted from floating-point input because large identifiers lose precision. |
| `merchant` | String | Required; unseen values use the frozen unknown-frequency behavior. |
| `category` | String | Required; unseen values map to the frozen nominal unknown level. |
| `amt` | Decimal currency | Strictly positive and limited to exact cent precision by feature engineering. |
| `city`, `state`, `zip`, `job` | String | Required categorical context. |
| `lat`, `long` | Decimal degrees | Cardholder latitude in `[-90, 90]` and longitude in `[-180, 180]`. |
| `merch_lat`, `merch_long` | Decimal degrees | Merchant latitude and longitude under the same coordinate bounds. |
| `city_pop` | Integer-compatible number | Nonnegative. |
| `dob` | `YYYY-MM-DD` | Cannot occur after the transaction date. |
| `trans_num` | String | Optional; deterministically generated when absent. Must be unique within an uploaded batch. |
| `unix_time` | Whole seconds | Optional. When absent, derived from the dataset-aligned event clock using the documented seven-year offset. |

Uploaded files may contain extra columns; only registered application fields are consumed. A file may contain at most 10,000 rows. Cards may be interleaved, but timestamps for each card must be nondecreasing. The entire submission is rejected on the first schema, coordinate, currency, timestamp, or ordering violation.

## Causal Scoring Contract

The application starts each browser session from independent copies of:

- `data/processed/test_velocity_state.json.gz`, containing terminal rolling and cumulative state;
- `artifacts/app/sequence_context.npz`, containing up to 11 prior transformed vectors plus the terminal timestamp-bucket representative for every registered card;
- `artifacts/models/hybrid_config.json`, containing the frozen blend and decision threshold.

The compact sequence context is bound to the model-data manifest, sequence-index manifest, development feature stream, holdout feature stream, and terminal velocity state by SHA-256 fingerprints. It currently covers 999 cards with 95 transformed features and a maximum LSTM sequence length of 12.

For every submitted row:

1. Coordinates are validated and Haversine distance is calculated.
2. The rolling state emits strictly prior 1-, 6-, and 24-hour velocity plus lifetime metrics.
3. The frozen preprocessor emits the registered 95-feature sparse vector.
4. The sequence state combines up to 11 strictly earlier same-card timestamp buckets with the active vector.
5. XGBoost, FNN, and LSTM probabilities are calculated under the frozen artifacts.
6. The registered log-odds fusion and threshold produce the final probability and Boolean review flag.

Transactions sharing the same `(cc_num, unix_time)` do not enter one another's velocity or sequence context. The lexicographically smallest `trans_num` represents that timestamp bucket for later sequence rows, matching the registered offline pointer rule.

State changes are transactional: a session advances only after every row in a submission completes validation, transformation, sequence construction, model inference, and explanation. Batch review does not alter the session by default. Reviewers can explicitly enable session continuation for uploaded rows. Resetting session history restores the registered holdout-continuation baseline.

## Explanation Contract

Each result presents:

- XGBoost, FNN, LSTM, and hybrid probabilities;
- the frozen decision threshold and Boolean review status;
- sequence depth and whether the card used registered or session-only history;
- distance, rolling counts, rolling spend, and cumulative behavior;
- the eight largest absolute XGBoost feature contributions in log-odds units.

XGBoost contributions explain the XGBoost component rather than the complete nonlinear hybrid. Positive contributions raise that component's local risk and negative contributions lower it. Model consensus and the behavioral feature panel provide the remaining hybrid context.

## Feedback Contract

The `Confirm Fraud` and `False Positive` controls write to:

```text
artifacts/app/feedback.sqlite3
```

The ledger stores the prediction identifier, timestamps, reviewer label, final and component probabilities, threshold, decision, model configuration digest, submitted transaction JSON, engineered behavior JSON, and sequence depth. `prediction_id` is the primary key; submitting a different label for the same prediction updates the reviewer label without duplicating the record.

Feedback remains local and does not retrain, calibrate, or replace the active model automatically. The sidebar export produces a controlled CSV intake for a separately reviewed training workflow. The ledger contains row-level transaction context and inherits the restricted classification of the source data.

## Operational Artifacts

| Path | Format | Contract |
|---|---|---|
| `artifacts/app/sequence_context.npz` | NumPy NPZ, pickle disabled | Compact per-card transformed sequence continuation arrays. |
| `artifacts/app/sequence_context_manifest.json` | Digest-protected JSON | Context dimensions, upstream lineage, file fingerprint, and schema version. |
| `artifacts/app/feedback.sqlite3` | SQLite | Local reviewer feedback and prediction lineage. |

Application artifacts are excluded from version control.
