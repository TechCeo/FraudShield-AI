# Fraud Shield AI

## Project Overview

Fraud Shield AI is a leakage-aware transaction feature platform for payment-fraud risk scoring. It converts chronologically ordered card transactions into deterministic geospatial, velocity, and cumulative behavioral signals that provide inputs for subsequent preprocessing, model fitting, and low-latency online scoring.

The executable codebase provides:

- chunked CSV ingestion and transformation;
- causal per-card feature computation with portable state;
- deterministic exploratory profiling and visualization;
- validation of time ordering, coordinates, currency precision, timestamp ties, state continuity, and label independence.

The repository does not contain a fitted estimator, decision threshold, scoring endpoint, or application runtime. Its operational output is a feature-enriched transaction stream plus the state required to continue causal computation across sequential data partitions.

### Design and performance objectives

| Objective | System behavior |
|---|---|
| Fraud detection quality | Produces behavioral signals intended for recall-, precision-, and PR-AUC-oriented classifiers rather than accuracy-only evaluation. |
| Leakage prevention | Excludes the active transaction and all same-card transactions sharing its timestamp from prior-behavior features. |
| Determinism | Uses explicit window boundaries, integer-cent accumulation, stable timestamp handling, and versioned JSON state. |
| Scalability | Reads CSV input in configurable chunks; state memory scales with unique cards and timestamp buckets retained inside configured windows rather than total source rows. |
| Stream continuity | Carries per-card state across sequential files so known cards do not receive artificial empty histories at partition boundaries. |
| Data integrity | Rejects missing card identifiers, invalid timestamps, non-finite amounts, unsupported currency precision, invalid coordinates, and per-card time reversals. |
| Privacy | EDA samples exclude names, streets, raw card identifiers, birth dates, and transaction identifiers. Feature-enriched CSVs retain source columns and require the same controls as raw data. |

## System Architecture & Pipeline Flow

### Runtime components

| Path | Responsibility |
|---|---|
| `src/features.py` | Geospatial calculations, causal velocity features, state serialization, chunked CSV processing, and the feature-generation CLI. |
| `src/eda.py` | Exact chunked profiling, deterministic sampling, aggregate quality checks, and pre-sampling feature computation. |
| `notebooks/01_eda.ipynb` | Correlation, imbalance, amount, distance, temporal, and geographic visualizations using outputs from `src/eda.py`. |
| `scripts/validate_notebook.py` | Headless execution of notebook code cells with optional figure export for verification. |
| `tests/test_features.py` | Feature-contract tests covering causal boundaries, timestamp ties, state continuation, geography, currency precision, and chunk invariance. |
| `tests/test_eda.py` | Profiling, sampling, privacy, aggregation, and source-fingerprint tests. |
| `data/` | Immutable raw CSV inputs and the `processed/` destination for generated feature streams and state files. |

### Data lineage

```text
data/fraudTrain.csv or data/fraudTest.csv
                    |
                    v
      Chunked CSV ingestion (default: 100,000 rows)
      - read cc_num as a string identifier
      - drop the redundant Unnamed: 0 export index
      - preserve source columns and row order
                    |
                    v
      add_geospatial_features(...)
      - validate latitude and longitude ranges
      - compute cardholder-to-merchant Haversine distance
                    |
                    v
      RollingFeatureState.transform_chunk(...)
      - use unix_time as the causal clock
      - aggregate same-card/same-second rows as one pending bucket
      - compute prior 1-hour, 6-hour, and 24-hour counts and spend
      - compute all-prior transaction count and spend
      - reject per-card timestamp reversals
                    |
          +---------+---------+
          |                   |
          v                   v
  Atomic CSV/CSV.GZ     JSON/JSON.GZ state
  feature output        schema_version = 1
          |                   |
          +---------+---------+
                    |
                    v
      Sequential partition or scoring consumer
```

`process_csv` does not inspect `is_fraud`; changing or removing the label does not change engineered features. The output retains every source column except the redundant export index by default, then appends the registered feature columns below.

### Feature-stream generation

Run commands from the `FraudShield` directory.

Transform the development stream and persist its terminal state:

```powershell
python -m src.features `
  --input data/fraudTrain.csv `
  --output data/processed/fraudTrain_features.csv.gz `
  --state-out data/processed/train_velocity_state.json.gz
```

Continue feature computation into the chronological evaluation stream:

```powershell
python -m src.features `
  --input data/fraudTest.csv `
  --output data/processed/fraudTest_features.csv.gz `
  --state-in data/processed/train_velocity_state.json.gz `
  --state-out data/processed/test_velocity_state.json.gz
```

Existing output paths are rejected unless `--force` is supplied. Each CSV or state file is written through a temporary file and atomically replaced after successful serialization. Feature output and state output are committed separately.

## Feature Schema Registry

Default configuration:

| Setting | Value |
|---|---|
| Card identifier | `cc_num` |
| Causal clock | `unix_time`, whole seconds |
| Transaction amount | `amt`, no more than two decimal places |
| Cardholder coordinates | `lat`, `long` |
| Merchant coordinates | `merch_lat`, `merch_long` |
| Rolling windows | 3,600; 21,600; and 86,400 seconds |
| Invalid-coordinate policy | `raise` |
| Earth radius | 6,371.0088 km |

For a transaction belonging to card `c` at time `t`, every rolling interval is left-inclusive and right-exclusive: `t - window <= event_time < t`.

| Output column | Type and unit | Definition |
|---|---|---|
| `distance_card_merchant_km` | `float64`, kilometers | Great-circle distance between (`lat`, `long`) and (`merch_lat`, `merch_long`) using the Haversine formula. |
| `cc_txn_count_prev_1h` | `int64`, transactions | Count for card `c` in `[t - 3600, t)`. |
| `cc_amt_sum_prev_1h` | `float64`, currency units | Sum of `amt` for card `c` in `[t - 3600, t)`. |
| `cc_txn_count_prev_6h` | `int64`, transactions | Count for card `c` in `[t - 21600, t)`. |
| `cc_amt_sum_prev_6h` | `float64`, currency units | Sum of `amt` for card `c` in `[t - 21600, t)`. |
| `cc_txn_count_prev_24h` | `int64`, transactions | Count for card `c` in `[t - 86400, t)`. |
| `cc_amt_sum_prev_24h` | `float64`, currency units | Sum of `amt` for card `c` in `[t - 86400, t)`. |
| `cc_txn_count_prior` | `int64`, transactions | Count of all transactions for card `c` in timestamp buckets strictly earlier than `t`, without a time limit. |
| `cc_amt_sum_prior` | `float64`, currency units | Sum of `amt` across all strictly earlier timestamp buckets for card `c`, without a time limit. |

### Numeric and state invariants

| Mechanism | Contract |
|---|---|
| Currency scaling | `amt` is converted to signed `int64` cents before accumulation. Values that cannot be represented exactly at two-decimal currency precision are rejected. Rolling sums are exposed as `float64` currency units. |
| Timestamp ties | Rows sharing `(cc_num, unix_time)` receive identical prior-behavior features. Their aggregate becomes visible only to a strictly later timestamp. |
| Window boundary | A bucket exactly at `t - window` is included; a bucket one second earlier is excluded. |
| Card isolation | State is keyed by the string representation of `cc_num`; one card never contributes to another card's features. |
| Stream ordering | Chunked transformation accepts interleaved cards but requires nondecreasing `unix_time` for each card across all chunks. A reversal raises `ValueError`. |
| Batch ordering | In-memory batch transformation stably sorts by the causal clock for computation and restores the original positional order. |
| Rolling state | Each card stores a pending timestamp bucket, one deque per configured window, cached rolling totals, and scalar all-prior totals. |
| State persistence | `RollingFeatureState` serializes to JSON or gzip-compressed JSON with `schema_version = 1`; arbitrary-code pickle deserialization is not used. |
| State compatibility | A loaded state must use the same column names, windows, and invalid-coordinate policy as the active `FeatureConfig`; a mismatch raises `ValueError`. |
| Partition continuity | Supplying `--state-in` makes the next file a causal continuation. Cards absent from the state start with zero prior counts and spend. |
| Label independence | `is_fraud` is neither required nor read by feature calculations. |

Custom positive whole-second windows are supported. Output suffixes use `Nh` for whole hours, `Nm` for whole minutes, and `Ns` otherwise.

## Environment Setup & Verification

### Prerequisites

- Python 3.11 or later; the verified local runtime is Python 3.13.4
- `fraudTrain.csv` and `fraudTest.csv` under `data/`
- Sufficient local storage for raw files, generated feature streams, and serialized state

### Installation

1. Change to the project directory:

   ```powershell
   cd C:\path\to\Capstone\FraudShield
   ```

2. Create an isolated Python environment:

   ```powershell
   python -m venv .venv
   ```

3. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install the declared dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### Automated verification

Run the feature-contract suite:

```powershell
python -m pytest tests/test_features.py -q --basetemp .pytest_tmp_features
```

Run all automated tests:

```powershell
python -m pytest -q --basetemp .pytest_tmp_all
```

Validate every notebook code cell against the full training dataset with a headless plotting backend:

```powershell
python scripts/validate_notebook.py notebooks/01_eda.ipynb
```

Confirm the feature CLI surface without writing output files:

```powershell
python -m src.features --help
```

A successful verification command exits with status code `0`. Test failures, schema violations, timestamp reversals, invalid coordinates, and unsupported currency precision produce nonzero exits or raised exceptions.
