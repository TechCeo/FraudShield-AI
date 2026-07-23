# Processed Artifact Contract

This directory is the controlled destination for reproducible feature, state, partition, preprocessing, and sparse model-data artifacts. Generated files are excluded from version control and inherit the restricted data classification of the raw transaction sources unless their schema explicitly contains metadata only.

## Artifact Registry

| Path | Producer | Format and schema | Operational contract |
|---|---|---|---|
| `split_manifest.json` | `python -m src.preprocessing split` | JSON; `chronological_split_manifest`, schema version 1 | Digest-protected train, validation, and out-of-time holdout boundaries with raw-file fingerprints, schemas, time ranges, row ranges, label counts, ordered transaction-key fingerprints, and holdout gap. |
| `fraudTrain_features.csv.gz` | `python -m src.features` | Gzip-compressed CSV | Development rows in source order, all raw columns except `Unnamed: 0`, and the registered geospatial and causal velocity features. Restricted transaction data. |
| `train_velocity_state.json.gz` | `python -m src.features` | Gzip-compressed JSON; `RollingFeatureState`, schema version 1 | Terminal per-card state for causal continuation from the development stream into the holdout stream. |
| `fraudTest_features.csv.gz` | `python -m src.features --state-in ...` | Gzip-compressed CSV | Out-of-time holdout rows transformed with incoming development velocity state. Restricted transaction data. |
| `test_velocity_state.json.gz` | `python -m src.features` | Gzip-compressed JSON; `RollingFeatureState`, schema version 1 | Terminal state after the holdout stream. |
| `fraud_preprocessor.json.gz` | `FraudPreprocessor.save` | Gzip-compressed JSON; `fraud_preprocessor`, schema version 1 | Digest-protected transformation configuration, numeric statistics, category vocabularies, frequency mappings, ordered feature names, schema fingerprint, dependency versions, and training-fit context. |
| `imbalance_strategy_report.json` | `python -m src.preprocessing imbalance-report` | JSON; `imbalance_strategy_report`, schema version 1 | Deterministic projections for unchanged data, class weighting, random under-sampling, and SMOTENC using training class counts only. |
| `model_data/fraud_preprocessor.json.gz` | `python -m src.models.train prepare-data` | Gzip-compressed JSON; `fraud_preprocessor`, schema version 1 | Preprocessing state fitted exclusively on the registered training prefix and bound to the feature-stream and split fingerprints. |
| `model_data/train_features.npz` | `python -m src.models.train prepare-data` | SciPy CSR NPZ; `float32`, 1,037,340 x 95 | Sparse classifier input for the chronological training partition. |
| `model_data/validation_features.npz` | `python -m src.models.train prepare-data` | SciPy CSR NPZ; `float32`, 259,335 x 95 | Sparse classifier input transformed with frozen training state. |
| `model_data/holdout_features.npz` | `python -m src.models.train prepare-data` | SciPy CSR NPZ; `float32`, 555,719 x 95 | Sparse out-of-time classifier input transformed with frozen training state. |
| `model_data/*_target.npy` | `python -m src.models.train prepare-data` | NumPy NPY; `int8` | Binary labels aligned one-to-one with each registered sparse matrix. Pickle loading is disabled. |
| `model_data/model_data_manifest.json` | `python -m src.models.train prepare-data` | JSON; `fraud_model_data`, schema version 1 | Digest-protected registry of matrix shapes, nonzero counts, file fingerprints, labels, ordered feature names, preprocessing identity, feature-stream fingerprints, and split identity. |
| `model_data/previous_transaction_index.npy` | `python -m src.models.deep_train prepare-sequences` | NumPy NPY; `int32`, 1,852,394 rows | One previous-row pointer per transaction across the complete chronological stream. Each value is `-1` or a strictly smaller global row index. |
| `model_data/sequence_index_manifest.json` | `python -m src.models.deep_train prepare-sequences` | JSON; `fraud_sequence_index`, schema version 1 | Digest-protected pointer identity, partition offsets, ordered transaction-key fingerprints, cold-start counts, card count, feature-stream lineage, split identity, and model-data identity. |

`prepare_training_batch` remains the training-only interface for direct imbalance experiments. The classifier runtime uses registered sparse partitions and estimator-level class weighting; it does not resample validation or holdout data. Fitted estimators, hybrid configuration, prediction cache, drift references, latency measurements, and evaluation reports are stored under `artifacts/models/`, outside this directory.

## Data Lineage and Dependency Rules

```text
immutable raw files
    |
    +-- SHA-256 + chronological scan --> split_manifest.json
    |
    +-- causal feature generation ----> fraudTrain_features.csv.gz
    |                                      |
    |                                      +-- training prefix only
    |                                      |       |
    |                                      |       v
    |                                      |  fraud_preprocessor.json.gz
    |                                      |
    |                                      +-- frozen transform --> train/validation matrices
    |
    +-- development velocity state ----> holdout feature generation
                                           |
                                           +-- frozen transform --> holdout matrix

train/validation/holdout matrices --> model_data/model_data_manifest.json
            |
            +-- static training + validation selection --> artifacts/models/fnn.*
            |
            +-- ordered transaction keys --> previous_transaction_index.npy
                                              |
                                              +-- causal sequence training --> artifacts/models/lstm.*
            |
            +-- classical training + validation selection --> artifacts/models/*
                                                               |
                                                               +-- frozen threshold holdout scoring
            |
            +-- XGBoost + FNN + LSTM probabilities --> validation-only fusion
                                                           |
                                                           +-- warm hybrid inference
                                                           |
                                                           +-- latency + drift registries

split_manifest.json -- training counts --> imbalance_strategy_report.json
training matrix + training labels --------> one TrainingBatch imbalance control
```

The following controls are mandatory:

- The raw SHA-256 fingerprints must match `data/README.md` before any generated artifact is trusted.
- The split manifest boundary applies to the development stream by row position and preserves complete `unix_time` buckets.
- Preprocessing statistics, vocabularies, and frequency mappings are fitted from the manifest-defined training prefix only; manifest-bound fitting verifies row count, time endpoints, target counts, and ordered transaction keys.
- Validation and holdout rows use frozen preprocessing state and retain their observed label prevalence.
- Random under-sampling and SMOTENC operate only through `prepare_training_batch(..., partition="train")`.
- Holdout feature generation loads `train_velocity_state.json.gz`; starting the holdout with empty state violates the causal continuity contract.
- JSON artifacts must pass their embedded payload digest and schema-version checks before use.
- Sparse matrix and target fingerprints must match `model_data_manifest.json` before classifier fitting.
- Model reports must match both their embedded payload digest and estimator SHA-256 digest before evaluation consolidation.
- Hybrid fusion weights, blend space, and threshold must be selected from validation probabilities only; holdout probabilities cannot alter the registered configuration.
- The warm hybrid engine must verify every component report and estimator fingerprint before scoring.
- Drift feature and prediction references must use the same registered validation window; current windows cannot mutate the frozen reference artifact.
- Every sequence pointer must be `-1` or strictly smaller than its row; partition transaction-key digests must match the split manifest.
- Validation sequences may use training feature history, and holdout sequences may use development feature history. Sequence inputs never contain targets or future rows.

## Canonical Commands

Run commands from the repository root.

Create the chronological boundary manifest:

```powershell
python -m src.preprocessing split `
  --development data/fraudTrain.csv `
  --holdout data/fraudTest.csv `
  --output data/processed/split_manifest.json
```

Create development features and terminal state:

```powershell
python -m src.features `
  --input data/fraudTrain.csv `
  --output data/processed/fraudTrain_features.csv.gz `
  --state-out data/processed/train_velocity_state.json.gz
```

Continue state into the holdout feature stream:

```powershell
python -m src.features `
  --input data/fraudTest.csv `
  --output data/processed/fraudTest_features.csv.gz `
  --state-in data/processed/train_velocity_state.json.gz `
  --state-out data/processed/test_velocity_state.json.gz
```

Create the default training-only imbalance comparison:

```powershell
python -m src.preprocessing imbalance-report `
  --split-manifest data/processed/split_manifest.json `
  --output data/processed/imbalance_strategy_report.json `
  --sampling-strategy 0.10 `
  --random-state 42
```

Generate or verify feature streams and persist the sparse classifier partitions:

```powershell
python -m src.models.train prepare-data
```

Optimize the registered classifiers and persist their evaluation reports:

```powershell
python -m src.models.train fit --model logistic_regression
python -m src.models.train fit --model random_forest
python -m src.models.train fit --model xgboost
python -m src.models.train summarize
```

Create causal sequence pointers, optimize both neural classifiers, and consolidate all reports:

```powershell
python -m src.models.deep_train prepare-sequences
python -m src.models.deep_train fit --model fnn
python -m src.models.deep_train fit --model lstm
python -m src.models.deep_train summarize
```

Optimize fusion, measure prepared-feature latency, evaluate drift, and consolidate six-model registries:

```powershell
python -m src.models.hybrid_train --device cpu fit
python -m src.models.hybrid_train --device cpu benchmark
python -m src.models.hybrid_train simulate-drift
python -m src.models.hybrid_train summarize
```

CLI outputs are write-once unless `--force` is supplied. JSON and feature outputs are written through temporary paths and atomically moved into place after successful serialization.

## Retention and Access Controls

- Feature CSVs retain sensitive account identifiers, personal attributes, transaction identifiers, and precise locations. Apply the same access controls as the raw CSVs.
- Velocity-state files contain card identifiers and behavioral aggregates and therefore remain restricted.
- The preprocessing artifact contains learned aggregate statistics and category values. Treat it as internal model data and prevent unreviewed publication.
- Sparse model-data matrices and targets retain row-level behavioral and label information and remain restricted.
- Joblib estimator artifacts are trusted-internal files and must never be loaded from an unverified source; use the report fingerprint before deserialization.
- Native XGBoost JSON is the registered portable format for the boosted classifier.
- Neural tensor artifacts are loaded through PyTorch's restricted `weights_only=True` path and must still match their report fingerprints.
- The hybrid configuration contains no executable payload. Its report binds every component report, estimator fingerprint, model-data manifest, feature schema, and sequence-index manifest.
- The hybrid probability cache is row-level model output and remains restricted. Drift references expose aggregate feature distributions and sampled scores and remain internal monitoring data.
- The sequence pointer array contains row relationships but no labels or raw account identifiers; it remains internal because it exposes transaction linkage and partition topology.
- The split and imbalance manifests contain file-system paths, fingerprints, time bounds, and label aggregates. They do not contain row-level transactions but remain internal operational metadata.
- Delete or replace generated artifacts only after confirming that dependent matrices, reports, or model artifacts are no longer in use. Recompute downstream consumers whenever an upstream fingerprint or schema digest changes.
