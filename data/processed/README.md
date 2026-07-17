# Processed Artifact Contract

This directory is the controlled destination for reproducible feature, state, partition, and preprocessing artifacts. Generated files are excluded from version control and inherit the restricted data classification of the raw transaction sources unless their schema explicitly contains metadata only.

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

Sparse `TrainingBatch.X` matrices are generated in memory by `prepare_training_batch`. The preprocessing module does not serialize model matrices or labels; downstream consumers own any persisted matrix format and must retain the accompanying `TrainingBatch.metadata` and preprocessing schema digest.

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

CLI outputs are write-once unless `--force` is supplied. JSON and feature outputs are written through temporary paths and atomically moved into place after successful serialization.

## Retention and Access Controls

- Feature CSVs retain sensitive account identifiers, personal attributes, transaction identifiers, and precise locations. Apply the same access controls as the raw CSVs.
- Velocity-state files contain card identifiers and behavioral aggregates and therefore remain restricted.
- The preprocessing artifact contains learned aggregate statistics and category values. Treat it as internal model data and prevent unreviewed publication.
- The split and imbalance manifests contain file-system paths, fingerprints, time bounds, and label aggregates. They do not contain row-level transactions but remain internal operational metadata.
- Delete or replace generated artifacts only after confirming that dependent matrices, reports, or model artifacts are no longer in use. Recompute downstream consumers whenever an upstream fingerprint or schema digest changes.
