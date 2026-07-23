# Fraud Shield AI

## Project Overview

Fraud Shield AI is a leakage-aware transaction data platform for payment-fraud risk scoring. It converts chronologically ordered card transactions into deterministic behavioral features, defines immutable temporal partitions, learns preprocessing state from training rows only, and emits sparse model-ready matrices under explicit imbalance controls.

The executable codebase provides:

- chunked CSV ingestion and transformation;
- causal per-card feature computation with portable state;
- fingerprinted chronological train, validation, and out-of-time holdout boundaries;
- train-fitted missing-value imputation, outlier clipping, scaling, frequency encoding, and sparse one-hot encoding;
- mutually exclusive class weighting, random under-sampling, and mixed-data SMOTENC controls restricted to training rows;
- persisted sparse model-data partitions bound to raw, feature, split, and preprocessing fingerprints;
- class-weighted logistic regression and random forest classifiers plus histogram-based XGBoost;
- a feedforward neural network over static engineered features;
- a unidirectional LSTM over strictly prior same-card transaction sequences;
- deterministic random search against an isolated chronological validation set;
- validation-derived F2 decision thresholds and holdout evaluation centered on fraud recall, precision, and average precision;
- integrity-bound estimator reports and a cross-model evaluation registry;
- deterministic exploratory profiling and visualization;
- validation of time ordering, coordinates, currency precision, timestamp ties, state continuity, label independence, artifact integrity, and fitting scope.

The repository contains fitted classical and neural estimators with validation-selected decision thresholds. Its operational outputs include feature-enriched transaction streams, causal velocity state, a chronological split manifest, a JSON-serialized preprocessing contract, imbalance-control metadata, registered sparse `float32` partitions, a causal per-card sequence index, native or restricted-load estimator artifacts, model reports, and evaluation matrices. A scoring endpoint and application runtime are not present.

### Design and performance objectives

| Objective | System behavior |
|---|---|
| Fraud detection quality | Produces behavioral signals intended for recall-, precision-, and PR-AUC-oriented classifiers rather than accuracy-only evaluation. |
| Leakage prevention | Excludes the active transaction and all same-card transactions sharing its timestamp from prior-behavior features. |
| Determinism | Uses explicit window boundaries, integer-cent accumulation, stable timestamp handling, ordered transaction-key fingerprints, fixed random seeds, and digest-protected JSON artifacts. |
| Scalability | Reads CSV input in configurable chunks; feature state scales with retained card histories, while `fit_csv` stores numeric fitting values in a temporary disk-backed matrix and keeps only categorical aggregates in memory. |
| Stream continuity | Carries per-card state across sequential files so known cards do not receive artificial empty histories at partition boundaries. |
| Sequence causality | Represents each transaction with the current row and a bounded chain of strictly earlier same-card rows; future rows and labels never enter a sequence. |
| Evaluation integrity | Fits transformation statistics and applies sampling only to the chronological training partition; validation and holdout prevalence remain unchanged. |
| Model selection integrity | Ranks candidates by validation average precision, freezes an F2 threshold from validation probabilities, and evaluates only the winning estimator on the holdout. |
| Artifact traceability | Binds every model report to its estimator digest, model-data manifest, feature schema, parameters, dependency versions, and random seed. |
| Data integrity | Rejects missing card identifiers, invalid timestamps, non-finite amounts, unsupported currency precision, invalid coordinates, and per-card time reversals. |
| Privacy | EDA samples exclude names, streets, raw card identifiers, birth dates, and transaction identifiers. Feature-enriched CSVs retain source columns and require the same controls as raw data. |

## System Architecture & Pipeline Flow

### Runtime components

| Path | Responsibility |
|---|---|
| `src/features.py` | Geospatial calculations, causal velocity features, state serialization, chunked CSV processing, and the feature-generation CLI. |
| `src/preprocessing.py` | Chronological split manifests, train-fitted sparse transformation, JSON preprocessing artifacts, imbalance controls, and the preprocessing CLI. |
| `src/models/data.py` | Raw-fingerprint verification, feature-stream generation, frozen preprocessing, sparse partition persistence, and model-data integrity checks. |
| `src/models/logistic.py` | Sparse class-weighted logistic classifier factory. |
| `src/models/random_forest.py` | Class-weighted random-forest classifier factory. |
| `src/models/xgboost_model.py` | Histogram-based XGBoost classifier factory with validation early stopping. |
| `src/models/evaluation.py` | Average precision, PR-AUC, calibration metrics, operating metrics, and validation-only F2 threshold selection. |
| `src/models/search.py` | Deterministic random parameter sampling, candidate fitting, winner selection, estimator persistence, and evaluation registries. |
| `src/models/train.py` | Model-data preparation, per-model fitting, complete classifier execution, and report consolidation CLI. |
| `src/models/deep_common.py` | Deterministic PyTorch configuration, training-only endpoint sampling, sparse minibatch materialization, and restricted neural weight persistence. |
| `src/models/fnn.py` | Static feedforward architecture, optimization loop, early stopping, inference, and artifact loading. |
| `src/models/sequences.py` | Digest-bound previous-transaction pointers, cross-partition sparse access, chronological padding, and sequence minibatches. |
| `src/models/lstm.py` | Unidirectional packed-sequence LSTM, optimization loop, early stopping, inference, and artifact loading. |
| `src/models/deep_train.py` | Sequence preparation, neural fitting, and combined classifier comparison CLI. |
| `src/utils.py` | Atomic JSON persistence, stable JSON digests, file hashing, and runtime dependency capture. |
| `src/eda.py` | Exact chunked profiling, deterministic sampling, aggregate quality checks, and pre-sampling feature computation. |
| `notebooks/01_eda.ipynb` | Correlation, imbalance, amount, distance, temporal, and geographic visualizations using outputs from `src/eda.py`. |
| `scripts/validate_notebook.py` | Headless execution of notebook code cells with optional figure export for verification. |
| `tests/test_features.py` | Feature-contract tests covering causal boundaries, timestamp ties, state continuation, geography, currency precision, and chunk invariance. |
| `tests/test_eda.py` | Profiling, sampling, privacy, aggregation, and source-fingerprint tests. |
| `tests/test_preprocessing.py` | Temporal partition, train-only fitting, artifact-integrity, encoding, class-weighting, and resampling tests. |
| `tests/test_models.py` | Probability, threshold, estimator-factory, search-determinism, report-integrity, and model-artifact tests. |
| `tests/test_deep_models.py` | Neural shapes, training sampling, cross-partition sequence causality, restricted weight loading, and neural search tests. |
| `data/` | Immutable raw CSV inputs and the `processed/` destination for generated feature, state, partition, and preprocessing artifacts. |

### Data lineage

```text
data/fraudTrain.csv                         data/fraudTest.csv
        |                                            |
        +---------- fingerprint and scan ------------+
        |                                            |
        |                             chronological_split_manifest
        |                             - whole unix_time buckets
        |                             - train/validation row boundaries
        |                             - isolated out-of-time holdout
        |                                            |
        v                                            v
add_geospatial_features(...)              causal continuation with
RollingFeatureState.transform_chunk(...)  development velocity state
        |                                            |
        v                                            v
fraudTrain_features.csv.gz                fraudTest_features.csv.gz
        |                                            |
        +-- training rows ---------------------------+
        |       fit FraudPreprocessor once           |
        |       - median imputation                   |
        |       - log1p and quantile clipping         |
        |       - standard scaling                    |
        |       - frequency and nominal vocabularies  |
        |                    |                       |
        |                    v                       |
        |       fraud_preprocessor.json.gz            |
        |                    |                       |
        +-- train ----------+-- validation ----------+-- holdout
             |                       |                     |
             v                       v                     v
       frozen transform       frozen transform      frozen transform
             |                       |                     |
             v                       v                     v
       imbalance control       unchanged class       unchanged class
       train rows only         prevalence            prevalence
             |                       |                     |
             +-----------------------+---------------------+
                                     |
                                     v
                        registered sparse CSR partitions
                                     |
                  +-------------+------------+-------------+
                  |             |            |             |
                  v             v            v             v
          logistic/forest    XGBoost     static FNN   previous-row index
                                                            |
                                                            v
                                                     causal card LSTM
                  |             |            |             |
                  +-------------+------------+-------------+
                                      |
                                      v
                          validation AP ranking
                                      |
                                      v
                          validation F2 threshold
                                      |
                                      v
                       winner-only holdout evaluation
```

`process_csv` does not inspect `is_fraud`; changing or removing the label does not change engineered features. The output retains every source column except the redundant export index by default, then appends the registered feature columns below.

`FraudPreprocessor.fit` and `FraudPreprocessor.fit_csv` require `partition="train"`. Passing `split_manifest=manifest` additionally binds fitting to the registered row count, time range, target counts, ordered transaction keys, and manifest digest. Frozen statistics are reused for validation and holdout transformation; those partitions cannot extend vocabularies, change frequencies, alter clipping bounds, or influence scaling. `prepare_training_batch` accepts only the exact fitted partition and enforces the same training-only scope for class weighting and resampling.

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

### Partition and preprocessing controls

Create the digest-protected chronological split manifest from the immutable raw files:

```powershell
python -m src.preprocessing split `
  --development data/fraudTrain.csv `
  --holdout data/fraudTest.csv `
  --output data/processed/split_manifest.json
```

The default `validation_fraction=0.20` places the final 20% of development rows in validation without dividing a shared `unix_time` bucket. The holdout must begin strictly after the development stream. File sizes, SHA-256 fingerprints, schemas, boundaries, row counts, class counts, and fraud rates are recorded in the manifest.

Fit and persist preprocessing state from the manifest-defined training prefix of the feature-enriched development stream:

```python
from src.eda import sha256_file
from src.preprocessing import FraudPreprocessor, load_split_manifest

manifest = load_split_manifest("data/processed/split_manifest.json")
feature_path = "data/processed/fraudTrain_features.csv.gz"
preprocessor = FraudPreprocessor().fit_csv(
    feature_path,
    partition="train",
    split_manifest=manifest,
    source_sha256=sha256_file(feature_path),
)
preprocessor.save("data/processed/fraud_preprocessor.json.gz")
```

Create the deterministic train-only imbalance comparison artifact:

```powershell
python -m src.preprocessing imbalance-report `
  --split-manifest data/processed/split_manifest.json `
  --output data/processed/imbalance_strategy_report.json `
  --sampling-strategy 0.10 `
  --random-state 42
```

Output paths are write-once by default. Pass `--force` only when deliberate replacement of a reproducible artifact is required.

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

## Preprocessing Schema Registry

`FraudPreprocessor` emits a SciPy CSR matrix with `float32` values. Feature order is stored in the preprocessing artifact and protected by `feature_schema_sha256`. Direct identifiers and source fields not listed below, including `cc_num`, `trans_num`, names, street address, raw date of birth, raw timestamps, and `is_fraud`, are not emitted into the matrix.

| Input group | Registered fields | Transformation and output contract |
|---|---|---|
| Numeric transaction and behavior | `amt`, `city_pop`, distance, 1/6/24-hour counts and spend, all-prior count and spend | Median imputation, `log1p`, training-quantile clipping at 0.5% and 99.5%, then training-mean standardization. One `num__log1p_*` column per field. |
| Derived numeric | `age_years` | Transaction date minus date of birth in mean solar years; median imputation, quantile clipping, and standardization. |
| Derived cyclical | hour of day, day of week, month of year | Sine/cosine pairs named `num__hour_*`, `num__day_*`, and `num__month_*`; quantile clipping and standardization use training statistics. |
| Low-cardinality nominal | `category`, `state` | Sparse one-hot columns with explicit `__MISSING__` and `__UNKNOWN__` levels. Vocabulary is fixed by training rows. |
| High-cardinality categorical | `merchant`, `city`, `job`, `zip` | Training relative frequency plus an unknown-level indicator per input: `freq__<field>` and `freq__<field>__unknown`. |

### Fitting and transformation invariants

| Mechanism | Contract |
|---|---|
| Fitting scope | `fit` and `fit_csv` accept only `partition="train"`. Validation and holdout rows cannot contribute preprocessing statistics. |
| Manifest binding | Supplying `split_manifest` derives the training boundary and verifies row count, time endpoints, label counts, ordered transaction-key digest, and manifest digest. `fit_csv` also recomputes the feature CSV SHA-256 fingerprint. |
| Missing values | Numeric medians and categorical missing tokens are learned or registered without using the target. Malformed non-null values remain validation errors. |
| Outliers | Quantile bounds are learned after optional `log1p` transformation and are frozen for every later partition. |
| Unknown categories | Unseen nominal values map to the registered `__UNKNOWN__` one-hot level; unseen high-cardinality values receive frequency `0.0` and unknown indicator `1.0`. |
| Output schema | Numeric fields precede frequency fields, followed by nominal one-hot fields. The ordered names and schema digest are serialized with the artifact. |
| Artifact format | JSON or gzip-compressed JSON, `schema_version = 1`, with content digest, configuration, numeric statistics, vocabularies, frequency mappings, fitted context, and dependency versions. Pickle deserialization is not used. |
| Immutability | Calling `transform`, `prepare_sampler_frame`, or `transform_sampler_frame` does not change fitted statistics or vocabularies. |

### Imbalance controls

Exactly one `ImbalanceConfig.strategy` is applied to a training batch. Validation and holdout rows are transformed without class rebalancing.

| Strategy | Training behavior | Default output ratio |
|---|---|---|
| `none` | Preserves source rows and labels. | Observed training prevalence. |
| `class_weight` | Preserves rows and supplies balanced per-row weights computed from training labels. | Observed training prevalence. |
| `random_under` | Deterministically reduces majority rows with `RandomUnderSampler`; selected row positions are SHA-256 fingerprinted in metadata. | Minority/majority = `0.10`. |
| `smotenc` | Applies SMOTENC to scaled continuous and frequency fields plus unencoded low-cardinality fields, then performs sparse one-hot encoding. | Minority/majority = `0.10`. |

SMOTENC rejects a minority class with no more rows than `k_neighbors`. Default guards reject projections above 2,000,000 rows or 2,000,000,000 conservatively estimated dense working bytes. Every `TrainingBatch` records class counts before and after handling, random seed, ordered input and selection fingerprints, sampler class, dependency versions, schema digest, and the assertion that validation and holdout were not sampled.

`TrainingBatch` is a fixed-partition final-fit interface. It must not be supplied to cross-validation search because its preprocessing state and sampling decision already reflect the complete fitting partition. Cross-validation consumers must fit transformations and apply resampling independently inside each training fold while leaving each fold's validation rows untouched.

## Classical Classifier Registry

### Optimization and decision contract

| Control | Registered behavior |
|---|---|
| Candidate sampling | `ParameterSampler` draws deterministic, nonrepeating parameter configurations from model-specific spaces using `random_state=42`. |
| Training scope | Every candidate fits only the 1,037,340-row chronological training partition. |
| Class imbalance | Logistic regression uses balanced class weights, random forest uses balanced per-tree weights, and XGBoost searches positive-class weights derived from the training ratio. |
| Candidate ranking | Validation average precision is primary; validation recall and precision break exact ties. Accuracy is not a selection metric. |
| Boosting control | XGBoost uses histogram trees, `aucpr` evaluation, and validation-based early stopping. |
| Threshold | The winning model's threshold maximizes validation F2; ties favor recall, precision, then the higher threshold. |
| Holdout access | Only the selected estimator is scored on holdout, using the frozen validation threshold. Holdout results do not alter parameters or thresholds. |
| Persistence | Scikit-learn estimators use compressed Joblib for trusted internal use. XGBoost uses native JSON. Every report stores the estimator SHA-256 digest. |

### Registered evaluation

The current evaluation registry is generated from the immutable model-data manifest and reports the following out-of-time performance:

| Classifier | Validation AP | Holdout AP | Holdout PR-AUC | Holdout recall | Holdout precision | Holdout FPR | Frozen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.601922 | 0.537643 | 0.537603 | 0.580420 | 0.359411 | 0.004008 | 0.870336 |
| Random forest | 0.943986 | 0.900815 | 0.900809 | 0.877855 | 0.743094 | 0.001176 | 0.273135 |
| XGBoost | 0.980621 | 0.965863 | 0.965858 | 0.945455 | 0.843945 | 0.000677 | 0.233010 |

Average precision is the non-interpolated weighted mean of precision over recall increments. `pr_auc_trapezoidal` is also recorded explicitly; the two measures are related but not interchangeable. Each report additionally contains ROC-AUC, Brier score, log loss, confusion counts, specificity, F1, F2, false-negative rate, alert rate, precision at recall targets, recall at precision targets, candidate parameters, and fit durations.

### Model commands

Generate causal features, fit preprocessing on training rows, and persist all sparse partitions:

```powershell
python -m src.models.train prepare-data
```

Optimize one classifier with its registered search count:

```powershell
python -m src.models.train fit --model logistic_regression
python -m src.models.train fit --model random_forest
python -m src.models.train fit --model xgboost
```

Run all classifier searches in one process:

```powershell
python -m src.models.train run-all
```

Verify independent model reports and rebuild the cross-model registry:

```powershell
python -m src.models.train summarize
```

Model-data outputs are written under `data/processed/model_data/`; fitted estimators and reports are written under `artifacts/models/`. Existing registered outputs are rejected unless the global `--force` option is supplied before the subcommand. Generated transaction matrices and model artifacts are excluded from version control.

## Neural Classifier Registry

### Static FNN contract

The static network consumes the same 95 ordered features as the classical estimators. Sparse CSR rows are materialized only for the active minibatch. The registered architecture uses two GELU and LayerNorm hidden blocks, dropout, and a single fraud logit. Training uses AdamW, gradient clipping, weighted `BCEWithLogitsLoss`, all fraud endpoints, and a deterministic training-only negative sample refreshed by epoch.

| Setting | Registered value |
|---|---|
| Hidden widths | 64, 32 |
| Dropout | 0.20 |
| Batch size | 2,048 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Training negative/fraud endpoint cap | 40:1 |
| Positive-loss multiplier | 0.50 of sampled negative/fraud ratio |
| Selected epoch | 6 |

### Sequential LSTM contract

`previous_transaction_index.npy` stores one `int32` pointer per transaction across training, validation, and holdout. A pointer is either `-1` or strictly less than the active global row. Sequence batches follow the pointers backward, restore oldest-to-current order, and use packed lengths so padding is ignored. Validation endpoints may inherit training history, and holdout endpoints may inherit development history; neither partition can point forward. Targets are endpoint labels only and are never sequence inputs.

The registered LSTM projects each 95-feature row to 64 dimensions, processes up to 12 transactions through two unidirectional 64-unit recurrent layers, and emits one fraud logit from the final valid hidden state.

| Setting | Registered value |
|---|---|
| Sequence length | 12 transactions |
| Input projection | 64 |
| Hidden size | 64 |
| Recurrent layers | 2 |
| Dropout | 0.10 |
| Batch size | 1,024 |
| Learning rate | 0.0007 |
| Weight decay | 0.00001 |
| Training negative/fraud endpoint cap | 20:1 |
| Positive-loss multiplier | 1.00 of sampled negative/fraud ratio |
| Selected epoch | 5 |

### Combined evaluation

Every row below uses the same frozen preprocessing schema, chronological partitions, validation average-precision selection rule, validation F2 threshold rule, and untouched out-of-time holdout.

| Classifier | Validation AP | Holdout AP | Holdout PR-AUC | Holdout recall | Holdout precision | Holdout FPR | Alert rate | Frozen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.601922 | 0.537643 | 0.537603 | 0.580420 | 0.359411 | 0.004008 | 0.006233 | 0.870336 |
| Static FNN | 0.902201 | 0.850898 | 0.850878 | 0.873660 | 0.539591 | 0.002889 | 0.006250 | 0.906579 |
| Random forest | 0.943986 | 0.900815 | 0.900809 | 0.877855 | 0.743094 | 0.001176 | 0.004560 | 0.273135 |
| Causal LSTM | 0.957186 | 0.931596 | 0.931586 | **0.951049** | 0.562603 | 0.002865 | 0.006525 | 0.774295 |
| XGBoost | **0.980621** | **0.965863** | **0.965858** | 0.945455 | **0.843945** | **0.000677** | **0.004324** | 0.233010 |

XGBoost remains the strongest holdout ranking and precision model. The LSTM produces the highest holdout recall, detecting 2,040 of 2,145 fraud rows, but its lower precision produces 1,586 false alerts. The FNN materially exceeds logistic regression while remaining below the sequential and tree-based nonlinear estimators.

### Neural commands

Create and verify the global causal pointer index:

```powershell
python -m src.models.deep_train prepare-sequences
```

Optimize each neural classifier:

```powershell
python -m src.models.deep_train --device auto fit --model fnn
python -m src.models.deep_train --device auto fit --model lstm
```

Verify all five reports and rebuild the combined registry:

```powershell
python -m src.models.deep_train summarize
```

Neural weights use PyTorch tensor artifacts loaded with `weights_only=True`. Reports bind each weight file to its SHA-256 digest, model configuration, model-data lineage, feature schema, search parameters, epoch metrics, runtime dependencies, threshold, and holdout metrics. The LSTM report additionally binds the sequence-index manifest.

## Environment Setup & Verification

### Prerequisites

- Python 3.12 or later; the verified local runtime is Python 3.13.4
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

Run the preprocessing-contract suite:

```powershell
python -m pytest tests/test_preprocessing.py -q --basetemp .pytest_tmp_preprocessing
```

Run the model-contract suite:

```powershell
python -m pytest tests/test_models.py -q --basetemp .pytest_tmp_models
```

Run the neural-contract suite:

```powershell
python -m pytest tests/test_deep_models.py -q --basetemp .pytest_tmp_deep
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

Confirm the preprocessing CLI surface without writing output files:

```powershell
python -m src.preprocessing --help
```

Confirm the classifier CLI surface without writing output files:

```powershell
python -m src.models.train --help
```

Confirm the neural CLI surface without writing output files:

```powershell
python -m src.models.deep_train --help
```

A successful verification command exits with status code `0`. Test failures, schema violations, artifact-digest mismatches, invalid fitting scope, timestamp reversals, invalid coordinates, and unsupported currency precision produce nonzero exits or raised exceptions.
