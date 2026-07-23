# Runtime Artifact Contract

This directory is the controlled destination for fitted estimators, evaluation reports, prediction caches, monitoring references, application sequence context, reviewer feedback, and runtime logs. Generated artifacts are excluded from version control; this registry is the only tracked file in the directory.

## Integrity and Loading Rules

- JSON registries contain `artifact_type`, `schema_version`, and `payload_sha256` fields unless their producer explicitly uses another registered format.
- Model reports bind the estimator filename and SHA-256 fingerprint to model-data, feature-schema, dependency, parameter, and evaluation context.
- Scikit-learn Joblib files are trusted-internal artifacts and must be fingerprint-verified before deserialization.
- XGBoost uses its native JSON estimator format.
- PyTorch weights are loaded with `weights_only=True` and must match the associated report fingerprint.
- NumPy artifacts are loaded with `allow_pickle=False`.
- The hybrid configuration verifies every component report and estimator before inference.
- Application sequence context verifies model-data, sequence-index, feature-stream, and terminal velocity-state lineage before scoring.
- Outputs are write-once unless the producing command receives an explicit `--force` option.

## Model Artifact Registry

| Path | Format | Producer | Operational contract |
|---|---|---|---|
| `models/logistic_regression.joblib` | Compressed Joblib | `python -m src.models.train fit --model logistic_regression` | Registered class-weighted logistic estimator. Trusted internal loading only. |
| `models/logistic_regression_report.json` | Digest-protected JSON | Same as estimator | Parameters, estimator fingerprint, validation threshold, lineage, and validation/holdout metrics. |
| `models/random_forest.joblib` | Compressed Joblib | `python -m src.models.train fit --model random_forest` | Registered class-weighted forest estimator. Trusted internal loading only. |
| `models/random_forest_report.json` | Digest-protected JSON | Same as estimator | Parameters, estimator fingerprint, validation threshold, lineage, and validation/holdout metrics. |
| `models/xgboost.json` | Native XGBoost JSON | `python -m src.models.train fit --model xgboost` | Registered histogram-based boosted estimator. |
| `models/xgboost_report.json` | Digest-protected JSON | Same as estimator | Parameters, estimator fingerprint, validation threshold, early-stopping context, lineage, and validation/holdout metrics. |
| `models/evaluation_matrix.json` | Digest-protected JSON | `python -m src.models.train summarize` | Three-classifier comparison using registered report metrics. |
| `models/fnn.pt` | PyTorch tensor state | `python -m src.models.deep_train fit --model fnn` | Static 95-feature feedforward weights. Restricted loading only. |
| `models/fnn_report.json` | Digest-protected JSON | Same as weights | Architecture, training trace, selected epoch, weight fingerprint, threshold, lineage, and evaluation. |
| `models/lstm.pt` | PyTorch tensor state | `python -m src.models.deep_train fit --model lstm` | Length-12 unidirectional sequence-model weights. Restricted loading only. |
| `models/lstm_report.json` | Digest-protected JSON | Same as weights | Architecture, training trace, selected epoch, weight fingerprint, sequence lineage, threshold, and evaluation. |
| `models/deep_evaluation_matrix.json` | Digest-protected JSON | `python -m src.models.deep_train summarize` | Five-classifier static and sequential comparison. |
| `models/hybrid_config.json` | Digest-protected JSON | `python -m src.models.hybrid_train fit` | Frozen component weights, log-odds blend, F2 threshold, and component fingerprints. |
| `models/hybrid_report.json` | Digest-protected JSON | Same as configuration | Fusion search, prediction-cache fingerprint, lineage, and validation/holdout metrics. |
| `models/hybrid_probabilities.npz` | Compressed NumPy NPZ | Same as configuration | Row-aligned component and hybrid probabilities for validation and holdout. Restricted model output. |
| `models/hybrid_evaluation_matrix.json` | Digest-protected JSON | `python -m src.models.hybrid_train summarize` | Six-classifier quality comparison. |
| `models/latency_benchmark.json` | Digest-protected JSON | `python -m src.models.hybrid_train benchmark` | Warm prepared-feature latency and throughput for all six classifiers. |
| `models/operational_tradeoff_matrix.json` | Digest-protected JSON | `python -m src.models.hybrid_train summarize` | Quality, input context, artifact footprint, latency, and throughput comparison. |
| `models/drift_detector.json` | Digest-protected JSON | `python -m src.models.hybrid_train simulate-drift` | Frozen validation-window feature and prediction references with PSI thresholds. |
| `models/drift_simulation_report.json` | Digest-protected JSON | Same as detector | Validation, out-of-time holdout, and controlled-shift monitoring results. |

## Application Artifact Registry

| Path | Format | Producer | Operational contract |
|---|---|---|---|
| `app/sequence_context.npz` | Compressed NumPy NPZ | `python -m app.scoring build-context --project-root .` | Per-card transformed sequence continuation; 999 cards, 11 prior-vector slots, and 95 features. |
| `app/sequence_context_manifest.json` | Digest-protected JSON | Same as context | Context fingerprint, dimensions, schema version, and upstream model/feature/state lineage. |
| `app/feedback.sqlite3` | SQLite | `python -m streamlit run app/main.py` | Local reviewer labels, transaction inputs, engineered context, component scores, final decision, and model lineage. |
| `app/streamlit.stdout.log` | Text | Optional local service wrapper | Non-authoritative Streamlit process output. |
| `app/streamlit.stderr.log` | Text | Optional local service wrapper | Non-authoritative Streamlit diagnostic output. |

SQLite may create temporary `-wal` and `-shm` companions while the application is active. They are implementation files, not independent records, and must remain co-located with the database.

## Registered Identity

| Registry | SHA-256 |
|---|---|
| Model-data manifest payload | `5A2850759C0F79A4163FB50A21612CCD408219499FB3A2FCBB956C8B290BD5AC` |
| Ordered feature schema | `E08BFD5A542A98001AD47A11AE5DE577FDC30CA5813C8A16C01748277BD50149` |
| Sequence-index manifest payload | `FE0254F4D417C5FC471F9A20A7404C7BBE686B776978E9600150BECFCC6530CD` |
| Hybrid configuration payload | `4769176FE87AD5A8AF3091680AEECEBEEDED4F0EA63570AB4A44D172B3291AD2` |

Any upstream fingerprint change invalidates dependent artifacts and requires regeneration in data-lineage order.

## Regeneration Commands

Run commands from the repository root:

```powershell
python -m src.models.train prepare-data
python -m src.models.train run-all
python -m src.models.deep_train prepare-sequences
python -m src.models.deep_train run-all
python -m src.models.hybrid_train --device cpu run-all
python -m app.scoring build-context --project-root .
python -m app.scoring verify-runtime --project-root . --device cpu
```

Existing artifacts cause these commands to stop unless the applicable global `--force` option is supplied before the subcommand. Replacement is appropriate only after confirming the upstream lineage and retention requirements.

## Classification and Retention

- Estimator artifacts, learned mappings, and monitoring references are internal model intellectual property.
- Prediction caches, feedback records, and application context contain row-level or linkable behavior and are restricted.
- Feedback contains user-submitted transaction attributes and must receive the same access controls as the source data.
- Runtime logs are diagnostic only and must not be treated as model, prediction, or feedback records.
- Deleting an estimator, configuration, or context invalidates downstream inference until it is regenerated.
- Feedback retention and deletion require an explicit data-governance policy before non-local use.
