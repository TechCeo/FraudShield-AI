# Fraud Shield Dataset Contract

| Contract property | Value |
|---|---|
| Identifier | `fraudshield.raw.credit_card_transactions` |
| Version | `1.0.0` |
| Status | Provisional for internal/local processing pending authoritative provenance and license metadata |
| Owner role | Fraud Shield data and ML engineering |
| Data classification | Restricted: account identifiers, personal attributes, and precise locations |
| Primary key | `trans_num` |
| Entity/grouping key | `cc_num` |
| Causal ordering key | `unix_time` |
| Display-time field | `trans_date_trans_time` |
| Target | `is_fraud` |
| External source reference and license | Not recorded in the repository; redistribution requires authoritative provenance and license review |

This contract defines the immutable raw transaction inputs consumed by `src/features.py` and `src/eda.py`. It specifies file identity, column order, logical types, validation constraints, temporal alignment, holdout isolation, and handling requirements.

## Dataset Fingerprints

Any change to file size, row count, column order, or SHA-256 digest represents a different immutable dataset revision and requires a new contract version and fingerprint set.

| File | Operational role | Shape | Size | SHA-256 |
|---|---|---:|---:|---|
| `fraudTrain.csv` | Development and fitting stream | 1,296,675 rows × 23 raw columns | 351,238,196 bytes (334.97 MiB) | `FD7139200DBFCBED0B6742BBE05A4F1ABCE532C4FEF20918228A651647A3E75D` |
| `fraudTest.csv` | Out-of-time evaluation holdout | 555,719 rows × 23 raw columns | 150,354,339 bytes (143.39 MiB) | `12D553AB19440C752D2531EE1AF44BB64F12CC3D3839F1649F19E81C230545F0` |

### Label distribution and time boundaries

| File | Legitimate (`0`) | Fraud (`1`) | Fraud rate | Display-time boundary | `unix_time` boundary |
|---|---:|---:|---:|---|---|
| `fraudTrain.csv` | 1,289,169 | 7,506 | 0.578865% | `2019-01-01 00:00:18` through `2020-06-21 12:13:37` | `1325376018` through `1371816817` |
| `fraudTest.csv` | 553,574 | 2,145 | 0.385986% | `2020-06-21 12:14:25` through `2020-12-31 23:59:34` | `1371816865` through `1388534374` |

The aggregate holdout label counts were inspected for contract-integrity validation only. Neither row-level holdout records nor aggregate holdout outcomes may influence exploratory analysis, fitting, encoder estimation, resampling, hyperparameter selection, probability calibration, or threshold selection.

## Immutable Schema, Shapes, and Boundaries

The shapes, fingerprints, and boundary values above are immutable properties of this contract version.

### Observed field boundaries

| Field | `fraudTrain.csv` | `fraudTest.csv` |
|---|---|---|
| Source export index | `0` through `1,296,674` | `0` through `555,718` |
| Cardholder latitude | `20.0271` through `66.6933` | `20.0271` through `65.6899` |
| Cardholder longitude | `-165.6723` through `-67.9503` | `-165.6723` through `-67.9503` |
| Merchant latitude | `19.027785` through `67.510267` | `19.027422` through `66.679297` |
| Merchant longitude | `-166.671242` through `-66.950902` | `-166.671575` through `-66.952026` |
| Date of birth | `1924-10-30` through `2005-01-29` | `1924-10-30` through `2005-01-29` |
| City population | Within the combined observed range `23` through `2,906,700` | Within the combined observed range `23` through `2,906,700` |

### Raw schema registry

The files are UTF-8 without a byte-order mark, comma-delimited, double-quoted where required, and stored with CRLF line endings. The 23 raw columns have identical order and default pandas-inferred dtypes in both files. Every field is required and non-nullable.

| Position | Column | Default pandas inference | Logical role and contract |
|---:|---|---|---|
| 1 | `<empty>` (`Unnamed: 0` in pandas) | `int64` export index | Redundant zero-based row ordinal. The physical header token is empty; pandas assigns `Unnamed: 0`. Project readers validate or drop this field. It is not a model feature. |
| 2 | `trans_date_trans_time` | String datetime | Human-readable transaction timestamp in `%Y-%m-%d %H:%M:%S` format. Used for calendar analysis only; not the causal elapsed-time clock. |
| 3 | `cc_num` | `int64` under generic inference | Sensitive card identifier. Project code reads it as a string to preserve all digits and prevent numeric interpretation. |
| 4 | `merchant` | String | High-cardinality merchant descriptor/identifier with a cosmetic `fraud_` prefix present in both target classes. The prefix is not a fraud indicator. |
| 5 | `category` | String | Transaction category with 14 observed values in each file. |
| 6 | `amt` | `float64` | Positive transaction amount with no more than two decimal places. Converted to integer cents for rolling aggregation. |
| 7 | `first` | String | Direct personal identifier; excluded from analytical samples and displays. |
| 8 | `last` | String | Direct personal identifier; excluded from analytical samples and displays. |
| 9 | `gender` | String | Categorical value in `{F, M}`. |
| 10 | `street` | String | Direct personal location attribute; excluded from analytical samples and displays. |
| 11 | `city` | String | Cardholder home city; privacy-sensitive quasi-identifier. |
| 12 | `state` | String | Cardholder home state, not merchant location. |
| 13 | `zip` | `int64` under generic inference | Postal identifier with 4–5 source digits. `src/eda.py` reads it as a string; `src/features.py` retains pandas' inferred integer representation. Canonical categorical use requires left-zero padding to five characters before encoding. |
| 14 | `lat` | `float64` | Cardholder home latitude in `[-90, 90]`. Fixed for a card within the supplied data. |
| 15 | `long` | `float64` | Cardholder home longitude in `[-180, 180]`. Fixed for a card within the supplied data. |
| 16 | `city_pop` | `int64` | Population associated with the cardholder home city. |
| 17 | `job` | String | Cardholder occupation; privacy-sensitive categorical attribute. |
| 18 | `dob` | String date | Cardholder date of birth in `%Y-%m-%d` format; direct personal attribute. |
| 19 | `trans_num` | String | Global transaction primary key matching `[0-9a-f]{32}`; unique across all 1,852,394 rows. Also used for deterministic hashing in EDA sampling. |
| 20 | `unix_time` | `int64` | Whole-second, globally nondecreasing event clock used for causal ordering and elapsed-time windows. Its absolute calendar year contains a documented offset. |
| 21 | `merch_lat` | `float64` | Merchant latitude in `[-90, 90]`. Represents transaction merchant geography. |
| 22 | `merch_long` | `float64` | Merchant longitude in `[-180, 180]`. Represents transaction merchant geography. |
| 23 | `is_fraud` | `int64` | Binary target constrained to `{0, 1}`. It does not participate in feature computation. |

## Data Constraints

### Validation status

A complete chunked scan of both files establishes the following invariants:

| Constraint | Status |
|---|---|
| Required columns and column order | Identical across both files |
| Missing entries | 0 across all 23 columns in both files |
| Target domain | Every `is_fraud` value is `0` or `1` |
| Transaction amounts | Every `amt` value is strictly positive |
| Currency precision | Values are compatible with two-decimal integer-cent conversion |
| Cardholder coordinates | Every latitude and longitude is finite and within geographic bounds |
| Merchant coordinates | Every latitude and longitude is finite and within geographic bounds |
| Raw export index | Sequential and gap-free within each file; the holdout index restarts at `0` |
| `unix_time` ordering | No backward step within either file |
| Transaction primary key | Every `trans_num` is a unique 32-character lowercase hexadecimal value across both files |
| Per-card timestamp ties | 20 groups/40 rows in the development stream and 24 groups/48 rows in the holdout; handled as unordered same-second buckets |

### Enumerated and lexical domains

- `cc_num` matches `[0-9]{11,19}` and is interpreted as an opaque string identifier.
- Raw `zip` matches `[0-9]{4,5}`; canonical categorical form is `zfill(5)`.
- `gender` is constrained to `{F, M}`.
- `category` is constrained to `{entertainment, food_dining, gas_transport, grocery_net, grocery_pos, health_fitness, home, kids_pets, misc_net, misc_pos, personal_care, shopping_net, shopping_pos, travel}`.
- Development `state` values cover `{AK, AL, AR, AZ, CA, CO, CT, DC, DE, FL, GA, HI, IA, ID, IL, IN, KS, KY, LA, MA, MD, ME, MI, MN, MO, MS, MT, NC, ND, NE, NH, NJ, NM, NV, NY, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VA, VT, WA, WI, WV, WY}`; the holdout contains the same domain except `DE`.
- `city_pop` is a strictly positive integer.
- `trans_num` matches `[0-9a-f]{32}` and is globally unique.

### Amount ranges

| File | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| `fraudTrain.csv` | $1.00 | $28,948.90 | $70.351 |
| `fraudTest.csv` | $1.00 | $22,768.11 | $69.393 |

### Cardinality and entity continuity

| Entity | Development stream | Holdout stream |
|---|---:|---:|
| Cards | 983 | 924 |
| Merchants | 693 | 693 |
| Categories | 14 | 14 |
| Genders | 2 | 2 |
| Home cities | 894 | 849 |
| Home states | 51 | 50 |
| Postal identifiers | 970 | 912 |
| Occupations | 494 | 478 |

Card continuity across the boundary is explicit:

- 908 cards occur in both files.
- 16 cards occur only in the holdout and therefore begin with zero prior state.
- 75 cards occur only in the development stream.
- All 693 merchants and all 14 transaction categories occur in both files.

The prevalence difference is recorded as contract and monitoring context only; it cannot influence fitting, selection, calibration, or threshold decisions.

## Temporal Rules

1. **Causal clock:** `unix_time` defines event ordering and all 1-hour, 6-hour, and 24-hour elapsed-time windows. Every window uses `[t - window, t)`: the lower cutoff is included, while the active timestamp and same-second peers are excluded.
2. **Partition alignment:** `fraudTest.csv` begins 48 seconds after `fraudTrain.csv` ends. The files form one chronological transaction stream rather than independent random samples.
3. **Forward-only state:** Feature state from the end of the development stream must initialize holdout transformation so the 908 continuing cards retain valid prior windows and lifetime totals. Holdout events and labels never alter features for earlier rows.
4. **Holdout isolation:** Except for aggregate contract-integrity validation, `fraudTest.csv` is excluded from EDA, preprocessing fit operations, resampling, model fitting, model selection, calibration, and threshold tuning. It is consumed only by finalized forward transformations and evaluation procedures.
5. **Cold starts:** A card absent from the incoming serialized state receives zero rolling and all-prior values until earlier events for that card exist in the active stream.
6. **Timestamp ties:** Transactions sharing `(cc_num, unix_time)` cannot observe one another. Their aggregate becomes visible only at a strictly later timestamp.
7. **Readable timestamp artifact:** Between source indices `100531` and `100532`, the development display timestamp moves from `2019-02-28 23:59:40` to `2019-02-28 00:02:34` while `unix_time` increases from `1330473580` to `1330473754`. Calendar plots may use `trans_date_trans_time`; causal calculations do not.
8. **Calendar offset artifact:** Decoding `unix_time` nominally produces the same month, day, and time seven calendar years earlier than `trans_date_trans_time`; source events on `2012-02-29` are remapped to display date `2019-02-28`. The fields are two representations of one event clock and must not be treated as independent predictive signals.

## Storage and Handling Rules

- Raw CSV files are immutable source assets and remain directly under `data/`.
- Generated feature streams and serialized velocity state are written under `data/processed/`.
- The feature CLI retains raw source columns except `Unnamed: 0` by default. Generated feature streams therefore remain privacy-sensitive and require the same access controls as the raw files.
- Names, addresses, full card identifiers, dates of birth, and transaction identifiers are excluded from notebook displays and deterministic analytical samples.
- Raw and generated transaction files are excluded from version control by `.gitignore`.
- Checksum comparison is a required pre-ingestion control. The Python readers can calculate fingerprints but do not automatically compare them with this registry, so deployment orchestration must reject an unrecognized digest before processing.
