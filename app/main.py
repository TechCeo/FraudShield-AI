"""Streamlit interface for causal transaction scoring and review."""

from __future__ import annotations

import io
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.feedback import FeedbackStore  # noqa: E402
from app.scoring import (  # noqa: E402
    MAX_BATCH_ROWS,
    PredictionResult,
    ScoringRuntime,
    utc_now_text,
)

st.set_page_config(
    page_title="Fraud Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Verifying model lineage and loading risk engines…")
def load_runtime(project_root: str) -> ScoringRuntime:
    """Load immutable scoring artifacts once per application process."""

    return ScoringRuntime.load(project_root, device_name="cpu")


@st.cache_resource
def load_feedback_store(path: str) -> FeedbackStore:
    """Open the process-shared, concurrency-safe feedback ledger."""

    return FeedbackStore(path)


@st.cache_data
def load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --shield-ink: #eaf4ff;
            --shield-muted: #91a4b8;
            --shield-panel: rgba(14, 30, 49, 0.82);
            --shield-line: rgba(142, 174, 204, 0.18);
            --shield-green: #52e6a4;
            --shield-amber: #ffc857;
            --shield-red: #ff6b7d;
        }
        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(41, 157, 129, .16), transparent 28rem),
                radial-gradient(circle at 18% 20%, rgba(36, 111, 168, .13), transparent 32rem),
                #07111f;
        }
        .block-container { max-width: 1500px; padding-top: 2rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091522 0%, #0a1828 100%);
            border-right: 1px solid var(--shield-line);
        }
        .shield-hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--shield-line);
            border-radius: 18px;
            background: linear-gradient(115deg, rgba(15, 38, 59, .96), rgba(10, 26, 44, .86));
            box-shadow: 0 18px 48px rgba(0, 0, 0, .22);
            margin-bottom: 1rem;
        }
        .shield-eyebrow {
            color: var(--shield-green);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .shield-hero h1 {
            margin: .25rem 0 .4rem;
            color: var(--shield-ink);
            font-size: clamp(2rem, 4vw, 3.35rem);
            letter-spacing: -.045em;
            line-height: 1;
        }
        .shield-hero p {
            margin: 0;
            color: var(--shield-muted);
            max-width: 62rem;
            font-size: 1rem;
        }
        .risk-card {
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            border: 1px solid var(--shield-line);
            background: var(--shield-panel);
            min-height: 150px;
        }
        .risk-card.flagged { border-color: rgba(255, 107, 125, .58); }
        .risk-card.clear { border-color: rgba(82, 230, 164, .48); }
        .risk-label {
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--shield-muted);
        }
        .risk-value {
            margin-top: .35rem;
            font-size: clamp(2.3rem, 6vw, 4.7rem);
            font-weight: 800;
            line-height: 1;
            letter-spacing: -.055em;
        }
        .risk-value.flagged { color: var(--shield-red); }
        .risk-value.clear { color: var(--shield-green); }
        .risk-status { color: var(--shield-muted); margin-top: .55rem; }
        .model-pill {
            display: inline-block;
            border: 1px solid rgba(82, 230, 164, .35);
            color: var(--shield-green);
            background: rgba(82, 230, 164, .08);
            padding: .25rem .55rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 700;
            margin: .15rem .2rem .15rem 0;
        }
        div[data-testid="stMetric"] {
            background: var(--shield-panel);
            border: 1px solid var(--shield-line);
            padding: .8rem 1rem;
            border-radius: 14px;
        }
        div[data-testid="stForm"], div[data-testid="stExpander"] {
            border-color: var(--shield-line);
            border-radius: 16px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def friendly_feature(name: str) -> str:
    replacements = {
        "num__log1p_": "",
        "num__": "",
        "freq__": "",
        "cat__": "",
        "__": " · ",
        "_": " ",
    }
    output = name
    for source, destination in replacements.items():
        output = output.replace(source, destination)
    return output.strip().title()


def mask_card(value: str) -> str:
    text = str(value)
    return f"•••• {text[-4:]}" if len(text) >= 4 else "••••"


def render_prediction(result: PredictionResult, feedback: FeedbackStore) -> None:
    state_class = "flagged" if result.fraud_flag else "clear"
    state_text = "Review required" if result.fraud_flag else "Below review threshold"
    st.markdown(
        f"""
        <div class="risk-card {state_class}">
            <div class="risk-label">Hybrid fraud probability</div>
            <div class="risk-value {state_class}">{result.fraud_probability:.1%}</div>
            <div class="risk-status">{state_text} · threshold {result.decision_threshold:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Decision",
        "FLAGGED" if result.fraud_flag else "CLEAR",
    )
    metric_columns[1].metric("Sequence context", f"{result.context_depth} events")
    metric_columns[2].metric(
        "Distance",
        f"{result.engineered_features['distance_card_merchant_km']:.1f} km",
    )
    metric_columns[3].metric(
        "24h activity",
        f"{int(result.engineered_features['cc_txn_count_prev_24h'])} transactions",
    )

    score_column, evidence_column = st.columns([0.85, 1.15])
    with score_column:
        st.subheader("Model consensus")
        component_frame = pd.DataFrame(
            {
                "Model": ["XGBoost", "FNN", "Causal LSTM", "Hybrid"],
                "Probability": [
                    result.component_probabilities["xgboost"],
                    result.component_probabilities["fnn"],
                    result.component_probabilities["lstm"],
                    result.fraud_probability,
                ],
            }
        ).set_index("Model")
        st.bar_chart(component_frame, y="Probability", height=310)
        st.caption(
            "The hybrid score combines all three model probabilities in "
            "validation-frozen log-odds space."
        )

    with evidence_column:
        st.subheader("Local risk evidence")
        drivers = pd.DataFrame(result.top_drivers)
        drivers["Signal"] = drivers["feature"].map(friendly_feature)
        drivers["Effect"] = drivers["contribution_log_odds"]
        st.bar_chart(
            drivers.set_index("Signal")[["Effect"]],
            height=310,
        )
        st.caption(
            "XGBoost log-odds contributions explain its component score. "
            "Positive values increase local risk; negative values reduce it."
        )

    with st.expander("Behavioral and lineage details"):
        behavior = pd.DataFrame(
            {
                "Signal": [
                    friendly_feature(name)
                    for name in result.engineered_features
                ],
                "Value": list(result.engineered_features.values()),
            }
        )
        st.dataframe(behavior, width="stretch", hide_index=True)
        st.code(
            f"prediction_id: {result.prediction_id}\n"
            f"context: {result.history_mode}\n"
            f"model_config: {result.model_config_sha256}",
            language="text",
        )

    st.markdown("#### Reviewer feedback")
    left, right, note = st.columns([1, 1, 2])
    if left.button(
        "Confirm Fraud",
        type="primary" if result.fraud_flag else "secondary",
        key=f"confirm_{result.prediction_id}",
        width="stretch",
    ):
        feedback.record(
            result.to_dict(),
            "confirm_fraud",
            updated_at_utc=utc_now_text(),
        )
        st.toast("Fraud confirmation saved locally.", icon="✅")
    if right.button(
        "False Positive",
        key=f"false_{result.prediction_id}",
        width="stretch",
    ):
        feedback.record(
            result.to_dict(),
            "false_positive",
            updated_at_utc=utc_now_text(),
        )
        st.toast("False-positive review saved locally.", icon="✅")
    note.caption(
        "Feedback is written to the local review ledger and is not used to "
        "change the active model automatically."
    )


def transaction_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2021-01-01 00:00:00",
                "cc_num": "2291163933867244",
                "merchant": "fraud_Kirlin and Sons",
                "category": "personal_care",
                "amt": 2.86,
                "city": "Columbia",
                "state": "SC",
                "zip": "29209",
                "lat": 33.9659,
                "long": -80.9355,
                "city_pop": 333497,
                "job": "Mechanical engineer",
                "dob": "1968-03-19",
                "merch_lat": 33.986391,
                "merch_long": -81.200714,
                "trans_num": "example-transaction-001",
                "unix_time": "",
            }
        ]
    )


def batch_summary(results: list[PredictionResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": result.prediction_id,
                "transaction_time": result.transaction["trans_date_trans_time"],
                "card": mask_card(str(result.transaction["cc_num"])),
                "amount": result.transaction["amt"],
                "fraud_probability": result.fraud_probability,
                "fraud_flag": result.fraud_flag,
                "xgboost_probability": result.component_probabilities["xgboost"],
                "fnn_probability": result.component_probabilities["fnn"],
                "lstm_probability": result.component_probabilities["lstm"],
                "context_depth": result.context_depth,
                "review": "Unreviewed",
            }
            for result in results
        ]
    )


def render_manual(runtime: ScoringRuntime, feedback: FeedbackStore) -> None:
    st.subheader("Manual transaction assessment")
    st.caption(
        "Submit raw transaction attributes. Geospatial, velocity, cumulative, "
        "and sequential features are computed by the registered causal pipeline."
    )
    categories = [
        value
        for value in runtime.preprocessor.nominal_vocabularies_["category"]
        if not value.startswith("__")
    ]
    states = [
        value
        for value in runtime.preprocessor.nominal_vocabularies_["state"]
        if not value.startswith("__")
    ]

    with st.form("manual_transaction", border=True):
        st.markdown("##### Transaction")
        c1, c2, c3, c4 = st.columns(4)
        transaction_date = c1.date_input(
            "Transaction date",
            value=date(2021, 1, 1),
        )
        transaction_time = c2.time_input(
            "Transaction time",
            value=time(0, 0),
            step=60,
        )
        amount = c3.number_input(
            "Amount",
            min_value=0.01,
            value=84.35,
            step=1.0,
            format="%.2f",
        )
        category = c4.selectbox(
            "Category",
            options=categories,
            index=categories.index("shopping_net")
            if "shopping_net" in categories
            else 0,
        )
        merchant = st.text_input(
            "Merchant",
            value="fraud_Kirlin and Sons",
            help="Unseen merchants are handled through the frozen unknown-frequency contract.",
        )

        st.markdown("##### Cardholder context")
        c1, c2, c3, c4 = st.columns(4)
        card = c1.text_input("Card number", value="2291163933867244")
        birth_date = c2.date_input(
            "Date of birth",
            value=date(1968, 3, 19),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        city = c3.text_input("City", value="Columbia")
        state = c4.selectbox(
            "State",
            options=states,
            index=states.index("SC") if "SC" in states else 0,
        )
        c1, c2, c3, c4 = st.columns(4)
        zip_code = c1.text_input("ZIP code", value="29209")
        city_population = c2.number_input(
            "City population",
            min_value=0,
            value=333497,
            step=100,
        )
        occupation = c3.text_input("Occupation", value="Mechanical engineer")
        transaction_key = c4.text_input(
            "Transaction ID",
            value="",
            placeholder="Generated when blank",
        )

        st.markdown("##### Coordinates")
        c1, c2, c3, c4 = st.columns(4)
        latitude = c1.number_input(
            "Cardholder latitude",
            min_value=-90.0,
            max_value=90.0,
            value=33.9659,
            format="%.6f",
        )
        longitude = c2.number_input(
            "Cardholder longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-80.9355,
            format="%.6f",
        )
        merchant_latitude = c3.number_input(
            "Merchant latitude",
            min_value=-90.0,
            max_value=90.0,
            value=33.986391,
            format="%.6f",
        )
        merchant_longitude = c4.number_input(
            "Merchant longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-81.200714,
            format="%.6f",
        )
        submitted = st.form_submit_button(
            "Score transaction",
            type="primary",
            width="stretch",
        )

    if submitted:
        display_time = datetime.combine(transaction_date, transaction_time)
        row = pd.DataFrame(
            [
                {
                    "trans_date_trans_time": display_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "cc_num": str(card),
                    "merchant": merchant,
                    "category": category,
                    "amt": amount,
                    "city": city,
                    "state": state,
                    "zip": str(zip_code),
                    "lat": latitude,
                    "long": longitude,
                    "city_pop": city_population,
                    "job": occupation,
                    "dob": birth_date.strftime("%Y-%m-%d"),
                    "merch_lat": merchant_latitude,
                    "merch_long": merchant_longitude,
                    "trans_num": transaction_key,
                }
            ]
        )
        try:
            with st.spinner("Computing causal context and model consensus…"):
                result = st.session_state.scoring_session.score(row, commit=True)[0]
            st.session_state.last_manual_result = result
        except Exception as exc:
            st.error(f"Transaction rejected: {exc}")

    if result := st.session_state.get("last_manual_result"):
        st.divider()
        render_prediction(result, feedback)


def render_batch(feedback: FeedbackStore) -> None:
    st.subheader("Batch review")
    st.caption(
        f"Upload up to {MAX_BATCH_ROWS:,} chronologically valid rows. "
        "Rows may interleave cards, but each card must move forward in causal time."
    )
    template_bytes = transaction_template().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV template",
        data=template_bytes,
        file_name="fraudshield_batch_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader(
        "Transaction CSV",
        type=["csv"],
        accept_multiple_files=False,
    )
    commit = st.checkbox(
        "Commit uploaded transactions to this session's history",
        value=False,
        help=(
            "Disabled keeps the session unchanged while preserving causal "
            "relationships within the uploaded file."
        ),
    )
    if uploaded is not None:
        try:
            batch = pd.read_csv(
                uploaded,
                dtype={
                    "cc_num": "string",
                    "zip": "string",
                    "trans_num": "string",
                },
            )
            st.dataframe(batch.head(25), width="stretch", hide_index=True)
            st.caption(f"{len(batch):,} uploaded rows")
        except Exception as exc:
            st.error(f"CSV could not be read: {exc}")
            batch = None
        if batch is not None and st.button(
            "Score uploaded transactions",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Scoring uploaded transactions…"):
                    results = st.session_state.scoring_session.score(
                        batch,
                        commit=commit,
                    )
                st.session_state.batch_results = results
            except Exception as exc:
                st.error(f"Batch rejected: {exc}")

    results = st.session_state.get("batch_results", [])
    if not results:
        return
    summary = batch_summary(results)
    flagged = int(summary["fraud_flag"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(summary):,}")
    c2.metric("Flagged", f"{flagged:,}")
    c3.metric("Mean risk", f"{summary['fraud_probability'].mean():.2%}")

    edited = st.data_editor(
        summary,
        width="stretch",
        hide_index=True,
        disabled=[
            column for column in summary.columns if column != "review"
        ],
        column_config={
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "review": st.column_config.SelectboxColumn(
                "Reviewer label",
                options=["Unreviewed", "Confirm Fraud", "False Positive"],
                required=True,
            ),
        },
        key="batch_review_editor",
    )
    result_by_id = {result.prediction_id: result for result in results}
    save_column, download_column = st.columns(2)
    if save_column.button(
        "Save reviewed labels",
        width="stretch",
    ):
        saved = 0
        for row in edited.to_dict(orient="records"):
            label = {
                "Confirm Fraud": "confirm_fraud",
                "False Positive": "false_positive",
            }.get(row["review"])
            if label is None:
                continue
            feedback.record(
                result_by_id[row["prediction_id"]].to_dict(),
                label,
                updated_at_utc=utc_now_text(),
            )
            saved += 1
        st.toast(f"Saved {saved:,} reviewed labels.", icon="✅")
    download_column.download_button(
        "Download scored CSV",
        data=summary.drop(columns=["review"]).to_csv(index=False).encode("utf-8"),
        file_name="fraudshield_scored_transactions.csv",
        mime="text/csv",
        width="stretch",
    )


def render_monitor(runtime: ScoringRuntime, feedback: FeedbackStore) -> None:
    st.subheader("System monitor")
    drift = load_json(
        str(PROJECT_ROOT / "artifacts" / "models" / "drift_simulation_report.json")
    )
    scenarios = drift["scenarios"]
    columns = st.columns(3)
    for column, name, label in zip(
        columns,
        ("validation", "holdout", "injected"),
        ("Reference window", "Out-of-time holdout", "Controlled shift"),
        strict=True,
    ):
        scenario = scenarios[name]
        column.metric(
            label,
            scenario["overall_status"].upper(),
            f"score PSI {scenario['prediction_drift']['psi']:.4f}",
            delta_color="off",
        )

    holdout_features = pd.DataFrame(scenarios["holdout"]["top_feature_drift"])
    holdout_features["feature"] = holdout_features["feature"].map(friendly_feature)
    st.markdown("##### Out-of-time feature drift")
    st.dataframe(
        holdout_features[
            ["feature", "psi", "status", "normalized_mean_shift"]
        ],
        width="stretch",
        hide_index=True,
    )

    quality = load_json(
        str(PROJECT_ROOT / "artifacts" / "models" / "hybrid_evaluation_matrix.json")
    )
    quality_frame = pd.DataFrame(quality["models"]).sort_values(
        "holdout_average_precision",
        ascending=False,
    )
    st.markdown("##### Registered classifier quality")
    st.dataframe(
        quality_frame[
            [
                "model_name",
                "holdout_average_precision",
                "holdout_recall",
                "holdout_precision",
                "holdout_false_positive_rate",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Context cards", f"{len(runtime.sequence_context.cards):,}")
    c2.metric("Feature width", len(runtime.preprocessor.feature_names_))
    c3.metric("Session rows", st.session_state.scoring_session.scored_rows)
    c4.metric("Reviewed predictions", feedback.count())
    st.caption(
        "Monitoring values are registered offline reference checks. "
        "They do not automatically retrain or replace the active model."
    )


def render_sidebar(runtime: ScoringRuntime, feedback: FeedbackStore) -> None:
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Shield")
        st.caption("Causal hybrid risk intelligence")
        st.markdown(
            """
            <span class="model-pill">XGBoost 56%</span>
            <span class="model-pill">LSTM 34.75%</span>
            <span class="model-pill">FNN 9.25%</span>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Runtime status**")
        st.success("Artifacts verified", icon="✅")
        st.metric(
            "Decision threshold",
            f"{runtime.engine.decision_threshold:.1%}",
        )
        st.metric("Reviewed predictions", feedback.count())
        if st.button("Reset session history", width="stretch"):
            st.session_state.scoring_session.reset()
            st.session_state.pop("last_manual_result", None)
            st.session_state.pop("batch_results", None)
            st.toast("Session history restored to the registered baseline.")
        export = feedback.export_csv()
        st.download_button(
            "Export feedback ledger",
            data=export,
            file_name="fraudshield_feedback.csv",
            mime="text/csv",
            disabled=not bool(export),
            width="stretch",
        )
        st.divider()
        st.caption(
            "Local analytical application. Decisions require human review "
            "before operational action."
        )


def main() -> None:
    inject_style()
    try:
        runtime = load_runtime(str(PROJECT_ROOT))
        feedback = load_feedback_store(
            str(PROJECT_ROOT / "artifacts" / "app" / "feedback.sqlite3")
        )
    except Exception as exc:
        st.error(f"Application runtime could not be initialized: {exc}")
        st.stop()

    if "scoring_session" not in st.session_state:
        st.session_state.scoring_session = runtime.new_session()

    render_sidebar(runtime, feedback)
    st.markdown(
        """
        <section class="shield-hero">
            <div class="shield-eyebrow">Real-time transaction intelligence</div>
            <h1>Fraud Shield AI</h1>
            <p>
                Score raw payment events through leakage-safe behavioral features,
                static and sequential models, and a validation-frozen hybrid decision.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    manual_tab, batch_tab, monitor_tab = st.tabs(
        ["Manual scoring", "Batch review", "System monitor"]
    )
    with manual_tab:
        render_manual(runtime, feedback)
    with batch_tab:
        render_batch(feedback)
    with monitor_tab:
        render_monitor(runtime, feedback)


if __name__ == "__main__":
    main()
