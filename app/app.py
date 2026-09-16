"""Professional Streamlit dashboard for the Uber Fare Prediction portfolio project.

The app is tied to the current coordinate-safe model bundle in ``models/``.
It validates the artifact, feature contract, estimator class, and persistence
status before rendering so a stale or incompatible model cannot be used.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "dataset" / "uber_trips_dataset_50k_cleaned.csv"
MODEL_PATH = BASE / "models" / "uber_fare_model.pkl"
META_PATH = BASE / "models" / "model_metadata.json"
CV_PATH = BASE / "models" / "cross_validation_results.csv"
FINAL_METRICS_PATH = BASE / "models" / "final_test_metrics.csv"

EXPECTED_ARTIFACT_STATUS = "coordinate_safe_final_model"
EXPECTED_MODEL = "Linear Regression"
FEATURES = [
    "city", "payment_method", "distance_km", "pickup_year", "pickup_month",
    "pickup_day", "pickup_hour", "day_of_week", "is_weekend", "is_rush_hour",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]

st.set_page_config(
    page_title="Uber Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional visual system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --panel: rgba(127,127,127,.055);
        --border: rgba(127,127,127,.20);
        --muted: #a1a1aa;
    }
    .block-container {
        max-width: 1540px;
        padding-top: 1.05rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        padding: 1.75rem 1.85rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #080808 0%, #151515 55%, #262626 100%);
        color: white;
        margin-bottom: 1.15rem;
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 18px 40px rgba(0,0,0,.18);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 1rem;
        align-items: end;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -.035em;
        line-height: 1.05;
    }
    .hero p {
        margin: .55rem 0 0;
        color: #d4d4d8;
        font-size: 1rem;
        max-width: 900px;
    }
    .eyebrow {
        font-size: .76rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #a1a1aa;
        margin-bottom: .5rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .82rem .92rem;
        background: var(--panel);
        box-shadow: 0 8px 20px rgba(0,0,0,.025);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
    }
    .section-card {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        background: var(--panel);
        margin: .35rem 0 .9rem;
    }
    .section-card strong {
        font-size: 1.04rem;
    }
    .section-card small {
        color: var(--muted);
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: .25rem 0 1rem;
    }
    .insight-card {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .95rem 1rem;
        background: var(--panel);
        min-height: 110px;
    }
    .insight-card .label {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: .35rem;
    }
    .insight-card .value {
        font-size: 1.22rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .insight-card .detail {
        color: var(--muted);
        font-size: .82rem;
        margin-top: .35rem;
    }
    .filter-chip {
        display: inline-block;
        padding: .26rem .6rem;
        margin: .12rem .15rem .12rem 0;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--panel);
        font-size: .78rem;
    }
    .subtle {
        color: var(--muted);
        font-size: .88rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(127,127,127,.14);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }
    [data-testid="stTabs"] button {
        font-weight: 600;
    }
    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; }
        .insight-grid { grid-template-columns: 1fr; }
        .hero h1 { font-size: 1.9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data/model loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned dataset and derive dashboard/model time features."""
    df = pd.read_csv(DATA_PATH)
    df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
    df["drop_time"] = pd.to_datetime(df["drop_time"], errors="coerce")
    df["pickup_year"] = df["pickup_time"].dt.year
    df["pickup_month"] = df["pickup_time"].dt.month
    df["pickup_day"] = df["pickup_time"].dt.day
    df["pickup_hour"] = df["pickup_time"].dt.hour
    df["day_of_week"] = df["pickup_time"].dt.dayofweek
    df["day_name"] = df["pickup_time"].dt.day_name()
    df["month_name"] = df["pickup_time"].dt.month_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin(RUSH_HOURS).astype(int)
    df["trip_duration_min"] = (df["drop_time"] - df["pickup_time"]).dt.total_seconds() / 60
    return df


@st.cache_resource
def load_model():
    """Load the repository's serialized final regression pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    """Load JSON metadata when present."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load an evidence CSV when present."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def validate_model_bundle(model, meta: dict) -> list[str]:
    """Validate that Streamlit is using the approved final model contract."""
    errors: list[str] = []
    if meta.get("artifact_status") != EXPECTED_ARTIFACT_STATUS:
        errors.append(f"artifact_status must be '{EXPECTED_ARTIFACT_STATUS}'.")
    if meta.get("best_model") != EXPECTED_MODEL:
        errors.append(f"Expected '{EXPECTED_MODEL}', got {meta.get('best_model')!r}.")
    if meta.get("features") != FEATURES:
        errors.append("Metadata feature list does not match the Streamlit feature contract.")
    if meta.get("persistence_roundtrip") != "PASS":
        errors.append("Saved-model persistence verification is not marked PASS.")
    if not hasattr(model, "named_steps"):
        errors.append("Saved artifact is not the expected scikit-learn Pipeline.")
    elif "preprocess" not in model.named_steps or "model" not in model.named_steps:
        errors.append("Pipeline must contain 'preprocess' and 'model' steps.")
    elif model.named_steps["model"].__class__.__name__ != "LinearRegression":
        errors.append("Final estimator is not LinearRegression.")
    return errors


@st.cache_data
def build_holdout_predictions(_model) -> pd.DataFrame:
    """Recreate the fixed holdout and score it with the committed model."""
    from sklearn.model_selection import train_test_split

    completed = data[data["status"].eq("Completed")].dropna(subset=FEATURES + ["fare_amount"]).copy()
    _, X_test, _, y_test = train_test_split(
        completed[FEATURES],
        completed["fare_amount"],
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
    prediction = _model.predict(X_test)
    return pd.DataFrame({"actual": y_test.to_numpy(), "prediction": prediction})


def get_feature_effects(model) -> tuple[pd.DataFrame, str | None]:
    """Extract transformed coefficients/importances for model diagnostics."""
    try:
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        names = preprocess.get_feature_names_out()
        if hasattr(estimator, "coef_"):
            values, label = np.ravel(estimator.coef_), "Coefficient"
        elif hasattr(estimator, "feature_importances_"):
            values, label = np.ravel(estimator.feature_importances_), "Importance"
        else:
            return pd.DataFrame(), None

        effects = pd.DataFrame({"Feature": names, label: values})
        effects["AbsoluteEffect"] = effects[label].abs()
        effects["Feature"] = (
            effects["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )
        return effects.sort_values("AbsoluteEffect", ascending=False), label
    except Exception:
        return pd.DataFrame(), None


def delta_value(current: float, baseline: float, decimals: int = 2, suffix: str = "") -> str:
    """Format a metric delta against an unfiltered baseline."""
    if pd.isna(current) or pd.isna(baseline):
        return "n/a"
    return f"{current - baseline:+.{decimals}f}{suffix}"


def plot_frame(title: str, xlabel: str = "", ylabel: str = ""):
    """Create a compact consistent Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ---------------------------------------------------------------------------
# Initialize the source-of-truth assets
# ---------------------------------------------------------------------------
try:
    data = load_data()
    meta = load_json(META_PATH)
    model = load_model()
    validation_errors = validate_model_bundle(model, meta)
    if validation_errors:
        st.error("The deployed model bundle is stale or inconsistent with the current project model.")
        for error in validation_errors:
            st.write(f"- {error}")
        st.stop()
    test_predictions = build_holdout_predictions(model)
    cv_results = load_optional_csv(CV_PATH)
    final_metrics = load_optional_csv(FINAL_METRICS_PATH)
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()

cities = sorted(data["city"].dropna().unique().tolist())
statuses = sorted(data["status"].dropna().unique().tolist())
payments = sorted(data["payment_method"].dropna().unique().tolist())
min_fare, max_fare = float(data["fare_amount"].min()), float(data["fare_amount"].max())
positive_distance = data.loc[data["distance_km"] > 0, "distance_km"]
min_distance, max_distance = float(positive_distance.min()), float(positive_distance.max())
date_min, date_max = data["pickup_time"].min().date(), data["pickup_time"].max().date()

# ---------------------------------------------------------------------------
# Page identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Machine Learning Portfolio · Operations Analytics</div>
          <h1>Uber Fare Intelligence</h1>
          <p>Executive dashboard for fare behavior, trip operations, model performance, and new-trip fare simulation using the validated 50K educational dataset.</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

header_cols = st.columns(2)
header_cols[0].metric("Training rows", f"{meta.get('training_rows', 0):,}")
header_cols[1].metric("Holdout rows", f"{meta.get('testing_rows', 0):,}")

# ---------------------------------------------------------------------------
# Sidebar control center
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Analytics filters")
st.sidebar.caption("Refine the Business Dashboard without changing the prediction model.")

if st.sidebar.button("Reset all filters", use_container_width=True):
    for key in [
        "filter_city", "filter_status", "filter_payment", "filter_days",
        "filter_fare", "filter_distance", "filter_hour", "filter_timing", "filter_dates",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

selected_cities = st.sidebar.multiselect("City", cities, default=cities, key="filter_city")
selected_statuses = st.sidebar.multiselect("Trip status", statuses, default=statuses, key="filter_status")
selected_payments = st.sidebar.multiselect("Payment method", payments, default=payments, key="filter_payment")
selected_days = st.sidebar.multiselect("Day of week", DAY_ORDER, default=DAY_ORDER, key="filter_days")

with st.sidebar.expander("Advanced ranges", expanded=True):
    fare_range = st.slider(
        "Fare range", min_value=min_fare, max_value=max_fare,
        value=(min_fare, max_fare), key="filter_fare",
    )
    distance_range = st.slider(
        "Distance (km)", min_value=min_distance, max_value=max_distance,
        value=(min_distance, max_distance), key="filter_distance",
    )
    hour_range = st.slider(
        "Pickup hour", min_value=0, max_value=23, value=(0, 23), key="filter_hour",
    )
    timing_filter = st.radio(
        "Traffic timing", ["All", "Rush hour", "Non-rush hour"],
        key="filter_timing",
    )
    date_range = st.date_input(
        "Pickup date range", value=(date_min, date_max),
        min_value=date_min, max_value=date_max, key="filter_dates",
    )

filtered = data[
    data["city"].isin(selected_cities)
    & data["status"].isin(selected_statuses)
    & data["payment_method"].isin(selected_payments)
    & data["day_name"].isin(selected_days)
    & data["fare_amount"].between(*fare_range)
    & data["distance_km"].between(*distance_range)
    & data["pickup_hour"].between(*hour_range)
].copy()

if timing_filter == "Rush hour":
    filtered = filtered[filtered["is_rush_hour"].eq(1)]
elif timing_filter == "Non-rush hour":
    filtered = filtered[filtered["is_rush_hour"].eq(0)]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[filtered["pickup_time"].dt.date.between(start_date, end_date)]

st.sidebar.divider()
coverage = len(filtered) / max(len(data), 1)
st.sidebar.metric("Rows in view", f"{len(filtered):,}", delta=f"{coverage:.1%} coverage")
st.sidebar.caption("Filters apply only to the Business Dashboard tab.")

predict_tab, dashboard_tab, model_tab, data_tab, about_tab = st.tabs(
    ["🚕 Predict Fare", "📊 Business Dashboard", "🤖 Model Performance", "🧪 Data Quality", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Predict Fare
# ---------------------------------------------------------------------------
with predict_tab:
    st.subheader("Fare estimator")
    st.markdown(
        '<div class="section-card"><strong>Scenario simulator</strong><br><small>Enter supported pre-trip inputs. The app scores the committed final Linear Regression pipeline; it is not a live Uber quote.</small></div>',
        unsafe_allow_html=True,
    )

    latest_pickup = data["pickup_time"].dropna().max()
    left, right = st.columns([1.05, .95], gap="large")

    with left:
        st.markdown("#### Trip inputs")
        city = st.selectbox("City", cities, key="predict_city")
        payment = st.selectbox("Payment method", payments, key="predict_payment")
        pickup_date = st.date_input("Pickup date", value=latest_pickup.date(), key="predict_date")
        pickup_clock = st.time_input(
            "Pickup time",
            value=latest_pickup.time().replace(second=0, microsecond=0),
            key="predict_time",
        )

    with right:
        st.markdown("#### Route profile")
        prediction_max_distance = float(max(30, data["distance_km"].max()))
        distance = st.slider(
            "Planned trip distance (km)",
            min_value=0.1,
            max_value=prediction_max_distance,
            value=7.0,
            step=0.1,
            key="predict_distance",
        )
        dt = pd.Timestamp.combine(pickup_date, pickup_clock)
        p1, p2, p3 = st.columns(3)
        p1.metric("Day", dt.day_name())
        p2.metric("Hour", f"{dt.hour:02d}:00")
        p3.metric("Timing", "Rush" if dt.hour in RUSH_HOURS else "Standard")
        st.caption("Uses the supplied planned route-distance feature (`distance_km`).")

    row = pd.DataFrame([{
        "city": city,
        "payment_method": payment,
        "distance_km": distance,
        "pickup_year": dt.year,
        "pickup_month": dt.month,
        "pickup_day": dt.day,
        "pickup_hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "is_weekend": int(dt.dayofweek >= 5),
        "is_rush_hour": int(dt.hour in RUSH_HOURS),
    }])

    st.markdown("#### Scenario summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Distance", f"{distance:.1f} km")
    s2.metric("City", city)
    s3.metric("Payment", payment)
    s4.metric("Model", meta.get("best_model", "Saved model"))

    if st.button("Predict Fare", type="primary", use_container_width=True):
        prediction = float(model.predict(row[FEATURES])[0])
        if np.isfinite(prediction):
            st.markdown(
                f'<div class="section-card"><strong>Estimated fare: ${prediction:,.2f}</strong><br><small>Educational model estimate based on the supplied 50K dataset.</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("The model returned a non-finite prediction.")

# ---------------------------------------------------------------------------
# Business Dashboard
# ---------------------------------------------------------------------------
with dashboard_tab:
    st.subheader("Executive business dashboard")
    st.caption("All metrics and visualizations below respond to the sidebar filters.")

    if filtered.empty:
        st.warning("No rows match the selected filters. Broaden the filter ranges or reset all filters.")
    else:
        baseline_avg_fare = float(data["fare_amount"].mean())
        baseline_avg_distance = float(data["distance_km"].mean())
        baseline_completion = float(data["status"].eq("Completed").mean() * 100)

        avg_fare = float(filtered["fare_amount"].mean())
        avg_distance = float(filtered["distance_km"].mean())
        completion = float(filtered["status"].eq("Completed").mean() * 100)
        total_fare = float(filtered["fare_amount"].sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Trips", f"{len(filtered):,}", delta=f"{coverage:.1%} of data")
        k2.metric("Average fare", f"${avg_fare:.2f}", delta=delta_value(avg_fare, baseline_avg_fare))
        k3.metric("Average distance", f"{avg_distance:.2f} km", delta=delta_value(avg_distance, baseline_avg_distance, suffix=" km"))
        k4.metric("Completion rate", f"{completion:.1f}%", delta=delta_value(completion, baseline_completion, suffix=" pp"))
        k5.metric("Fare value", f"${total_fare:,.0f}")

        completed_filtered = filtered[filtered["status"].eq("Completed")].copy()
        hour_avg = filtered.groupby("pickup_hour")["fare_amount"].mean()
        city_avg = filtered.groupby("city")["fare_amount"].mean()
        pay_counts = filtered["payment_method"].value_counts()

        peak_hour = int(hour_avg.idxmax()) if not hour_avg.empty else 0
        top_city = str(city_avg.idxmax()) if not city_avg.empty else "N/A"
        top_payment = str(pay_counts.idxmax()) if not pay_counts.empty else "N/A"

        st.markdown("#### Decision snapshot")
        st.markdown(
            f"""
            <div class="insight-grid">
              <div class="insight-card">
                <div class="label">Peak average-fare hour</div>
                <div class="value">{peak_hour:02d}:00</div>
                <div class="detail">Highest mean fare within the current filtered view.</div>
              </div>
              <div class="insight-card">
                <div class="label">Highest average-fare city</div>
                <div class="value">{top_city}</div>
                <div class="detail">Relative to the selected filters and date range.</div>
              </div>
              <div class="insight-card">
                <div class="label">Most-used payment method</div>
                <div class="value">{top_payment}</div>
                <div class="detail">Most frequent payment method in the current slice.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Fare and route behavior")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig, ax = plot_frame("Fare distribution", "Fare", "Trips")
            ax.hist(filtered["fare_amount"].dropna(), bins=28)
            ax.axvline(avg_fare, linestyle="--", linewidth=1.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            sample = filtered.dropna(subset=["distance_km", "fare_amount"])
            sample = sample.sample(min(5000, len(sample)), random_state=42)
            fig, ax = plot_frame("Fare vs supplied trip distance", "Distance (km)", "Fare")
            ax.scatter(sample["distance_km"], sample["fare_amount"], alpha=.28, s=12)
            if len(sample) >= 2:
                x = sample["distance_km"].to_numpy()
                y = sample["fare_amount"].to_numpy()
                slope, intercept = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, slope * xline + intercept, linewidth=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### Time intelligence")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            hourly = filtered.groupby("pickup_hour", as_index=False)["fare_amount"].mean()
            fig, ax = plot_frame("Average fare by pickup hour", "Hour", "Average fare")
            ax.plot(hourly["pickup_hour"], hourly["fare_amount"], marker="o", linewidth=1.7, markersize=4)
            ax.set_xticks(range(0, 24, 3))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c4:
            day_avg = filtered.groupby("day_name", as_index=False)["fare_amount"].mean()
            day_avg["day_name"] = pd.Categorical(day_avg["day_name"], categories=DAY_ORDER, ordered=True)
            day_avg = day_avg.sort_values("day_name")
            fig, ax = plot_frame("Average fare by weekday", "Day", "Average fare")
            ax.bar(day_avg["day_name"].astype(str), day_avg["fare_amount"])
            ax.tick_params(axis="x", rotation=28)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### Operations mix")
        c5, c6, c7 = st.columns(3, gap="large")
        with c5:
            status_counts = filtered["status"].value_counts()
            fig, ax = plot_frame("Trip status mix", "Status", "Trips")
            ax.bar(status_counts.index.astype(str), status_counts.values)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c6:
            city_view = filtered.groupby("city", as_index=False).agg(
                Trips=("trip_id", "count"), Avg_Fare=("fare_amount", "mean")
            ).sort_values("Avg_Fare", ascending=False)
            fig, ax = plot_frame("Average fare by city", "City", "Average fare")
            ax.bar(city_view["city"].astype(str), city_view["Avg_Fare"])
            ax.tick_params(axis="x", rotation=28)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c7:
            pay_view = filtered["payment_method"].value_counts()
            fig, ax = plot_frame("Payment method usage", "Payment", "Trips")
            ax.bar(pay_view.index.astype(str), pay_view.values)
            ax.tick_params(axis="x", rotation=28)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### Filtered data diagnostics")
        d1, d2 = st.columns([.8, 1.2], gap="large")
        with d1:
            if len(completed_filtered) > 1:
                numeric_cols = ["fare_amount", "distance_km", "pickup_hour", "pickup_month", "day_of_week"]
                st.dataframe(
                    completed_filtered[numeric_cols].corr().round(3),
                    use_container_width=True,
                    height=245,
                )
        with d2:
            preview_cols = [
                "trip_id", "city", "status", "payment_method",
                "distance_km", "fare_amount", "pickup_time",
            ]
            st.dataframe(
                filtered[preview_cols].sort_values("pickup_time", ascending=False).head(50),
                use_container_width=True,
                hide_index=True,
                height=245,
            )

# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------
with model_tab:
    st.subheader("Model performance and validation")
    st.markdown(
        '<div class="section-card"><strong>Final model contract</strong><br><small>Model selection uses training-only 5-fold cross-validation. The final selected pipeline is evaluated once on the untouched holdout set.</small></div>',
        unsafe_allow_html=True,
    )

    if not final_metrics.empty:
        result = final_metrics.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{result['MAE']:.3f}")
        b.metric("MSE", f"{result['MSE']:.3f}")
        c.metric("RMSE", f"{result['RMSE']:.3f}")
        d.metric("R²", f"{result['R2']:.3f}")

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("#### Cross-validation comparison")
        if not cv_results.empty:
            display_cols = [c for c in cv_results.columns if c != "Unnamed: 0"]
            st.dataframe(cv_results[display_cols].round(4), use_container_width=True, hide_index=True)
            ordered = cv_results.sort_values("CV_RMSE_Mean", ascending=True)
            fig, ax = plot_frame("Mean CV RMSE by model", "Mean CV RMSE", "Model")
            ax.barh(ordered["Model"], ordered["CV_RMSE_Mean"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with right:
        st.markdown("#### Active model bundle")
        st.write(f"**Selected model:** {meta.get('best_model', 'Unknown')}")
        st.write(f"**Artifact status:** `{meta.get('artifact_status', 'Unknown')}`")
        st.write(f"**Selection rule:** {meta.get('selection_method', 'Training-only CV')}")
        st.write(f"**Persistence test:** {meta.get('persistence_roundtrip', 'Unknown')}")
        st.write(f"**Training Python:** {meta.get('python', 'Unknown')}")
        st.caption("Active features")
        st.code(", ".join(meta.get("features", FEATURES)), language=None)

    effects, effect_label = get_feature_effects(model)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if not effects.empty and effect_label:
            st.markdown("#### Feature influence")
            top = effects.head(15).sort_values("AbsoluteEffect")
            fig, ax = plot_frame("Top transformed coefficients", effect_label, "Feature")
            ax.barh(top["Feature"], top[effect_label])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Coefficient magnitude indicates fitted model influence, not causation.")
    with c2:
        if not test_predictions.empty:
            st.markdown("#### Actual vs predicted")
            sample = test_predictions.sample(min(5000, len(test_predictions)), random_state=42)
            lower = float(min(sample["actual"].min(), sample["prediction"].min()))
            upper = float(max(sample["actual"].max(), sample["prediction"].max()))
            fig, ax = plot_frame("Untouched holdout", "Actual fare", "Predicted fare")
            ax.scatter(sample["actual"], sample["prediction"], alpha=.28, s=12)
            ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------
with data_tab:
    st.subheader("Data quality and assignment diagnostics")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{meta.get('raw_rows', len(data)):,}")
    q2.metric("Clean rows", f"{meta.get('valid_clean_rows', 0):,}")
    q3.metric("Modeling rows", f"{meta.get('completed_model_rows', 0):,}")
    q4.metric("Fare–distance corr.", f"{meta.get('fare_vs_supplied_distance_correlation', np.nan):.3f}")

    st.markdown("#### Coordinate-distance validation")
    d1, d2 = st.columns(2)
    d1.metric(
        "Haversine vs supplied distance correlation",
        f"{meta.get('haversine_vs_supplied_distance_correlation', np.nan):.5f}",
    )
    d2.metric(
        "Rows within 0.1 km",
        f"{meta.get('haversine_within_0_1_km_share', 0) * 100:.2f}%",
    )

    st.markdown(
        '<div class="section-card"><strong>Why coordinates are diagnostic-only</strong><br><small>The pickup/drop coordinates are range-valid but internally inconsistent with the supplied route-distance field. Haversine distance is therefore retained for validation, not used as an active prediction feature.</small></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "`passenger_count` is absent from the instructor-provided 50K CSV. Passenger-count distribution, "
        "fare-vs-passenger analysis, and passenger-count significance are therefore correctly reported as N/A."
    )

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with about_tab:
    st.subheader("Project scope")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### What this project demonstrates")
        st.markdown(
            """
            - End-to-end regression workflow on a 50K educational trip dataset
            - Explicit data-quality validation and leakage-aware feature selection
            - Training-only cross-validation for model comparison
            - Untouched holdout evaluation
            - Saved-pipeline persistence verification
            - Interactive business analytics and fare simulation in Streamlit
            """
        )
    with right:
        st.markdown("#### Important limitations")
        st.markdown(
            """
            - `passenger_count` is not available and is never fabricated
            - Coordinates are excluded from active prediction because they conflict with supplied `distance_km`
            - The dataset is educational/synthetic, not production Uber data
            - Real-world fares may depend on surge, traffic, tolls, ride class, weather, and demand
            """
        )

    st.divider()
    st.caption("Built as a portfolio project to demonstrate data science, model validation, business analysis, and deployment discipline.")
