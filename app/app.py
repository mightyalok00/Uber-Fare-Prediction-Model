"""Interactive Streamlit dashboard for the Uber Fare Prediction project.

The app is intentionally tied to the latest coordinate-safe model bundle in
``models/``.  At startup it validates the metadata, feature contract, model
class, and persistence status so a stale model cannot be used silently.
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

st.set_page_config(page_title="Uber Fare Intelligence", page_icon="🚕", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1rem; padding-bottom: 2.4rem;}
    .hero {
        padding: 1.65rem 1.75rem; border-radius: 22px;
        background: linear-gradient(135deg, #09090b 0%, #18181b 48%, #27272a 100%);
        color: white; margin-bottom: 1.05rem; box-shadow: 0 14px 34px rgba(0,0,0,.18);
    }
    .hero h1 {margin: 0; font-size: 2.35rem; letter-spacing: -.03em;}
    .hero p {margin: .45rem 0 0; color: #d4d4d8; font-size: 1.02rem;}
    .eyebrow {font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: #a1a1aa; margin-bottom: .45rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20); border-radius: 15px;
        padding: .78rem .88rem; background: rgba(128,128,128,.035);
    }
    .section-note {
        border-left: 4px solid #71717a; padding: .65rem .9rem;
        background: rgba(128,128,128,.065); border-radius: 9px; margin-bottom: .8rem;
    }
    .filter-chip {
        display: inline-block; padding: .28rem .62rem; margin: .15rem .2rem .15rem 0;
        border-radius: 999px; border: 1px solid rgba(128,128,128,.28);
        font-size: .82rem; background: rgba(128,128,128,.06);
    }
    .insight-card {
        border: 1px solid rgba(128,128,128,.20); border-radius: 16px;
        padding: 1rem 1.1rem; min-height: 118px; background: rgba(128,128,128,.035);
    }
    .insight-card h4 {margin: 0 0 .35rem 0; font-size: .93rem; color: #a1a1aa;}
    .insight-card strong {font-size: 1.18rem;}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.12);}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """Load the repository's serialized final pipeline."""
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
    """Validate that Streamlit is using the approved current model contract."""
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
    """Recreate the fixed untouched holdout and score it with the saved model."""
    from sklearn.model_selection import train_test_split

    completed = data[data["status"].eq("Completed")].dropna(subset=FEATURES + ["fare_amount"]).copy()
    _, X_test, _, y_test = train_test_split(
        completed[FEATURES], completed["fare_amount"],
        test_size=0.20, random_state=42, shuffle=True,
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
            effects["Feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
        )
        return effects.sort_values("AbsoluteEffect", ascending=False), label
    except Exception:
        return pd.DataFrame(), None


def safe_delta(current: float, baseline: float, suffix: str = "") -> str:
    """Return a compact delta label against the unfiltered dataset baseline."""
    if pd.isna(current) or pd.isna(baseline):
        return "n/a"
    return f"{current - baseline:+.2f}{suffix}"


# Load and verify all source-of-truth assets before rendering the dashboard.
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

# Hero / project identity.
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Portfolio Analytics · Fare Intelligence</div>
      <h1>🚕 Uber Fare Intelligence Cockpit</h1>
      <p>Explore trip economics, operational patterns, fare drivers, and the validated prediction model from one interactive workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_version_cols = st.columns(4)
model_version_cols[0].metric("Active model", meta.get("best_model", "Unknown"))
model_version_cols[1].metric("Model status", "Validated")
model_version_cols[2].metric("Training rows", f"{meta.get('training_rows', 0):,}")
model_version_cols[3].metric("Holdout rows", f"{meta.get('testing_rows', 0):,}")

# ------------------------------ Sidebar filters ------------------------------
st.sidebar.markdown("## 🎛️ Analytics Control Center")
st.sidebar.caption("Filters instantly update the Business Dashboard.")

if st.sidebar.button("↻ Reset all filters", use_container_width=True):
    for key in [
        "filter_city", "filter_status", "filter_payment", "filter_days", "filter_fare",
        "filter_distance", "filter_hour", "filter_timing", "filter_dates",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

selected_cities = st.sidebar.multiselect("🌆 City", cities, default=cities, key="filter_city")
selected_statuses = st.sidebar.multiselect("🚦 Trip status", statuses, default=statuses, key="filter_status")
selected_payments = st.sidebar.multiselect("💳 Payment method", payments, default=payments, key="filter_payment")
selected_days = st.sidebar.multiselect("📅 Day of week", DAY_ORDER, default=DAY_ORDER, key="filter_days")

st.sidebar.markdown("### Range filters")
fare_range = st.sidebar.slider(
    "Fare range", min_value=min_fare, max_value=max_fare,
    value=(min_fare, max_fare), key="filter_fare",
)
distance_range = st.sidebar.slider(
    "Distance range (km)", min_value=min_distance, max_value=max_distance,
    value=(min_distance, max_distance), key="filter_distance",
)
hour_range = st.sidebar.slider(
    "Pickup hour", min_value=0, max_value=23, value=(0, 23), key="filter_hour",
)
timing_filter = st.sidebar.radio(
    "Traffic timing", ["All", "Rush hour", "Non-rush hour"], horizontal=True, key="filter_timing",
)
date_range = st.sidebar.date_input(
    "Pickup date range", value=(date_min, date_max), min_value=date_min, max_value=date_max,
    key="filter_dates",
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
st.sidebar.metric("Rows in view", f"{len(filtered):,}", delta=f"{len(filtered) - len(data):+,} vs all")
st.sidebar.caption(f"Coverage: {len(filtered) / max(len(data), 1):.1%} of cleaned trips")

predict_tab, dashboard_tab, model_tab, data_tab, about_tab = st.tabs(
    ["🚕 Predict Fare", "📊 Business Dashboard", "🤖 Model Performance", "🧪 Data Quality", "ℹ️ About"]
)

# ------------------------------ Prediction tab -------------------------------
with predict_tab:
    st.subheader("New-trip fare simulator")
    st.markdown(
        '<div class="section-note">Adjust supported pre-trip inputs and score them with the final saved Linear Regression pipeline. '
        'Coordinates and passenger count are intentionally not used.</div>',
        unsafe_allow_html=True,
    )

    latest_pickup = data["pickup_time"].dropna().max()
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        city = st.selectbox("City", cities, key="predict_city")
        payment = st.selectbox("Payment method", payments, key="predict_payment")
        pickup_date = st.date_input("Pickup date", value=latest_pickup.date(), key="predict_date")
        pickup_clock = st.time_input(
            "Pickup time", value=latest_pickup.time().replace(second=0, microsecond=0), key="predict_time"
        )
    with right:
        prediction_max_distance = float(max(30, data["distance_km"].max()))
        distance = st.slider(
            "Planned trip distance (km)", min_value=0.1, max_value=prediction_max_distance,
            value=7.0, step=0.1, key="predict_distance",
        )
        dt = pd.Timestamp.combine(pickup_date, pickup_clock)
        p1, p2 = st.columns(2)
        p1.metric("Day", dt.day_name())
        p2.metric("Timing", "Rush" if dt.hour in RUSH_HOURS else "Standard")
        st.caption("Distance uses the supplied route-distance feature that is associated with fare in this educational dataset.")

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distance", f"{distance:.1f} km")
    k2.metric("Pickup", f"{dt.hour:02d}:00")
    k3.metric("Weekend", "Yes" if dt.dayofweek >= 5 else "No")
    k4.metric("Model", meta.get("best_model", "Saved pipeline"))

    if st.button("Predict Fare", type="primary", use_container_width=True):
        prediction = float(model.predict(row[FEATURES])[0])
        if np.isfinite(prediction):
            st.markdown(
                f'<div class="insight-card"><h4>Estimated fare</h4><strong>${prediction:,.2f}</strong><br>'
                '<span style="color:#a1a1aa">Educational estimate — not a live Uber quote.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("The model returned a non-finite prediction.")

# ------------------------------- Dashboard tab -------------------------------
with dashboard_tab:
    title_col, context_col = st.columns([1.35, .65], gap="large")
    with title_col:
        st.subheader("Business & Operations Dashboard")
        st.caption("Every visual and KPI below responds to the Analytics Control Center filters.")
    with context_col:
        st.markdown(
            f"<div class='filter-chip'>🌆 {len(selected_cities)} cities</div>"
            f"<div class='filter-chip'>🚦 {len(selected_statuses)} statuses</div>"
            f"<div class='filter-chip'>🕒 {hour_range[0]:02d}:00–{hour_range[1]:02d}:00</div>"
            f"<div class='filter-chip'>⚡ {timing_filter}</div>",
            unsafe_allow_html=True,
        )

    if filtered.empty:
        st.warning("No trips match the selected filters. Use ‘Reset all filters’ or broaden the ranges.")
    else:
        base_avg_fare = float(data["fare_amount"].mean())
        base_avg_distance = float(data["distance_km"].mean())
        current_avg_fare = float(filtered["fare_amount"].mean())
        current_avg_distance = float(filtered["distance_km"].mean())
        completion_rate = float(filtered["status"].eq("Completed").mean() * 100)
        completed_view = filtered[filtered["status"].eq("Completed")].copy()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trips", f"{len(filtered):,}", delta=f"{len(filtered)/len(data):.1%} coverage")
        m2.metric("Avg fare", f"${current_avg_fare:.2f}", delta=safe_delta(current_avg_fare, base_avg_fare, " vs all"))
        m3.metric("Avg distance", f"{current_avg_distance:.2f} km", delta=safe_delta(current_avg_distance, base_avg_distance, " km"))
        m4.metric("Completion rate", f"{completion_rate:.1f}%")
        m5.metric("Total fare value", f"${filtered['fare_amount'].sum():,.0f}")

        # Auto-generated insight strip makes the dashboard feel like a decision cockpit.
        peak_hour = int(filtered.groupby("pickup_hour")["fare_amount"].mean().idxmax())
        top_city = str(filtered.groupby("city")["fare_amount"].mean().idxmax())
        top_payment = str(filtered["payment_method"].value_counts().idxmax())
        median_fare = float(filtered["fare_amount"].median())
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.markdown(f"<div class='insight-card'><h4>Peak fare hour</h4><strong>{peak_hour:02d}:00</strong><br><span style='color:#a1a1aa'>Highest average fare in current view</span></div>", unsafe_allow_html=True)
        with i2:
            st.markdown(f"<div class='insight-card'><h4>Highest avg-fare city</h4><strong>{top_city}</strong><br><span style='color:#a1a1aa'>Within active filters</span></div>", unsafe_allow_html=True)
        with i3:
            st.markdown(f"<div class='insight-card'><h4>Top payment method</h4><strong>{top_payment}</strong><br><span style='color:#a1a1aa'>Most frequent in current view</span></div>", unsafe_allow_html=True)
        with i4:
            st.markdown(f"<div class='insight-card'><h4>Median fare</h4><strong>${median_fare:.2f}</strong><br><span style='color:#a1a1aa'>Less sensitive to fare outliers</span></div>", unsafe_allow_html=True)

        st.markdown("### Fare & demand patterns")
        c1, c2 = st.columns([1.05, .95], gap="large")
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.hist(filtered["fare_amount"].dropna(), bins=32)
            ax.axvline(current_avg_fare, linestyle="--", linewidth=1.4, label=f"Mean ${current_avg_fare:.2f}")
            ax.set(title="Fare Distribution", xlabel="Fare", ylabel="Trips")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            hourly = filtered.groupby("pickup_hour", as_index=False).agg(
                avg_fare=("fare_amount", "mean"), trips=("trip_id", "count")
            )
            fig, ax = plt.subplots(figsize=(7.3, 4.4))
            ax.plot(hourly["pickup_hour"], hourly["avg_fare"], marker="o")
            ax.set(title="Average Fare by Pickup Hour", xlabel="Pickup hour", ylabel="Average fare")
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            scatter = filtered.dropna(subset=["distance_km", "fare_amount"])
            sample = scatter.sample(min(5000, len(scatter)), random_state=42)
            fig, ax = plt.subplots(figsize=(7.3, 4.4))
            ax.scatter(sample["distance_km"], sample["fare_amount"], alpha=.3, s=12)
            if len(sample) > 1:
                x = sample["distance_km"].to_numpy()
                y = sample["fare_amount"].to_numpy()
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, linewidth=1.6)
            ax.set(title="Fare vs Supplied Trip Distance", xlabel="Distance (km)", ylabel="Fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c4:
            status_mix = filtered["status"].value_counts()
            fig, ax = plt.subplots(figsize=(7.3, 4.4))
            ax.bar(status_mix.index.astype(str), status_mix.values)
            ax.set(title="Trip Status Mix", xlabel="Status", ylabel="Trips")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("### Segment intelligence")
        c5, c6 = st.columns(2, gap="large")
        with c5:
            city_perf = filtered.groupby("city", as_index=False).agg(
                trips=("trip_id", "count"), avg_fare=("fare_amount", "mean"), avg_distance=("distance_km", "mean")
            ).sort_values("avg_fare", ascending=False)
            st.markdown("#### City performance")
            st.dataframe(city_perf.round(2), use_container_width=True, hide_index=True)
        with c6:
            payment_perf = filtered.groupby("payment_method", as_index=False).agg(
                trips=("trip_id", "count"), avg_fare=("fare_amount", "mean")
            ).sort_values("trips", ascending=False)
            st.markdown("#### Payment mix")
            st.dataframe(payment_perf.round(2), use_container_width=True, hide_index=True)

        c7, c8 = st.columns(2, gap="large")
        with c7:
            day_avg = filtered.groupby("day_name", as_index=False).agg(
                avg_fare=("fare_amount", "mean"), trips=("trip_id", "count")
            )
            day_avg["day_name"] = pd.Categorical(day_avg["day_name"], categories=DAY_ORDER, ordered=True)
            day_avg = day_avg.sort_values("day_name")
            fig, ax = plt.subplots(figsize=(7.3, 4.3))
            ax.bar(day_avg["day_name"].astype(str), day_avg["avg_fare"])
            ax.set(title="Average Fare by Day", xlabel="Day", ylabel="Average fare")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c8:
            rush_summary = filtered.assign(
                timing=np.where(filtered["is_rush_hour"].eq(1), "Rush hour", "Non-rush hour")
            ).groupby("timing", as_index=False).agg(
                trips=("trip_id", "count"), avg_fare=("fare_amount", "mean")
            )
            fig, ax = plt.subplots(figsize=(7.3, 4.3))
            ax.bar(rush_summary["timing"], rush_summary["avg_fare"])
            ax.set(title="Rush vs Non-Rush Average Fare", xlabel="Timing", ylabel="Average fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("🔎 Explore filtered trip records"):
            show_cols = [
                "trip_id", "city", "status", "payment_method", "pickup_time",
                "distance_km", "fare_amount", "day_name", "pickup_hour",
            ]
            st.dataframe(filtered[show_cols].sort_values("pickup_time", ascending=False), use_container_width=True, hide_index=True)

        with st.expander("📐 Filtered correlation matrix"):
            numeric_cols = ["fare_amount", "distance_km", "pickup_hour", "pickup_month", "day_of_week"]
            if len(completed_view) > 1:
                st.dataframe(completed_view[numeric_cols].corr().round(3), use_container_width=True)
            else:
                st.info("Select more Completed trips to calculate correlations.")

# ----------------------------- Model evidence tab ----------------------------
with model_tab:
    st.subheader("Model Performance & Evidence")
    st.markdown(
        '<div class="section-note">This section reports the saved final model, training-only cross-validation evidence, and untouched holdout diagnostics.</div>',
        unsafe_allow_html=True,
    )
    st.write(f"**Selected model:** {meta.get('best_model', 'Unknown')}")
    st.write(f"**Selection rule:** {meta.get('selection_method', 'Training-only cross-validation')}")
    st.write("**Active features:** " + ", ".join(meta.get("features", FEATURES)))

    if not final_metrics.empty:
        result = final_metrics.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{result['MAE']:.3f}")
        b.metric("MSE", f"{result['MSE']:.3f}")
        c.metric("RMSE", f"{result['RMSE']:.3f}")
        d.metric("R²", f"{result['R2']:.3f}")

    left, right = st.columns(2, gap="large")
    with left:
        if not cv_results.empty:
            st.markdown("#### 5-fold CV comparison")
            display_cols = [c for c in cv_results.columns if c != "Unnamed: 0"]
            st.dataframe(cv_results[display_cols].round(4), use_container_width=True, hide_index=True)
            ordered = cv_results.sort_values("CV_RMSE_Mean", ascending=True)
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            ax.barh(ordered["Model"], ordered["CV_RMSE_Mean"])
            ax.set(title="Mean CV RMSE", xlabel="RMSE", ylabel="Model")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with right:
        if not test_predictions.empty:
            st.markdown("#### Actual vs predicted")
            sample = test_predictions.sample(min(5000, len(test_predictions)), random_state=42)
            lower = float(min(sample["actual"].min(), sample["prediction"].min()))
            upper = float(max(sample["actual"].max(), sample["prediction"].max()))
            fig, ax = plt.subplots(figsize=(7.2, 5.25))
            ax.scatter(sample["actual"], sample["prediction"], alpha=.3, s=12)
            ax.plot([lower, upper], [lower, upper], linestyle="--")
            ax.set(title="Untouched Holdout", xlabel="Actual fare", ylabel="Predicted fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    effects, effect_label = get_feature_effects(model)
    if not effects.empty and effect_label:
        st.markdown("#### Feature influence")
        top = effects.head(15).sort_values("AbsoluteEffect")
        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.barh(top["Feature"], top[effect_label])
        ax.set(xlabel=effect_label, ylabel="Transformed feature")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Coefficient magnitude reflects fitted model influence after preprocessing; it is not proof of causation.")

# ------------------------------ Data-quality tab -----------------------------
with data_tab:
    st.subheader("Data Quality & Assignment Diagnostics")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{meta.get('raw_rows', len(data)):,}")
    q2.metric("Modeling rows", f"{meta.get('completed_model_rows', 0):,}")
    q3.metric("passenger_count", "Unavailable")
    q4.metric("Fare–distance corr.", f"{meta.get('fare_vs_supplied_distance_correlation', np.nan):.3f}")

    st.markdown("#### Coordinate-distance integrity")
    st.write(
        "Haversine distance is calculated for assignment/data-quality validation, but is excluded from prediction because "
        "the supplied coordinates are not internally consistent with `distance_km`."
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Haversine vs supplied corr.", f"{meta.get('haversine_vs_supplied_distance_correlation', np.nan):.5f}")
    d2.metric("Within 0.1 km", f"{meta.get('haversine_within_0_1_km_share', 0) * 100:.2f}%")
    d3.metric("Persistence test", meta.get("persistence_roundtrip", "Unknown"))

    st.warning(
        "`passenger_count` is not in the instructor-provided 50K CSV. Passenger-count distribution, "
        "fare-vs-passenger analysis, and significance conclusions remain correctly marked N/A."
    )

    st.markdown("#### Dataset status snapshot")
    snapshot = pd.DataFrame({
        "Check": ["Missing values", "Duplicate trip IDs", "Valid cleaned rows", "Completed modeling rows"],
        "Result": [
            int(data.isna().sum().sum()),
            int(data["trip_id"].duplicated().sum()),
            f"{meta.get('valid_clean_rows', len(data)):,}",
            f"{meta.get('completed_model_rows', 0):,}",
        ],
    })
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

# -------------------------------- About tab ----------------------------------
with about_tab:
    st.subheader("Project Scope")
    st.markdown(
        """
        - Uses the instructor-provided **50,000-row educational dataset**.
        - Streamlit loads the repository's **final coordinate-safe saved model**.
        - Predicts `fare_amount` using information available before or at trip start.
        - Model selection uses **5-fold cross-validation only on the training partition**.
        - The selected pipeline is evaluated **once on the untouched holdout set**.
        - `passenger_count` is absent and is never fabricated.
        - Coordinates remain diagnostic-only because they conflict with supplied `distance_km`.
        - Dashboard filters are analytical only and do not alter the saved model or holdout evaluation.
        - Results are dataset-specific and are not claims about real-world Uber pricing.
        """
    )
    st.info(
        "Dashboard design: interactive control center + fare simulator + operational segmentation + model evidence + data-quality audit."
    )
