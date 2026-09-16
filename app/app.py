"""Streamlit dashboard and prediction app for the Uber Fare Prediction project.

This app is intentionally tied to the latest coordinate-safe model bundle in
`models/`. At startup it validates the metadata, feature contract, model type,
and persistence status so an older/legacy model cannot be used silently.
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

st.set_page_config(page_title="Uber Fare Intelligence", page_icon="🚕", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.1rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .35rem 0 0; color: #d1d5db;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px; padding: .7rem .8rem;
    }
    .section-note {
        border-left: 4px solid #6b7280; padding: .55rem .8rem;
        background: rgba(128,128,128,.07); border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned dataset and derive analysis/model time features."""
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
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["trip_duration_min"] = (df["drop_time"] - df["pickup_time"]).dt.total_seconds() / 60
    return df


@st.cache_resource
def load_model():
    """Load the repository's serialized final pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    """Load JSON metadata."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load an evidence CSV when present."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def validate_model_bundle(model, meta: dict) -> list[str]:
    """Validate that Streamlit is using the latest approved model contract."""
    errors: list[str] = []

    if meta.get("artifact_status") != EXPECTED_ARTIFACT_STATUS:
        errors.append(
            f"artifact_status must be '{EXPECTED_ARTIFACT_STATUS}', got {meta.get('artifact_status')!r}."
        )
    if meta.get("best_model") != EXPECTED_MODEL:
        errors.append(f"Expected '{EXPECTED_MODEL}', got {meta.get('best_model')!r}.")
    if meta.get("features") != FEATURES:
        errors.append("Metadata feature list does not match the current Streamlit feature contract.")
    if meta.get("persistence_roundtrip") != "PASS":
        errors.append("Saved-model persistence verification is not marked PASS.")

    if not hasattr(model, "named_steps"):
        errors.append("Saved artifact is not the expected scikit-learn Pipeline.")
    else:
        if "preprocess" not in model.named_steps or "model" not in model.named_steps:
            errors.append("Pipeline must contain 'preprocess' and 'model' steps.")
        else:
            estimator_name = model.named_steps["model"].__class__.__name__
            if estimator_name != "LinearRegression":
                errors.append(f"Expected LinearRegression estimator, got {estimator_name}.")

    return errors


@st.cache_data
def build_holdout_predictions(_model) -> pd.DataFrame:
    """Recreate the fixed untouched holdout and score it using the saved model."""
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
            values = np.ravel(estimator.coef_)
            label = "Coefficient"
        elif hasattr(estimator, "feature_importances_"):
            values = np.ravel(estimator.feature_importances_)
            label = "Importance"
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


# Load the latest model bundle and fail clearly if the deployment is stale.
try:
    data = load_data()
    meta = load_json(META_PATH)
    model = load_model()
    validation_errors = validate_model_bundle(model, meta)
    if validation_errors:
        st.error("The deployed model bundle is stale or inconsistent with the latest project model.")
        for error in validation_errors:
            st.write(f"- {error}")
        st.stop()

    test_predictions = build_holdout_predictions(model)
    cv_results = load_optional_csv(CV_PATH)
    final_metrics = load_optional_csv(FINAL_METRICS_PATH)
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <h1>🚕 Uber Fare Intelligence</h1>
      <p>Fare prediction, business exploration, and validated model evidence from the instructor-provided 50K dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Show the active model bundle so reviewers can verify the deployment at a glance.
model_version_cols = st.columns(4)
model_version_cols[0].metric("Active model", meta.get("best_model", "Unknown"))
model_version_cols[1].metric("Artifact", "Final coordinate-safe")
model_version_cols[2].metric("Training rows", f"{meta.get('training_rows', 0):,}")
model_version_cols[3].metric("Holdout rows", f"{meta.get('testing_rows', 0):,}")

# Sidebar filters affect business EDA only.
st.sidebar.header("📌 Dashboard Filters")
st.sidebar.caption("These filters affect the Business Dashboard only.")

cities = sorted(data["city"].dropna().unique().tolist())
statuses = sorted(data["status"].dropna().unique().tolist())
payments = sorted(data["payment_method"].dropna().unique().tolist())

selected_cities = st.sidebar.multiselect("City", cities, default=cities)
selected_statuses = st.sidebar.multiselect("Trip status", statuses, default=statuses)
selected_payments = st.sidebar.multiselect("Payment method", payments, default=payments)

min_fare = float(data["fare_amount"].min())
max_fare = float(data["fare_amount"].max())
fare_range = st.sidebar.slider("Fare range", min_value=min_fare, max_value=max_fare, value=(min_fare, max_fare))

positive_distance = data.loc[data["distance_km"] > 0, "distance_km"]
min_distance = float(positive_distance.min())
max_distance = float(positive_distance.max())
distance_range = st.sidebar.slider(
    "Distance range (km)", min_value=min_distance, max_value=max_distance,
    value=(min_distance, max_distance)
)

date_min = data["pickup_time"].min().date()
date_max = data["pickup_time"].max().date()
date_range = st.sidebar.date_input(
    "Pickup date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

filtered = data[
    data["city"].isin(selected_cities)
    & data["status"].isin(selected_statuses)
    & data["payment_method"].isin(selected_payments)
    & data["fare_amount"].between(*fare_range)
    & data["distance_km"].between(*distance_range)
].copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[filtered["pickup_time"].dt.date.between(start_date, end_date)]

st.sidebar.divider()
st.sidebar.metric("Filtered trips", f"{len(filtered):,}")
if st.sidebar.button("Reset guidance"):
    st.sidebar.info("Use the widget reset controls or reload the page to restore all defaults.")

predict_tab, dashboard_tab, model_tab, data_tab, about_tab = st.tabs(
    ["🚕 Predict Fare", "📊 Business Dashboard", "🤖 Model Performance", "🧪 Data Quality", "ℹ️ About"]
)

with predict_tab:
    st.subheader("Estimate a fare before the trip is completed")
    st.markdown(
        '<div class="section-note">This tab uses the latest saved coordinate-safe Linear Regression pipeline. '
        'Coordinates are excluded because they conflict with the supplied route-distance field; passenger count is absent.</div>',
        unsafe_allow_html=True,
    )

    latest_pickup = data["pickup_time"].dropna().max()
    left, right = st.columns(2, gap="large")

    with left:
        city = st.selectbox("City", cities)
        payment = st.selectbox("Payment method", payments)
        pickup_date = st.date_input("Pickup date", value=latest_pickup.date())
        pickup_clock = st.time_input("Pickup time", value=latest_pickup.time().replace(second=0, microsecond=0))

    with right:
        prediction_max_distance = float(max(30, data["distance_km"].max()))
        distance = st.slider(
            "Planned trip distance (km)", min_value=0.1,
            max_value=prediction_max_distance, value=7.0, step=0.1,
        )
        st.caption("Uses the supplied planned route-distance field (`distance_km`).")
        dt = pd.Timestamp.combine(pickup_date, pickup_clock)
        st.metric("Day", dt.day_name())
        st.metric("Trip timing", "Rush hour" if dt.hour in [7, 8, 9, 16, 17, 18, 19] else "Non-rush hour")

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
        "is_rush_hour": int(dt.hour in [7, 8, 9, 16, 17, 18, 19]),
    }])

    st.markdown("#### Prediction summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distance", f"{distance:.1f} km")
    k2.metric("Pickup hour", f"{dt.hour:02d}:00")
    k3.metric("Weekend", "Yes" if dt.dayofweek >= 5 else "No")
    k4.metric("Model", meta.get("best_model", "Saved pipeline"))

    if st.button("Predict Fare", type="primary", use_container_width=True):
        prediction = float(model.predict(row[FEATURES])[0])
        if np.isfinite(prediction):
            st.success(f"Estimated fare: ${prediction:,.2f}")
            st.caption("Educational model estimate — not a live Uber quote.")
        else:
            st.error("The model returned a non-finite prediction.")

with dashboard_tab:
    st.subheader("Business Dashboard")
    st.caption("All charts in this tab respond to the sidebar filters.")

    if filtered.empty:
        st.warning("No rows match the selected filters. Broaden the sidebar filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trips", f"{len(filtered):,}")
        m2.metric("Average fare", f"${filtered['fare_amount'].mean():.2f}")
        m3.metric("Average distance", f"{filtered['distance_km'].mean():.2f} km")
        m4.metric("Completion rate", f"{filtered['status'].eq('Completed').mean() * 100:.1f}%")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered["fare_amount"].dropna(), bins=30)
            ax.set(title="Fare Distribution", xlabel="Fare", ylabel="Trips")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            scatter = filtered.dropna(subset=["distance_km", "fare_amount"])
            sample = scatter.sample(min(5000, len(scatter)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(sample["distance_km"], sample["fare_amount"], alpha=.35, s=12)
            ax.set(title="Fare vs Supplied Trip Distance", xlabel="Distance (km)", ylabel="Fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            hourly = filtered.groupby("pickup_hour", as_index=False)["fare_amount"].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hourly["pickup_hour"], hourly["fare_amount"], marker="o")
            ax.set(title="Average Fare by Pickup Hour", xlabel="Pickup hour", ylabel="Average fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c4:
            day_avg = filtered.groupby("day_name", as_index=False)["fare_amount"].mean()
            day_avg["day_name"] = pd.Categorical(day_avg["day_name"], categories=DAY_ORDER, ordered=True)
            day_avg = day_avg.sort_values("day_name")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(day_avg["day_name"].astype(str), day_avg["fare_amount"])
            ax.set(title="Average Fare by Day of Week", xlabel="Day", ylabel="Average fare")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c5, c6 = st.columns(2, gap="large")
        with c5:
            monthly = filtered.groupby("pickup_month", as_index=False)["fare_amount"].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(monthly["pickup_month"], monthly["fare_amount"])
            ax.set(title="Average Fare by Month", xlabel="Month", ylabel="Average fare")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c6:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered["distance_km"].dropna(), bins=30)
            ax.set(title="Trip-Distance Distribution", xlabel="Distance (km)", ylabel="Trips")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("#### Correlation Matrix")
        completed = filtered[filtered["status"].eq("Completed")]
        numeric_cols = ["fare_amount", "distance_km", "pickup_hour", "pickup_month", "day_of_week"]
        if len(completed) > 1:
            st.dataframe(completed[numeric_cols].corr().round(3), use_container_width=True)

with model_tab:
    st.subheader("Latest Final Model Performance")
    st.write(f"**Selected model:** {meta.get('best_model', 'Unknown')}")
    st.write(f"**Artifact status:** `{meta.get('artifact_status', 'Unknown')}`")
    st.write(f"**Selection rule:** {meta.get('selection_method', 'Training-only cross-validation')}")
    st.write("**Active features:** " + ", ".join(meta.get("features", FEATURES)))
    st.success("Streamlit is using the final coordinate-safe saved pipeline from `models/uber_fare_model.pkl`.")

    if not final_metrics.empty:
        result = final_metrics.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{result['MAE']:.3f}")
        b.metric("MSE", f"{result['MSE']:.3f}")
        c.metric("RMSE", f"{result['RMSE']:.3f}")
        d.metric("R²", f"{result['R2']:.3f}")

    if not cv_results.empty:
        st.markdown("#### Training-only 5-Fold Cross-Validation")
        display_cols = [c for c in cv_results.columns if c != "Unnamed: 0"]
        st.dataframe(cv_results[display_cols].round(4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ordered = cv_results.sort_values("CV_RMSE_Mean", ascending=True)
        ax.barh(ordered["Model"], ordered["CV_RMSE_Mean"])
        ax.set(title="Model Comparison by Mean CV RMSE", xlabel="Mean CV RMSE", ylabel="Model")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    effects, effect_label = get_feature_effects(model)
    if not effects.empty and effect_label:
        st.markdown("#### Feature Influence / Coefficients")
        top = effects.head(15).sort_values("AbsoluteEffect")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["Feature"], top[effect_label])
        ax.set(xlabel=effect_label, ylabel="Transformed feature")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Coefficient magnitude is model influence after preprocessing, not proof of causation.")

    if not test_predictions.empty:
        st.markdown("#### Actual vs Predicted Fares")
        sample = test_predictions.sample(min(5000, len(test_predictions)), random_state=42)
        lower = float(min(sample["actual"].min(), sample["prediction"].min()))
        upper = float(max(sample["actual"].max(), sample["prediction"].max()))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sample["actual"], sample["prediction"], alpha=.35, s=12)
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set(title="Actual vs Predicted — Untouched Holdout", xlabel="Actual fare", ylabel="Predicted fare")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with data_tab:
    st.subheader("Data Quality & Assignment Diagnostics")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{meta.get('raw_rows', len(data)):,}")
    q2.metric("Modeling rows", f"{meta.get('completed_model_rows', 0):,}")
    q3.metric("Missing passenger_count", "Yes")
    q4.metric("Fare–distance corr.", f"{meta.get('fare_vs_supplied_distance_correlation', np.nan):.3f}")

    st.markdown("#### Coordinate-derived distance validation")
    st.write(
        "The assignment asks for coordinate-derived trip distance. It was calculated as a data-quality diagnostic, "
        "but it is not used for prediction because it does not agree with the supplied `distance_km` field."
    )
    d1, d2 = st.columns(2)
    d1.metric(
        "Haversine vs supplied distance correlation",
        f"{meta.get('haversine_vs_supplied_distance_correlation', np.nan):.4f}",
    )
    d2.metric(
        "Rows within 0.1 km",
        f"{meta.get('haversine_within_0_1_km_share', 0) * 100:.2f}%",
    )
    st.warning(
        "`passenger_count` is not in the instructor-provided 50K CSV. Passenger-count distribution, "
        "fare-vs-passenger analysis, and statistical conclusions are therefore correctly marked N/A."
    )

with about_tab:
    st.subheader("Project Scope")
    st.markdown(
        """
        - Uses the instructor-provided **50,000-row educational dataset**.
        - Streamlit loads the repository's **latest final coordinate-safe saved model**.
        - Predicts `fare_amount` using information available before or at trip start.
        - Model selection uses **5-fold cross-validation only on the training partition**.
        - The selected pipeline is evaluated **once on the untouched holdout set**.
        - `passenger_count` is absent and is never fabricated.
        - Coordinates are retained for validation but excluded from active prediction because they conflict with supplied `distance_km`.
        - Results are dataset-specific and are not claims about real-world Uber pricing.
        """
    )
