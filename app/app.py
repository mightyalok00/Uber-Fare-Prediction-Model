import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "uber_fare_model.pkl"
META_PATH = BASE / "models" / "model_metadata.json"
DATA_PATH = BASE / "dataset" / "uber_trips_dataset_50k_cleaned.csv"

CORRECTED_FEATURES = [
    "city", "payment_method", "distance_km", "pickup_year", "pickup_month",
    "pickup_day", "pickup_hour", "day_of_week", "is_weekend", "is_rush_hour"
]
LEGACY_COORDS = ["pickup_lat", "pickup_lng", "drop_lat", "drop_lng"]

st.set_page_config(page_title="Uber Fare Intelligence", page_icon="🚕", layout="wide")

if not DATA_PATH.exists():
    st.error("Required dataset file is missing.")
    st.stop()


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["pickup_time"] = pd.to_datetime(data["pickup_time"], errors="coerce")
    data["drop_time"] = pd.to_datetime(data["drop_time"], errors="coerce")
    data["pickup_year"] = data["pickup_time"].dt.year
    data["pickup_month"] = data["pickup_time"].dt.month
    data["pickup_day"] = data["pickup_time"].dt.day
    data["pickup_hour"] = data["pickup_time"].dt.hour
    data["day_of_week"] = data["pickup_time"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["is_rush_hour"] = data["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    data["day_name"] = data["pickup_time"].dt.day_name()
    data["month_name"] = data["pickup_time"].dt.month_name().str.slice(stop=3)
    data["trip_duration_min"] = (
        data["drop_time"] - data["pickup_time"]
    ).dt.total_seconds() / 60
    return data


@st.cache_data
def load_meta():
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def build_corrected_pipeline():
    cat = ["city", "payment_method"]
    num = [c for c in CORRECTED_FEATURES if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", LinearRegression()),
    ])


@st.cache_resource
def load_prediction_model(data):
    meta = load_meta()
    packaged_features = meta.get("features", [])

    if MODEL_PATH.exists() and packaged_features and not any(c in packaged_features for c in LEGACY_COORDS):
        return joblib.load(MODEL_PATH), packaged_features, "packaged coordinate-safe model", None

    completed = data[data["status"].eq("Completed")].dropna(
        subset=CORRECTED_FEATURES + ["fare_amount"]
    ).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        completed[CORRECTED_FEATURES], completed["fare_amount"],
        test_size=0.20, random_state=42, shuffle=True
    )
    model = build_corrected_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, pred)),
    }
    return model, CORRECTED_FEATURES, "runtime coordinate-safe Linear Regression", metrics


df = load_data()
model, model_features, model_source, runtime_metrics = load_prediction_model(df)
meta = load_meta()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(135deg,#111827,#1f2937);color:white;margin-bottom:1rem}
    .hero h1 {margin:0}.hero p {margin:.35rem 0 0;color:#d1d5db}
    </style>
    <div class="hero"><h1>🚕 Uber Fare Intelligence</h1><p>Coordinate-safe fare prediction, business exploration, and model evidence</p></div>
    """,
    unsafe_allow_html=True,
)

pred_tab, dash_tab, model_tab, about_tab = st.tabs(
    ["🚕 Predict Fare", "📊 Business Dashboard", "🤖 Model Performance", "ℹ️ About"]
)

with pred_tab:
    st.subheader("Estimate a fare before the trip starts")
    st.info(
        "Pickup/drop-off coordinates are intentionally not used for prediction because the supplied "
        "coordinates are not internally consistent with the dataset's distance_km field."
    )

    cities = sorted(df["city"].dropna().unique().tolist())
    payments = sorted(df["payment_method"].dropna().unique().tolist())
    latest_pickup = df["pickup_time"].dropna().max()

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("City", cities)
        payment = st.selectbox("Payment method", payments)
        pickup_date = st.date_input("Pickup date", value=latest_pickup.date())
        pickup_clock = st.time_input("Pickup time")
    with c2:
        max_distance = float(max(30, df["distance_km"].max()))
        distance = st.slider("Planned trip distance (km)", 0.1, max_distance, 7.0, 0.1)
        st.caption("Uses the supplied route-distance field; no coordinate-derived distance is substituted.")

    dt = pd.Timestamp.combine(pickup_date, pickup_clock)
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance", f"{distance:.2f} km")
    m2.metric("Pickup hour", f"{dt.hour:02d}:00")
    m3.metric("Weekend", "Yes" if dt.dayofweek >= 5 else "No")
    m4.metric("Rush hour", "Yes" if dt.hour in [7, 8, 9, 16, 17, 18, 19] else "No")

    if st.button("Predict fare", type="primary", width="stretch"):
        pred = float(model.predict(row[model_features])[0])
        st.success(f"Estimated fare: ${pred:,.2f}")
        st.caption(f"Prediction source: {model_source}. Educational dataset; not a live Uber quote.")

with dash_tab:
    st.subheader("Explore the ride dataset")
    f1, f2, f3 = st.columns(3)
    all_cities = sorted(df.city.dropna().unique())
    all_status = sorted(df.status.dropna().unique())
    all_payments = sorted(df.payment_method.dropna().unique())
    sel_city = f1.multiselect("City", all_cities, default=all_cities)
    sel_status = f2.multiselect("Status", all_status, default=all_status)
    sel_pay = f3.multiselect("Payment", all_payments, default=all_payments)

    filtered = df[
        df.city.isin(sel_city)
        & df.status.isin(sel_status)
        & df.payment_method.isin(sel_pay)
    ].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trips", f"{len(filtered):,}")
    k2.metric("Average fare", f"${filtered.fare_amount.mean():.2f}" if len(filtered) else "—")
    k3.metric("Average distance", f"{filtered.distance_km.mean():.2f} km" if len(filtered) else "—")
    k4.metric("Completion rate", f"{filtered.status.eq('Completed').mean() * 100:.1f}%" if len(filtered) else "—")

    if filtered.empty:
        st.warning("No rows match the selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered["fare_amount"], bins=30)
            ax.set(title="Fare distribution", xlabel="Fare", ylabel="Trips")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            sample = filtered.sample(min(5000, len(filtered)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(sample["distance_km"], sample["fare_amount"], alpha=.4, s=10)
            ax.set(title="Fare vs supplied distance", xlabel="Distance (km)", ylabel="Fare")
            st.pyplot(fig)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            hourly = filtered.groupby("pickup_hour", as_index=False)["fare_amount"].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hourly["pickup_hour"], hourly["fare_amount"], marker="o")
            ax.set(title="Average fare by pickup hour", xlabel="Hour", ylabel="Average fare")
            st.pyplot(fig)
            plt.close(fig)
        with c4:
            monthly = filtered.groupby("pickup_month", as_index=False)["fare_amount"].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(monthly["pickup_month"], monthly["fare_amount"])
            ax.set(title="Average fare by month", xlabel="Month", ylabel="Average fare")
            st.pyplot(fig)
            plt.close(fig)

        completed = filtered[filtered.status.eq("Completed")]
        if len(completed) > 1:
            corr = completed[["fare_amount", "distance_km", "pickup_hour", "pickup_month", "day_of_week"]].corr()
            st.markdown("### Numeric correlation matrix")
            st.dataframe(corr.round(3), width="stretch")

with model_tab:
    st.subheader("Model evidence")
    st.write(f"**Active prediction model:** {model_source}")
    st.write("**Prediction features:** " + ", ".join(model_features))
    st.success("Latitude/longitude are excluded from active prediction inputs.")

    if runtime_metrics:
        st.markdown("### Coordinate-safe runtime holdout metrics")
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{runtime_metrics['MAE']:.3f}")
        b.metric("MSE", f"{runtime_metrics['MSE']:.3f}")
        c.metric("RMSE", f"{runtime_metrics['RMSE']:.3f}")
        d.metric("R²", f"{runtime_metrics['R2']:.3f}")
        st.caption("These metrics are for the coordinate-safe Linear Regression rebuilt on the same fixed 80/20 split. Full model selection remains in train.py with training-only 5-fold CV.")

    historical = meta.get("metrics", [])
    if historical:
        st.markdown("### Historical packaged baseline comparison")
        st.dataframe(pd.DataFrame(historical), width="stretch")
        st.caption("Historical metrics are retained for reproducibility; the older packaged artifact included coordinate columns and is not used for live prediction when detected as legacy.")

with about_tab:
    st.subheader("Project scope and limitations")
    st.markdown(
        """
- The repository uses the instructor-provided 50,000-row educational dataset.
- `passenger_count` is absent, so passenger-count analysis is not fabricated.
- The supplied `distance_km` is strongly related to fare and is retained as the distance feature.
- Pickup/drop-off coordinates are internally inconsistent with `distance_km`, so they are excluded from active prediction.
- `train.py` performs training-only 5-fold cross-validation, selects the lowest mean CV RMSE, and evaluates the selected model once on the untouched holdout set.
- Results are dataset-specific and should not be interpreted as real-world Uber pricing claims.
        """
    )
