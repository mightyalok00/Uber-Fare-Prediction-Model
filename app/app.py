import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "uber_fare_model.pkl"
META_PATH = BASE / "models" / "model_metadata.json"
DATA_PATH = BASE / "dataset" / "uber_trips_dataset_50k_cleaned.csv"

st.set_page_config(page_title="Uber Fare Intelligence", page_icon="🚕", layout="wide")

REQUIRED_FILES = [MODEL_PATH, META_PATH, DATA_PATH]
missing = [str(p.relative_to(BASE)) for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("Required project files are missing: " + ", ".join(missing))
    st.stop()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .hero {padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(135deg,#111827,#1f2937);color:white;margin-bottom:1rem}
    .hero h1 {margin:0;font-size:2.1rem}.hero p{margin:.35rem 0 0;color:#d1d5db}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["pickup_time"] = pd.to_datetime(data["pickup_time"], errors="coerce")
    data["drop_time"] = pd.to_datetime(data["drop_time"], errors="coerce")
    if "pickup_hour" not in data.columns:
        data["pickup_hour"] = data["pickup_time"].dt.hour
    if "trip_duration_min" not in data.columns:
        data["trip_duration_min"] = (data["drop_time"] - data["pickup_time"]).dt.total_seconds() / 60
    return data


model = load_model()
meta = load_meta()
df = load_data()

required_columns = {
    "city", "payment_method", "pickup_lat", "pickup_lng", "drop_lat", "drop_lng",
    "distance_km", "fare_amount", "status", "pickup_time", "pickup_hour"
}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

st.markdown(
    '<div class="hero"><h1>🚕 Uber Fare Intelligence</h1><p>Fare prediction, business exploration, and model evidence</p></div>',
    unsafe_allow_html=True,
)

pred_tab, dash_tab, model_tab, about_tab = st.tabs(
    ["🚕 Predict Fare", "📊 Business Dashboard", "🤖 Model Performance", "ℹ️ About"]
)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


with pred_tab:
    st.subheader("Estimate a fare before the trip starts")
    cities = sorted(df["city"].dropna().unique().tolist())
    payments = sorted(df["payment_method"].dropna().unique().tolist())
    latest_pickup = df["pickup_time"].dropna().max()

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", cities)
        payment = st.selectbox("Payment method", payments)
        pickup_date = st.date_input("Pickup date", value=latest_pickup.date())
        pickup_clock = st.time_input("Pickup time")

    city_df = df[df["city"] == city]
    with c2:
        st.markdown("**Pickup location**")
        pickup_lat = st.number_input("Pickup latitude", -90.0, 90.0, float(city_df["pickup_lat"].median()), format="%.6f")
        pickup_lng = st.number_input("Pickup longitude", -180.0, 180.0, float(city_df["pickup_lng"].median()), format="%.6f")
    with c3:
        st.markdown("**Drop-off location**")
        drop_lat = st.number_input("Drop-off latitude", -90.0, 90.0, float(city_df["drop_lat"].median()), format="%.6f")
        drop_lng = st.number_input("Drop-off longitude", -180.0, 180.0, float(city_df["drop_lng"].median()), format="%.6f")

    coordinate_distance = float(haversine(pickup_lat, pickup_lng, drop_lat, drop_lng))
    st.caption(f"Straight-line coordinate distance: {coordinate_distance:.2f} km (reference only).")

    max_distance = float(max(30, df["distance_km"].max()))
    distance = st.slider("Planned trip distance (km)", 0.1, max_distance, 7.0, 0.1)
    st.caption("The model uses the supplied planned route distance. Coordinate distance is not substituted automatically.")

    dt = pd.Timestamp.combine(pickup_date, pickup_clock)
    row = pd.DataFrame([{
        "city": city,
        "payment_method": payment,
        "pickup_lat": pickup_lat,
        "pickup_lng": pickup_lng,
        "drop_lat": drop_lat,
        "drop_lng": drop_lng,
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
        try:
            pred = float(model.predict(row[meta["features"]])[0])
            st.success(f"Estimated fare: ${pred:,.2f}")
            st.caption("Portfolio/demo estimate from the supplied 50K educational dataset; not a live Uber quote.")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

with dash_tab:
    st.subheader("Explore the ride dataset")
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        all_cities = sorted(df["city"].dropna().unique())
        all_status = sorted(df["status"].dropna().unique())
        all_payments = sorted(df["payment_method"].dropna().unique())
        sel_city = f1.multiselect("City", all_cities, default=all_cities)
        sel_status = f2.multiselect("Trip status", all_status, default=all_status)
        sel_pay = f3.multiselect("Payment method", all_payments, default=all_payments)

        r1, r2, r3 = st.columns(3)
        dist_rng = r1.slider("Distance range (km)", float(df.distance_km.min()), float(df.distance_km.max()), (float(df.distance_km.min()), float(df.distance_km.max())))
        fare_rng = r2.slider("Fare range", float(df.fare_amount.min()), float(df.fare_amount.max()), (float(df.fare_amount.min()), float(df.fare_amount.max())))
        hour_rng = r3.slider("Pickup hour", 0, 23, (0, 23))

        dmin, dmax = df.pickup_time.min().date(), df.pickup_time.max().date()
        date_rng = st.date_input("Pickup date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    filtered = df[
        df.city.isin(sel_city)
        & df.status.isin(sel_status)
        & df.payment_method.isin(sel_pay)
        & df.distance_km.between(*dist_rng)
        & df.fare_amount.between(*fare_rng)
        & df.pickup_hour.between(*hour_rng)
    ].copy()

    if isinstance(date_rng, (tuple, list)) and len(date_rng) == 2:
        filtered = filtered[filtered.pickup_time.dt.date.between(date_rng[0], date_rng[1])]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trips", f"{len(filtered):,}")
    k2.metric("Avg fare", f"${filtered.fare_amount.mean():.2f}" if len(filtered) else "—")
    k3.metric("Avg distance", f"{filtered.distance_km.mean():.2f} km" if len(filtered) else "—")
    k4.metric("Completion rate", f"{filtered.status.eq('Completed').mean() * 100:.1f}%" if len(filtered) else "—")
    k5.metric("Avg duration", f"{filtered.trip_duration_min.mean():.1f} min" if len(filtered) else "—")

    if filtered.empty:
        st.warning("No trips match the selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered.fare_amount, bins=30)
            ax.set(title="Fare distribution", xlabel="Fare", ylabel="Trips")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            sample = filtered.sample(min(5000, len(filtered)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(sample.distance_km, sample.fare_amount, alpha=.4, s=10)
            ax.set(title="Fare vs distance", xlabel="Distance (km)", ylabel="Fare")
            st.pyplot(fig)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            grouped = filtered.groupby("city", as_index=False).fare_amount.mean().sort_values("fare_amount")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(grouped.city, grouped.fare_amount)
            ax.set(title="Average fare by city", xlabel="Average fare")
            st.pyplot(fig)
            plt.close(fig)
        with c4:
            grouped = filtered.groupby("pickup_hour", as_index=False).fare_amount.mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grouped.pickup_hour, grouped.fare_amount, marker="o")
            ax.set(title="Average fare by pickup hour", xlabel="Pickup hour", ylabel="Average fare")
            st.pyplot(fig)
            plt.close(fig)

        st.download_button("Download filtered data", filtered.to_csv(index=False).encode("utf-8"), "uber_filtered_data.csv", "text/csv")

with model_tab:
    st.subheader("Packaged model evidence")
    metrics = pd.DataFrame(meta.get("metrics", [])).sort_values("RMSE")
    if metrics.empty:
        st.warning("No packaged model metrics were found in model_metadata.json.")
    else:
        st.caption("These are the packaged historical holdout-comparison metrics. The current train.py uses training-only 5-fold CV for model selection and an untouched final holdout evaluation.")
        st.dataframe(metrics.style.format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.4f}"}), width="stretch")
        best = metrics.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Packaged model", meta.get("best_model", "—"))
        c2.metric("MAE", f"{best.MAE:.3f}")
        c3.metric("RMSE", f"{best.RMSE:.3f}")
        c4.metric("R²", f"{best.R2:.4f}")

    st.info("Training is restricted to Completed trips. IDs, trip status, actual drop time, and actual trip duration are excluded from prediction features to reduce leakage.")
    st.write("**Training features:**", ", ".join(meta.get("features", [])))

with about_tab:
    st.subheader("Project notes")
    st.markdown("""
- Source: supplied `uber_trips_dataset_50k.csv` (50,000 rows).
- `passenger_count` is not present in the source and is not fabricated.
- `distance_km` is supplied by the dataset and is treated as the planned route-distance input.
- Completed rides are used for fare-model training; cancelled/no-show rides remain available for dashboard exploration.
- The dataset is educational/synthetic and should not be interpreted as a production Uber pricing system.
- Post-trip fields such as actual drop time and duration are not used to estimate fare before a trip starts.
""")
