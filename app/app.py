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
PREDICTIONS_PATH = BASE / "models" / "test_predictions.csv"

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
    if "pickup_month" not in data.columns:
        data["pickup_month"] = data["pickup_time"].dt.month
    if "day_of_week" not in data.columns:
        data["day_of_week"] = data["pickup_time"].dt.dayofweek
    if "trip_duration_min" not in data.columns:
        data["trip_duration_min"] = (
            data["drop_time"] - data["pickup_time"]
        ).dt.total_seconds() / 60

    data["day_name"] = data["pickup_time"].dt.day_name()
    data["month_name"] = data["pickup_time"].dt.month_name().str.slice(stop=3)
    return data


@st.cache_data
def load_test_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


model = load_model()
meta = load_meta()
df = load_data()
test_predictions = load_test_predictions()

model_features = meta.get("features")
if not isinstance(model_features, list) or not model_features:
    st.error("Model metadata is invalid: a non-empty 'features' list is required.")
    st.stop()

required_columns = {
    "city", "payment_method", "pickup_lat", "pickup_lng", "drop_lat", "drop_lng",
    "distance_km", "fare_amount", "status", "pickup_time", "pickup_hour",
    "pickup_month", "day_of_week"
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


def transformed_feature_effects():
    try:
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        feature_names = preprocess.get_feature_names_out()

        if hasattr(estimator, "coef_"):
            values = np.ravel(estimator.coef_)
            label = "Coefficient"
        elif hasattr(estimator, "feature_importances_"):
            values = np.ravel(estimator.feature_importances_)
            label = "Importance"
        else:
            return pd.DataFrame(), None

        if len(feature_names) != len(values):
            return pd.DataFrame(), None

        effects = pd.DataFrame({"Feature": feature_names, label: values})
        effects["AbsoluteEffect"] = effects[label].abs()
        effects["Feature"] = (
            effects["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )
        return effects.sort_values("AbsoluteEffect", ascending=False), label
    except Exception:
        return pd.DataFrame(), None


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
        pickup_lat = st.number_input(
            "Pickup latitude", -90.0, 90.0,
            float(city_df["pickup_lat"].median()), format="%.6f"
        )
        pickup_lng = st.number_input(
            "Pickup longitude", -180.0, 180.0,
            float(city_df["pickup_lng"].median()), format="%.6f"
        )
    with c3:
        st.markdown("**Drop-off location**")
        drop_lat = st.number_input(
            "Drop-off latitude", -90.0, 90.0,
            float(city_df["drop_lat"].median()), format="%.6f"
        )
        drop_lng = st.number_input(
            "Drop-off longitude", -180.0, 180.0,
            float(city_df["drop_lng"].median()), format="%.6f"
        )

    coordinate_distance = float(haversine(
        pickup_lat, pickup_lng, drop_lat, drop_lng
    ))
    st.caption(
        f"Straight-line coordinate distance: {coordinate_distance:.2f} km (reference only)."
    )

    max_distance = float(max(30, df["distance_km"].max()))
    distance = st.slider(
        "Planned trip distance (km)", 0.1, max_distance, 7.0, 0.1
    )
    st.caption(
        "The model uses the supplied planned route distance. "
        "Coordinate distance is not substituted automatically."
    )

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

    missing_prediction_features = [
        feature for feature in model_features if feature not in row.columns
    ]
    if missing_prediction_features:
        st.error(
            "Prediction form is missing model features: "
            + ", ".join(missing_prediction_features)
        )
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance", f"{distance:.2f} km")
    m2.metric("Pickup hour", f"{dt.hour:02d}:00")
    m3.metric("Weekend", "Yes" if dt.dayofweek >= 5 else "No")
    m4.metric(
        "Rush hour",
        "Yes" if dt.hour in [7, 8, 9, 16, 17, 18, 19] else "No"
    )

    if st.button("Predict fare", type="primary", width="stretch"):
        try:
            pred = float(model.predict(row[model_features])[0])
            st.success(f"Estimated fare: ${pred:,.2f}")
            st.caption(
                "Portfolio/demo estimate from the supplied 50K educational dataset; "
                "not a live Uber quote."
            )
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
        sel_status = f2.multiselect(
            "Trip status", all_status, default=all_status
        )
        sel_pay = f3.multiselect(
            "Payment method", all_payments, default=all_payments
        )

        r1, r2, r3 = st.columns(3)
        dist_rng = r1.slider(
            "Distance range (km)",
            float(df.distance_km.min()),
            float(df.distance_km.max()),
            (float(df.distance_km.min()), float(df.distance_km.max()))
        )
        fare_rng = r2.slider(
            "Fare range",
            float(df.fare_amount.min()),
            float(df.fare_amount.max()),
            (float(df.fare_amount.min()), float(df.fare_amount.max()))
        )
        hour_rng = r3.slider("Pickup hour", 0, 23, (0, 23))

        dmin, dmax = df.pickup_time.min().date(), df.pickup_time.max().date()
        date_rng = st.date_input(
            "Pickup date range",
            value=(dmin, dmax),
            min_value=dmin,
            max_value=dmax
        )

    filtered = df[
        df.city.isin(sel_city)
        & df.status.isin(sel_status)
        & df.payment_method.isin(sel_pay)
        & df.distance_km.between(*dist_rng)
        & df.fare_amount.between(*fare_rng)
        & df.pickup_hour.between(*hour_rng)
    ].copy()

    if isinstance(date_rng, (tuple, list)) and len(date_rng) == 2:
        filtered = filtered[
            filtered.pickup_time.dt.date.between(date_rng[0], date_rng[1])
        ]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trips", f"{len(filtered):,}")
    k2.metric(
        "Avg fare",
        f"${filtered.fare_amount.mean():.2f}" if len(filtered) else "—"
    )
    k3.metric(
        "Avg distance",
        f"{filtered.distance_km.mean():.2f} km" if len(filtered) else "—"
    )
    k4.metric(
        "Completion rate",
        f"{filtered.status.eq('Completed').mean() * 100:.1f}%"
        if len(filtered) else "—"
    )
    k5.metric(
        "Avg duration",
        f"{filtered.trip_duration_min.mean():.1f} min"
        if len(filtered) else "—"
    )

    if filtered.empty:
        st.warning("No trips match the selected filters.")
    else:
        st.markdown("### Core EDA")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered.fare_amount.dropna(), bins=30)
            ax.set(title="Fare distribution", xlabel="Fare", ylabel="Trips")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered.distance_km.dropna(), bins=30)
            ax.set(
                title="Trip-distance distribution",
                xlabel="Distance (km)",
                ylabel="Trips"
            )
            st.pyplot(fig)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            sample = filtered.sample(
                min(5000, len(filtered)), random_state=42
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(
                sample.distance_km, sample.fare_amount,
                alpha=.4, s=10
            )
            ax.set(
                title="Fare vs distance",
                xlabel="Distance (km)",
                ylabel="Fare"
            )
            st.pyplot(fig)
            plt.close(fig)
        with c4:
            grouped = (
                filtered.groupby("pickup_hour", as_index=False)
                .fare_amount.mean()
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grouped.pickup_hour, grouped.fare_amount, marker="o")
            ax.set(
                title="Average fare by pickup hour",
                xlabel="Pickup hour",
                ylabel="Average fare"
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Time-based fare analysis")
        t1, t2 = st.columns(2)
        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
        with t1:
            day_grouped = (
                filtered.groupby("day_name", as_index=False)
                .fare_amount.mean()
            )
            day_grouped["day_name"] = pd.Categorical(
                day_grouped["day_name"],
                categories=day_order,
                ordered=True
            )
            day_grouped = day_grouped.sort_values("day_name")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                day_grouped["day_name"].astype(str),
                day_grouped["fare_amount"]
            )
            ax.set(
                title="Average fare by day of week",
                xlabel="Day",
                ylabel="Average fare"
            )
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig)
            plt.close(fig)

        with t2:
            month_grouped = (
                filtered.groupby(
                    ["pickup_month", "month_name"], as_index=False
                )
                .fare_amount.mean()
                .sort_values("pickup_month")
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                month_grouped["month_name"],
                month_grouped["fare_amount"],
                marker="o"
            )
            ax.set(
                title="Average fare by month",
                xlabel="Month",
                ylabel="Average fare"
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Location and correlation analysis")
        l1, l2 = st.columns(2)
        with l1:
            grouped = (
                filtered.groupby("city", as_index=False)
                .fare_amount.mean()
                .sort_values("fare_amount")
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(grouped.city, grouped.fare_amount)
            ax.set(
                title="Average fare by city",
                xlabel="Average fare"
            )
            st.pyplot(fig)
            plt.close(fig)

        with l2:
            corr_columns = [
                "fare_amount", "distance_km", "pickup_hour",
                "pickup_lat", "pickup_lng", "drop_lat", "drop_lng",
                "trip_duration_min"
            ]
            corr = filtered[corr_columns].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(corr.to_numpy(), aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation matrix")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Business answers from the selected data")
        valid_distance = filtered[["distance_km", "fare_amount"]].dropna()
        fare_distance_corr = (
            valid_distance.corr().iloc[0, 1]
            if len(valid_distance) > 1 else np.nan
        )
        hour_avg = filtered.groupby("pickup_hour").fare_amount.mean()
        day_avg = filtered.groupby("day_name").fare_amount.mean()
        month_avg = filtered.groupby(
            ["pickup_month", "month_name"]
        ).fare_amount.mean()

        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "Fare-distance correlation",
            f"{fare_distance_corr:.3f}"
            if np.isfinite(fare_distance_corr) else "—"
        )
        b2.metric(
            "Highest avg-fare hour",
            f"{int(hour_avg.idxmax()):02d}:00" if len(hour_avg) else "—"
        )
        b3.metric(
            "Highest avg-fare day",
            str(day_avg.idxmax()) if len(day_avg) else "—"
        )
        if len(month_avg):
            best_month = month_avg.idxmax()[1]
        else:
            best_month = "—"
        b4.metric("Highest avg-fare month", best_month)

        st.info(
            "Passenger-count analysis is intentionally not shown because "
            "the instructor-provided 50K dataset has no passenger_count field."
        )

        st.download_button(
            "Download filtered data",
            filtered.to_csv(index=False).encode("utf-8"),
            "uber_filtered_data.csv",
            "text/csv"
        )


with model_tab:
    st.subheader("Model performance and explanation")
    metrics = pd.DataFrame(meta.get("metrics", []))
    if metrics.empty or "RMSE" not in metrics.columns:
        st.warning(
            "No packaged model metrics were found in model_metadata.json."
        )
    else:
        metrics = metrics.sort_values("RMSE")
        st.caption(
            "These are the packaged historical holdout-comparison metrics. "
            "The current train.py uses training-only 5-fold CV for model "
            "selection and an untouched final holdout evaluation."
        )
        st.dataframe(
            metrics.style.format({
                "MAE": "{:.3f}",
                "MSE": "{:.3f}",
                "RMSE": "{:.3f}",
                "R2": "{:.4f}"
            }),
            width="stretch"
        )
        best = metrics.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Packaged model", meta.get("best_model", "—"))
        c2.metric("MAE", f"{best.MAE:.3f}")
        c3.metric("MSE", f"{best.MSE:.3f}")
        c4.metric("RMSE", f"{best.RMSE:.3f}")
        c5.metric("R²", f"{best.R2:.4f}")

        st.caption(
            "MAE and RMSE are fare-unit errors; lower is better. "
            "R² measures explained variance and is not percentage accuracy."
        )

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("### Feature effect / importance")
        effects, effect_label = transformed_feature_effects()
        if effects.empty:
            st.info(
                "The packaged estimator does not expose coefficients or "
                "feature importances."
            )
        else:
            top_effects = effects.head(15).sort_values(
                "AbsoluteEffect", ascending=True
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(
                top_effects["Feature"],
                top_effects[effect_label]
            )
            ax.set(
                title=f"Top transformed features by {effect_label.lower()}",
                xlabel=effect_label
            )
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Model-level influence only; this is not a causal interpretation."
            )

    with e2:
        st.markdown("### Actual vs predicted")
        required_prediction_cols = {"actual", "prediction"}
        if (
            test_predictions.empty
            or not required_prediction_cols.issubset(test_predictions.columns)
        ):
            st.info(
                "models/test_predictions.csv is unavailable or does not contain "
                "actual and prediction columns."
            )
        else:
            plot_data = test_predictions[
                ["actual", "prediction"]
            ].dropna()
            plot_data = plot_data.sample(
                min(4000, len(plot_data)),
                random_state=42
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                plot_data["actual"],
                plot_data["prediction"],
                alpha=.4,
                s=10
            )
            low = min(
                plot_data["actual"].min(),
                plot_data["prediction"].min()
            )
            high = max(
                plot_data["actual"].max(),
                plot_data["prediction"].max()
            )
            ax.plot([low, high], [low, high], linestyle="--")
            ax.set(
                title="Actual vs predicted fares",
                xlabel="Actual fare",
                ylabel="Predicted fare"
            )
            st.pyplot(fig)
            plt.close(fig)

    st.info(
        "Training is restricted to Completed trips. IDs, trip status, "
        "actual drop time, and actual trip duration are excluded from "
        "prediction features to reduce leakage."
    )
    st.write("**Training features:**", ", ".join(model_features))


with about_tab:
    st.subheader("Project notes")
    st.markdown("""
- Source: instructor-provided `uber_trips_dataset_50k.csv` (50,000 rows).
- `passenger_count` is not present in the source and is not fabricated.
- `distance_km` is supplied by the dataset and is treated as the planned route-distance input.
- Completed rides are used for fare-model training; cancelled/no-show rides remain available for dashboard exploration.
- The dashboard now covers fare distribution, distance distribution, fare vs distance, fare by hour, fare by day, fare by month, city averages, correlations, and model explanation.
- The dataset is educational/synthetic and should not be interpreted as a production Uber pricing system.
- Post-trip fields such as actual drop time and duration are not used to estimate fare before a trip starts.
- The packaged model metadata contains historical baseline metrics; `train.py` is the authoritative reproducible training workflow for new runs.
""")
