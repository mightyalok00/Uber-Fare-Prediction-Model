"""Train and export the final coordinate-safe Uber fare regression pipeline.

Design principles
-----------------
- Never use the final test set for model selection.
- Fit preprocessing inside each CV fold through sklearn Pipeline.
- Do not fabricate passenger_count: it is absent from the supplied CSV.
- Calculate Haversine distance for assignment/data-quality validation only.
- Exclude pickup/drop coordinates from the active model because they conflict with supplied distance_km.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent

# Only features available before/at trip start and supported by the supplied data.
FEATURES = [
    "city", "payment_method", "distance_km", "pickup_year", "pickup_month",
    "pickup_day", "pickup_hour", "day_of_week", "is_weekend", "is_rush_hour",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized straight-line great-circle distance used only for validation."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def regression_metrics(y_true, prediction):
    """Return standard regression metrics requested by the assignment."""
    assert np.isfinite(prediction).all(), "Model generated non-finite predictions."
    mse = mean_squared_error(y_true, prediction)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, prediction)),
    }


def load_training_data():
    """Validate the instructor dataset and return clean Completed rides."""
    raw_path = BASE / "dataset" / "uber_trips_dataset_50k.csv"
    raw = pd.read_csv(raw_path)

    # Parse timestamps once so cleaning and engineered features use consistent datetimes.
    raw["pickup_time"] = pd.to_datetime(raw["pickup_time"], errors="coerce")
    raw["drop_time"] = pd.to_datetime(raw["drop_time"], errors="coerce")

    # Validate values explicitly rather than silently dropping arbitrary rows.
    valid = (
        (raw["fare_amount"] > 0)
        & (raw["distance_km"] > 0)
        & (raw["drop_time"] > raw["pickup_time"])
        & raw["pickup_lat"].between(-90, 90)
        & raw["drop_lat"].between(-90, 90)
        & raw["pickup_lng"].between(-180, 180)
        & raw["drop_lng"].between(-180, 180)
    )

    # The business objective concerns completed historical rides.
    data = raw.loc[valid & raw["status"].eq("Completed")].drop_duplicates().copy()
    assert data["trip_id"].is_unique, "Duplicate trip IDs remain after cleaning."

    # Engineer only time features derivable at trip start.
    pickup = data["pickup_time"]
    data["pickup_year"] = pickup.dt.year
    data["pickup_month"] = pickup.dt.month
    data["pickup_day"] = pickup.dt.day
    data["pickup_hour"] = pickup.dt.hour
    data["day_of_week"] = pickup.dt.dayofweek
    data["is_weekend"] = pickup.dt.dayofweek.isin([5, 6]).astype(int)
    data["is_rush_hour"] = pickup.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Assignment diagnostic: calculate Haversine distance, but do not use it for prediction.
    data["haversine_distance_km"] = haversine_km(
        data["pickup_lat"], data["pickup_lng"], data["drop_lat"], data["drop_lng"]
    )

    assert data[FEATURES + ["fare_amount"]].notna().all().all()
    return raw, data, raw_path, int(valid.sum())


def build_preprocessor():
    """Create leakage-safe preprocessing that is fitted inside each CV fold."""
    categorical = ["city", "payment_method"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return preprocessor, categorical, numeric


def run(output):
    """Train, evaluate, persist, and document the final model in one reproducible run."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw, data, raw_path, valid_clean_rows = load_training_data()

    # Split once. The test partition must remain untouched until the final selected model.
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data["fare_amount"],
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
    assert set(data.loc[X_train.index, "trip_id"]).isdisjoint(data.loc[X_test.index, "trip_id"])

    preprocessor, categorical, numeric = build_preprocessor()

    # Compare assignment models with identical folds and identical preprocessing.
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=140, max_depth=16, min_samples_split=4,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=140, learning_rate=0.05, max_depth=3, random_state=42,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    # Model selection occurs only on training folds.
    cv_rows = []
    for model_name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", clone(estimator)),
        ])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_rows.append({
            "Model": model_name,
            "CV_MAE_Mean": float(-scores["test_MAE"].mean()),
            "CV_MAE_Std": float(scores["test_MAE"].std()),
            "CV_RMSE_Mean": float(-scores["test_RMSE"].mean()),
            "CV_RMSE_Std": float(scores["test_RMSE"].std()),
            "CV_R2_Mean": float(scores["test_R2"].mean()),
            "CV_R2_Std": float(scores["test_R2"].std()),
        })

    cv_results = pd.DataFrame(cv_rows).sort_values("CV_RMSE_Mean").reset_index(drop=True)
    best_name = str(cv_results.iloc[0]["Model"])

    # Refit the winner on the full training partition, then evaluate once on holdout.
    final_model = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(estimators[best_name])),
    ])
    final_model.fit(X_train, y_train)
    prediction = final_model.predict(X_test)
    final_metrics = regression_metrics(y_test, prediction)

    # Persist and immediately reload the pipeline to prove serialization works.
    model_path = output / "uber_fare_model.pkl"
    joblib.dump(final_model, model_path)
    reloaded = joblib.load(model_path)
    np.testing.assert_allclose(prediction, reloaded.predict(X_test), rtol=1e-12, atol=1e-12)

    # Data-quality diagnostics requested by the project brief.
    hav_corr = float(data[["distance_km", "haversine_distance_km"]].corr().iloc[0, 1])
    hav_within = float((data["distance_km"].sub(data["haversine_distance_km"]).abs() <= 0.1).mean())
    fare_distance_corr = float(data[["fare_amount", "distance_km"]].corr().iloc[0, 1])

    # Export human-readable evidence alongside the model binary.
    cv_results.to_csv(output / "cross_validation_results.csv", index=False)
    cv_results.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame([{"Model": best_name, **final_metrics}]).to_csv(
        output / "final_test_metrics.csv", index=False
    )
    X_test.iloc[[0]].to_csv(output / "example_input.csv", index=False)
    (output / "example_prediction.json").write_text(
        json.dumps({
            "prediction": float(reloaded.predict(X_test.iloc[[0]])[0]),
            "actual_fare": float(y_test.iloc[0]),
        }, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "best_model": best_name,
        "artifact_status": "coordinate_safe_final_model",
        "selection_method": "5-fold cross-validation on training partition; lowest mean CV RMSE",
        "features": FEATURES,
        "excluded_features": [
            "pickup_lat", "pickup_lng", "drop_lat", "drop_lng",
            "passenger_count", "drop_time", "status",
            "trip_id", "driver_id", "rider_id",
        ],
        "exclusion_reason": (
            "Coordinates are range-valid but internally inconsistent with supplied distance_km; "
            "passenger_count is absent from the supplied dataset."
        ),
        "categorical_features": categorical,
        "numeric_features": numeric,
        "cross_validation": cv_results.to_dict("records"),
        "final_test_metrics": final_metrics,
        "raw_rows": int(len(raw)),
        "valid_clean_rows": int(valid_clean_rows),
        "completed_model_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "split": {"test_size": 0.20, "random_state": 42, "shuffle": True},
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "dataset_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "haversine_vs_supplied_distance_correlation": hav_corr,
        "haversine_within_0_1_km_share": hav_within,
        "fare_vs_supplied_distance_correlation": fare_distance_corr,
        "python": platform.python_version(),
        "versions": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
        "persistence_roundtrip": "PASS",
        "note": (
            "Final coordinate-safe model. Passenger count is absent and not fabricated. "
            "Haversine distance is retained as a validation diagnostic only."
        ),
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("PASS: training-only CV, final holdout, coordinate diagnostics, model reload")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New or empty folder for exported artifacts")
    run(parser.parse_args().output_dir)
