"""Reproduce training with coordinate-safe features, CV model selection, and an untouched final test set."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent

# Coordinates are intentionally excluded from prediction because the supplied
# pickup/drop-off coordinates are not internally consistent with distance_km.
FEATURES = [
    'city', 'payment_method', 'distance_km', 'pickup_year', 'pickup_month',
    'pickup_day', 'pickup_hour', 'day_of_week', 'is_weekend', 'is_rush_hour'
]


def score(y, prediction):
    assert np.isfinite(prediction).all(), 'Non-finite predictions'
    mse = mean_squared_error(y, prediction)
    return dict(
        MAE=mean_absolute_error(y, prediction),
        MSE=mse,
        RMSE=float(np.sqrt(mse)),
        R2=r2_score(y, prediction),
    )


def load_training_data():
    path = BASE / 'dataset/uber_trips_completed_training_ready.csv'
    data = pd.read_csv(path)
    raw = pd.read_csv(BASE / 'dataset/uber_trips_dataset_50k.csv')

    pickup = pd.to_datetime(raw.pickup_time, errors='coerce')
    drop = pd.to_datetime(raw.drop_time, errors='coerce')
    valid = (
        (raw.fare_amount > 0)
        & (raw.distance_km > 0)
        & (drop > pickup)
        & raw.pickup_lat.between(-90, 90)
        & raw.drop_lat.between(-90, 90)
        & raw.pickup_lng.between(-180, 180)
        & raw.drop_lng.between(-180, 180)
    )
    expected = raw.loc[valid & raw.status.eq('Completed')].drop_duplicates()

    assert data.trip_id.is_unique, 'Duplicate trip IDs'
    assert data.status.eq('Completed').all()
    assert data[FEATURES + ['fare_amount']].notna().all().all()

    numeric = [c for c in FEATURES if c not in ['city', 'payment_method']] + ['fare_amount']
    assert np.isfinite(data[numeric]).all().all()

    actual_source = data[raw.columns].reset_index(drop=True).copy()
    expected = expected.reset_index(drop=True).copy()
    for column in ['pickup_time', 'drop_time']:
        actual_source[column] = pd.to_datetime(actual_source[column]).dt.floor('s').astype('datetime64[ns]')
        expected[column] = pd.to_datetime(expected[column]).dt.floor('s').astype('datetime64[ns]')
    pd.testing.assert_frame_equal(actual_source, expected)

    time = pd.to_datetime(data.pickup_time)
    derived = dict(
        pickup_year=time.dt.year,
        pickup_month=time.dt.month,
        pickup_day=time.dt.day,
        pickup_hour=time.dt.hour,
        day_of_week=time.dt.dayofweek,
        is_weekend=time.dt.dayofweek.isin([5, 6]).astype(int),
        is_rush_hour=time.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int),
    )
    for name, values in derived.items():
        assert np.array_equal(data[name], values), f'Incorrect feature: {name}'

    return data, path


def build_preprocessor():
    cat = ['city', 'payment_method']
    num = [c for c in FEATURES if c not in cat]
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
        ('num', StandardScaler(), num),
    ]), cat, num


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {output}')

    data, data_path = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.fare_amount, test_size=.2, random_state=42, shuffle=True
    )

    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(data.loc[X_train.index, 'trip_id']).isdisjoint(data.loc[X_test.index, 'trip_id'])
    assert len(X_train) + len(X_test) == len(data)

    pre, cat, num = build_preprocessor()
    estimators = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=140, max_depth=16, min_samples_split=4,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=140, learning_rate=.05, max_depth=3, random_state=42
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'R2': 'r2',
    }

    cv_rows = []
    for name, estimator in estimators.items():
        pipe = Pipeline([('preprocess', clone(pre)), ('model', clone(estimator))])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows.append({
            'Model': name,
            'CV_MAE_Mean': float(-scores['test_MAE'].mean()),
            'CV_MAE_Std': float(scores['test_MAE'].std()),
            'CV_RMSE_Mean': float(-scores['test_RMSE'].mean()),
            'CV_RMSE_Std': float(scores['test_RMSE'].std()),
            'CV_R2_Mean': float(scores['test_R2'].mean()),
            'CV_R2_Std': float(scores['test_R2'].std()),
        })

    cv_results = pd.DataFrame(cv_rows).sort_values('CV_RMSE_Mean').reset_index(drop=True)
    best_name = cv_results.iloc[0]['Model']
    best = Pipeline([('preprocess', clone(pre)), ('model', clone(estimators[best_name]))])
    best.fit(X_train, y_train)

    prediction = best.predict(X_test)
    final_metrics = score(y_test, prediction)

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, output / 'uber_fare_model.pkl')
    reloaded = joblib.load(output / 'uber_fare_model.pkl')
    np.testing.assert_allclose(prediction, reloaded.predict(X_test), rtol=1e-12, atol=1e-12)

    cv_results.to_csv(output / 'cross_validation_results.csv', index=False)
    pd.DataFrame([{'Model': best_name, **final_metrics}]).to_csv(
        output / 'final_test_metrics.csv', index=False
    )

    metadata = dict(
        best_model=best_name,
        artifact_status='coordinate_safe_training_run',
        selection_method='5-fold cross-validation on training partition; lowest mean CV RMSE',
        features=FEATURES,
        excluded_features=['pickup_lat', 'pickup_lng', 'drop_lat', 'drop_lng'],
        exclusion_reason=(
            'Coordinates are excluded because they are not internally consistent with the '
            'supplied distance_km field in this educational dataset.'
        ),
        categorical_features=cat,
        numeric_features=num,
        cross_validation=cv_results.to_dict('records'),
        final_test_metrics=final_metrics,
        training_rows=len(X_train),
        testing_rows=len(X_test),
        completed_model_rows=len(data),
        split=dict(test_size=.2, random_state=42, shuffle=True),
        cv=dict(n_splits=5, shuffle=True, random_state=42),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        python=platform.python_version(),
        versions={
            p: importlib.metadata.version(p)
            for p in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'scipy', 'streamlit']
        },
        persistence_roundtrip='PASS',
        note=(
            'Coordinate-safe model: passenger_count is absent and not fabricated; '
            'coordinates are excluded; the held-out test set is evaluated once after CV selection.'
        ),
    )
    (output / 'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    X_test.iloc[[0]].to_csv(output / 'example_input.csv', index=False)
    (output / 'example_prediction.json').write_text(json.dumps({
        'prediction': float(reloaded.predict(X_test.iloc[[0]])[0]),
        'actual_fare': float(y_test.iloc[0]),
    }, indent=2), encoding='utf-8')
    pd.DataFrame({
        'trip_id': data.loc[X_test.index, 'trip_id'],
        'actual': y_test,
        'prediction': prediction,
    }).to_csv(output / 'test_predictions.csv', index=False)

    print('PASS: coordinate-safe features, training-only CV, final holdout evaluation, model reload')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='New or empty folder for this run')
    run(parser.parse_args().output_dir)
