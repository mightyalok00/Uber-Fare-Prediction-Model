# 🚕 Uber Fare Prediction Model

An end-to-end regression portfolio project built from an **instructor-provided 50,000-trip educational dataset**, covering validation, feature engineering, leakage-aware preprocessing, training-only cross-validation, final holdout evaluation, reproducibility, and Streamlit deployment.

🌐 **Live demo:** https://uber-fare-prediction-model.streamlit.app/

## Business objective

Estimate `fare_amount` from information available before or at trip start while demonstrating a reproducible machine-learning workflow suitable for technical review.

## Key findings

- **Trip distance is the strongest observed fare driver**, with Pearson correlation of approximately **0.871** on valid completed rides.
- The historical packaged baseline identified **Linear Regression** as the strongest of the three compared models by RMSE: **MAE 2.474**, **RMSE 3.093**, **R² 0.754**.
- The dataset shows time-based fare patterns, including the highest average fare at **06:00**, on **Monday**, and in **January**.
- The Streamlit app provides interactive exploration and new-trip fare estimation.

> Findings are specific to the supplied educational dataset and are not claims about real-world Uber pricing.

## Dataset integrity decisions

The raw dataset contains trip identifiers, city, pickup/drop-off coordinates, supplied `distance_km`, `fare_amount`, status, payment method, pickup time, and drop time.

Two important limitations are handled explicitly:

1. **Passenger count is absent.** The project does not fabricate, impute, or randomly generate `passenger_count`. Passenger-count distribution, fare-vs-passenger-count analysis, and passenger-count business conclusions are therefore not applicable to this supplied dataset.
2. **Coordinates and supplied distance are internally inconsistent.** The stored `distance_km` and coordinate-derived straight-line distance have almost no relationship in this synthetic dataset. Coordinates are therefore **excluded from active fare prediction**. The supplied `distance_km` remains the distance feature because it is the field actually associated with fare.

See [`doc/ASSIGNMENT_COVERAGE.md`](doc/ASSIGNMENT_COVERAGE.md) and [`doc/uber_dataset_quality_report.txt`](doc/uber_dataset_quality_report.txt).

## Dataset and modeling scope

- Original dataset: **50,000 trips**
- Valid cleaned rows: **49,997**
- Completed trips used for modeling: **42,538**
- Train/test split: **80:20**, `random_state=42`
- Training rows: **34,030**
- Holdout rows: **8,508**
- Trip IDs checked for train/test separation
- Preprocessing fitted inside scikit-learn pipelines

## Coordinate-safe prediction features

Current `train.py` uses:

- city
- payment method
- supplied `distance_km`
- pickup year
- pickup month
- pickup day
- pickup hour
- day of week
- weekend indicator
- rush-hour indicator

The following are intentionally excluded from prediction:

- pickup latitude / longitude
- drop-off latitude / longitude
- passenger count (not present)
- actual duration
- drop time
- trip status
- trip IDs
- fare-derived fields

## Modeling methodology

`train.py` follows this evaluation design:

1. Create one 80/20 train/test split.
2. Keep the test partition untouched during model selection.
3. Compare **Linear Regression, Random Forest, and Gradient Boosting** using **5-fold KFold cross-validation** on the training partition only (`shuffle=True`, `random_state=42`).
4. Report CV mean ± standard deviation for **MAE, RMSE, and R²**.
5. Select the model with the lowest mean CV RMSE.
6. Fit that selected pipeline on the complete training partition.
7. Evaluate it once on the held-out test set.
8. Save/reload the pipeline and verify prediction consistency.

## Historical packaged baseline

The repository retains an older packaged `.pkl` and comparison metrics for reproducibility. That historical artifact included coordinate columns before the coordinate-quality issue was fully addressed.

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Linear Regression** | **2.473802** | **9.566163** | **3.092921** | **0.754264** |
| Gradient Boosting | 2.475648 | 9.595738 | 3.097699 | 0.753504 |
| Random Forest | 2.511922 | 9.981314 | 3.159322 | 0.743600 |

These values are historical evidence, not fresh coordinate-safe CV results.

> R² is not percentage accuracy. MAE and RMSE are regression error measures in fare units.

## Streamlit application

Run locally:

```powershell
python -m streamlit run app/app.py
```

The app has four sections:

- **Predict Fare** — city, payment method, pickup date/time, planned trip distance, and derived time indicators. Coordinates are not requested.
- **Business Dashboard** — filters, fare distribution, fare-vs-distance, time-based analysis, and correlations.
- **Model Performance** — active coordinate-safe model information, holdout metrics, and historical comparison evidence.
- **About** — dataset scope and limitations.

### Live-app coordinate fix

If the repository still contains the older coordinate-dependent packaged model, the Streamlit app **does not use it for live prediction**. It detects the legacy feature list and builds a coordinate-safe Linear Regression from the cleaned completed-trip data using the corrected feature set. Once a newly generated coordinate-safe model artifact is placed in `models/`, the app can use that packaged model directly.

## Reproduce coordinate-safe training

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --output-dir work/run_001
```

A run generates:

- `cross_validation_results.csv`
- `final_test_metrics.csv`
- `model_metadata.json`
- `uber_fare_model.pkl`
- `example_input.csv`
- `example_prediction.json`
- `test_predictions.csv`

## Business questions answered

The repository answers every requested question supported by the supplied data, including fare distribution, distance distribution, fare vs distance, fare by hour/day/month, correlations, model comparison, regression metrics, feature influence, and new-trip prediction.

The three passenger-count questions remain correctly marked **not applicable** because the source CSV does not contain genuine passenger-count labels.

## Repository structure

```text
Uber-Fare-Prediction-Model/
├── .github/              # CI checks
├── app/
│   └── app.py            # Coordinate-safe Streamlit app
├── dataset/              # Raw, cleaned and training-ready data
├── doc/                  # Assignment, validation and quality documentation
├── images/               # Project visuals
├── models/               # Packaged/historical model evidence
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
├── train.py              # Coordinate-safe CV + holdout workflow
├── requirements.txt
└── README.md
```

## Limitations

- `passenger_count` is unavailable.
- Coordinate columns are inconsistent with `distance_km` and are excluded from prediction.
- The dataset is educational/synthetic.
- Real-world fares may depend on ride category, surge, traffic, tolls, weather, local demand, and other unavailable variables.
- A production system would require temporal/geographic validation, monitoring, drift detection, security controls, CI/CD, and ongoing model governance.

## Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • Digital Marketing
