# 🚕 Uber Fare Prediction Model

An end-to-end regression portfolio project built from an **instructor-provided 50,000-trip dataset**, covering data validation, feature engineering, leakage-aware preprocessing, training-only cross-validation, final holdout evaluation, model persistence, reproducibility, and Streamlit deployment.

🌐 **Live demo:** https://uber-fare-prediction-model.streamlit.app/

## Business objective

Estimate `fare_amount` from information available before or at trip start while demonstrating a reproducible machine-learning workflow suitable for technical review.

## Dataset and assignment compatibility

This repository is built around the **actual 50,000-row CSV supplied by the instructor**. That file is the authoritative dataset for this project.

The raw dataset contains trip identifiers, city, pickup/drop-off coordinates, supplied `distance_km`, `fare_amount`, trip status, payment method, pickup time, and drop time.

### Important dataset notes

- The source dataset does **not** contain `passenger_count`.
- Passenger counts are therefore **not fabricated, imputed, or randomly generated**.
- Passenger-count distribution, fare-vs-passenger-count analysis, and passenger-count business conclusions are marked **not applicable to the supplied data**.
- The source dataset already contains `distance_km`; it is treated as the primary planned trip-distance feature.
- Coordinate-derived straight-line/Haversine distance may be used for reference or validation, but it is not falsely presented as the original route-distance field.

See [`doc/ASSIGNMENT_COVERAGE.md`](doc/ASSIGNMENT_COVERAGE.md) for the requirement-by-requirement mapping.

## Dataset and modeling scope

- Original dataset: **50,000 trips**
- Clean records retained: **49,997**
- Completed trips used for modeling: **42,538**
- Train/test split: **80:20**, `random_state=42`
- Training rows: **34,030**
- Final holdout rows: **8,508**
- Trip IDs are checked for train/test separation
- Preprocessing is fitted inside scikit-learn pipelines
- Passenger count is intentionally excluded because it is absent from the instructor-provided data

## Key business findings

Using valid **Completed** rides from the supplied dataset:

- Fare and supplied trip distance have a strong positive relationship, with Pearson correlation of approximately **0.871**.
- **06:00** has the highest average fare among pickup hours in the current completed-ride data.
- **Monday** has the highest average fare among days of the week.
- **January** has the highest average fare among months.
- The packaged baseline comparison identifies **Linear Regression** as the best of the three compared models by RMSE.
- Baseline Linear Regression results: **MAE 2.474**, **RMSE 3.093**, **R² 0.754**.
- The saved model can generate a new-trip fare estimate in the Streamlit app.

These are findings from this educational dataset and are not claims about real-world Uber pricing.

## Business questions answered

The project answers every requested question that is supported by the supplied data:

- What factors most strongly influence predicted Uber fares?
- How does trip distance relate to fare amount?
- Which pickup hours have higher average fares?
- How do fares vary by day of week?
- How do fares vary by month?
- What relationships appear in the numeric correlation matrix?
- Which regression model performs best?
- How accurate is the final selected model?
- Can the saved model provide a reasonable fare estimate for a new trip?

### Passenger-count question

**Does passenger count significantly affect fare?**

This cannot be answered reliably because the instructor-provided CSV does not contain `passenger_count`. Any conclusion would require genuine passenger-count labels from another dataset. This project intentionally avoids unsupported conclusions.

## Modeling methodology

`train.py` uses a stricter evaluation design:

1. Create one 80/20 train/test split.
2. Keep the test partition untouched during model selection.
3. Compare **Linear Regression, Random Forest, and Gradient Boosting** using **5-fold KFold cross-validation** on the training partition only (`shuffle=True`, `random_state=42`).
4. Report CV mean ± standard deviation for **MAE, RMSE, and R²**.
5. Select the model with the lowest mean CV RMSE.
6. Fit that selected pipeline on the complete training partition.
7. Evaluate it once on the held-out test set.
8. Save/reload the pipeline and verify prediction consistency.

This avoids selecting a model by repeatedly inspecting the final test set.

## Previously verified baseline results

Before the training-only CV selection upgrade, the same fixed 80/20 split produced:

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Linear Regression** | **2.473802** | **9.566163** | **3.092921** | **0.754264** |
| Gradient Boosting | 2.475648 | 9.595738 | 3.097699 | 0.753504 |
| Random Forest | 2.511922 | 9.981314 | 3.159322 | 0.743600 |

These values are retained as a **historical baseline**. Run `train.py` to generate current cross-validation and final-test artifacts for a new run.

> R² is not percentage accuracy. MAE and RMSE are regression error measures in fare units.

## Features

The model uses:

- city
- payment method
- pickup/drop-off latitude and longitude
- supplied `distance_km`
- pickup year, month, day and hour
- day of week
- weekend indicator
- rush-hour indicator

Post-trip or leakage-prone information such as actual duration, drop time, trip status, IDs, and fare-derived fields is excluded from the prediction feature set.

## Streamlit application

Run locally with:

```powershell
python -m streamlit run app/app.py
```

The app contains four sections:

### 🚕 Predict Fare

- city and payment-method selection
- pickup and drop-off coordinates
- pickup date/time
- planned trip distance
- derived weekend and rush-hour indicators
- new-trip fare prediction

### 📊 Business Dashboard

- interactive filters for city, status, payment method, distance, fare, hour and date
- fare distribution
- trip-distance distribution
- fare vs distance
- average fare by hour
- average fare by day of week
- average fare by month
- average fare by city
- correlation matrix
- direct business-answer metrics for distance correlation, highest-fare hour, day and month
- filtered-data download

### 🤖 Model Performance

- model comparison table
- MAE, MSE, RMSE and R²
- feature coefficient / importance visualization
- Actual vs Predicted chart
- leakage-prevention note
- model feature list

### ℹ️ About

- instructor-provided 50K dataset scope
- passenger-count limitation
- distance-field interpretation
- educational-data limitation

### App validation safeguards

The app validates:

- required model, metadata and cleaned dataset files exist;
- required dataset columns exist;
- `model_metadata.json` contains a non-empty feature list;
- the prediction form contains every feature expected by the packaged model;
- prediction inputs are ordered according to model metadata;
- Haversine distance is shown only as reference and does not replace supplied `distance_km`.

## Reproduce training

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --output-dir work/run_001
```

Each run generates:

- `cross_validation_results.csv`
- `final_test_metrics.csv`
- `model_metadata.json`
- `uber_fare_model.pkl`
- `example_input.csv`
- `example_prediction.json`
- `test_predictions.csv`

Use a new or empty output directory for each run.

## Repository structure

```text
Uber-Fare-Prediction-Model/
├── .github/              # Repository automation/configuration
├── app/
│   └── app.py            # Streamlit prediction + analytics application
├── dataset/              # Raw, cleaned and training-ready datasets
├── doc/
│   ├── ASSIGNMENT_COVERAGE.md
│   ├── PROJECT_SUMMARY.txt
│   ├── VALIDATION_REPORT.md
│   └── ...               # Business/validation reports
├── images/
│   └── charts/           # Analysis and model visuals
├── models/               # Saved model, metadata and prediction evidence
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
├── train.py              # Reproducible CV + final-test training workflow
├── requirements.txt
├── .gitignore
└── README.md
```

## Technical evidence

- Data integrity and provenance checks
- Feature engineering verification
- Train/test ID separation
- Pipeline-based preprocessing
- Independent model pipelines
- 5-fold training-only cross-validation
- MAE, MSE, RMSE and R² reporting
- Untouched final holdout evaluation
- Saved-model persistence verification
- Feature-effect/model-importance visualization
- Actual vs Predicted visualization
- Streamlit deployment
- Reproducible output artifacts

## Tech stack

`Python` `Pandas` `NumPy` `scikit-learn` `Matplotlib` `Jupyter Notebook` `Joblib` `Streamlit`

## Limitations

- The supplied dataset does not include passenger count, so passenger-count analysis is outside the scope of this version.
- `distance_km` is supplied by the dataset and is not claimed to be reconstructed from coordinates.
- The dataset is educational and should not be interpreted as a production Uber pricing dataset.
- Real-world fares may also depend on ride category, surge pricing, traffic, tolls, weather, local demand and other operational variables not available here.
- A production system would require temporal/geographic validation, monitoring, drift detection, security controls, automated CI/CD and ongoing model governance.

## Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • Digital Marketing
