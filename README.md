# 🚕 Uber Fare Prediction Model

An end-to-end regression portfolio project built from a supplied **50,000-trip dataset**, covering data validation, feature engineering, leakage-aware preprocessing, training-only cross-validation, final holdout evaluation, model persistence, reproducibility, and Streamlit deployment.

🌐 **Live demo:** https://uber-fare-prediction-model.streamlit.app/

## Business objective

Estimate `fare_amount` from information available for a trip while demonstrating a reproducible machine-learning workflow suitable for technical review.

## Dataset and modeling scope

- Original dataset: approximately **50,000 trips**
- Clean records retained: **49,997**
- Completed trips used for modeling: **42,538**
- Train/test split: **80:20**, `random_state=42`
- Training rows: **34,030**
- Final holdout rows: **8,508**
- Trip IDs are checked for train/test separation
- Preprocessing is fitted inside scikit-learn pipelines
- Passenger count is not fabricated because it is absent from the supplied dataset

## Modeling methodology

`train.py` now uses a stricter evaluation design:

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

Before the training-only CV selection upgrade, the same fixed 80/20 split produced these model-comparison results:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| **Linear Regression** | **2.473802** | **3.092921** | **0.754264** |
| Gradient Boosting | 2.475648 | 3.097699 | 0.753504 |
| Random Forest | 2.511922 | 3.159322 | 0.743600 |

These values are retained as a **historical baseline**, not presented as results from the new CV-selection workflow. Run `train.py` to generate the current `cross_validation_results.csv` and `final_test_metrics.csv` for a fresh output directory.

> R² is not percentage accuracy. MAE/RMSE are regression error measures in fare units.

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

Use a new/empty output directory for each run so existing artifacts are not overwritten accidentally.

## Features

The model uses trip/location, distance, payment/city, and pickup-time-derived features. Post-trip or leakage-prone information such as actual duration, drop time, trip status, IDs, and fare-derived columns is excluded from model input.

The supplied `distance_km` is treated as planned route distance. Coordinate-derived straight-line distance is not used as a substitute for the supplied route-distance field.

## Repository structure

```text
Uber-Fare-Prediction-Model/
├── app/                 # Streamlit prediction/dashboard app
├── dataset/             # Source and prepared datasets
├── doc/                 # Validation and project documentation
├── images/charts/       # Analysis visuals
├── models/              # Packaged model and supporting artifacts
├── notebooks/           # EDA and modeling notebook
├── train.py             # Reproducible CV + final-test training workflow
├── requirements.txt
├── PROJECT_SUMMARY.txt
└── README.md
```

## Technical evidence

- Data integrity and provenance checks
- Feature engineering verification
- Train/test ID separation
- Pipeline-based preprocessing
- Independent model pipelines
- 5-fold training-only cross-validation
- MAE, RMSE and R² reporting
- Untouched final holdout evaluation
- Saved-model persistence verification
- Streamlit deployment
- Reproducible output artifacts

## Tech stack

`Python` `Pandas` `NumPy` `scikit-learn` `Matplotlib` `Jupyter Notebook` `Joblib` `Streamlit`

## Limitations

This is an educational dataset and should not be interpreted as a model of real Uber pricing. The project demonstrates regression methodology and engineering practice, not a production pricing system. A production system would require real operational data, temporal/geographic validation, monitoring, drift detection, security controls, automated CI/CD, and ongoing model governance.

## Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • Digital Marketing
