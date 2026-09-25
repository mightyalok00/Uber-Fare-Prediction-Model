# 🚕 Uber Fare Prediction Model

An end-to-end regression portfolio project built from the **instructor-provided 50,000-trip educational dataset**. It covers data validation, EDA, feature engineering, leakage-safe model selection, untouched holdout evaluation, model persistence, business interpretation, and Streamlit deployment.

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
</p>

🌐 **Live Demo:** [https://uber-fare-prediction-model.streamlit.app/](https://uber-fare-prediction-model.streamlit.app/)

## Business Objective

Estimate `fare_amount` before a trip is completed using historical ride information that is genuinely supported by the supplied dataset.

## Key Findings / Business Impact

- **Supplied trip distance is the strongest observed fare signal**; fare vs `distance_km` Pearson correlation is approximately **0.871** on valid completed rides.
- **Linear Regression** is selected by the lowest mean **5-fold CV RMSE** on the training partition.
- Final untouched-test performance: **MAE 2.474**, **MSE 9.567**, **RMSE 3.093**, **R² 0.754**.
- In the valid completed-ride data, **06:00** has the highest average fare by pickup hour.
- The Streamlit app provides aligned filters, business EDA, model evidence, data-quality diagnostics, and new-trip fare prediction.

> These findings are specific to the supplied educational dataset and are not claims about Uber pricing in the real world.

## ⭐ Project highlights

- End-to-end regression workflow with training-only cross-validation and an untouched final test set.
- Reproducible saved pipeline, data-quality checks, and documented limitations.
- Interactive Streamlit fare-prediction and business-analysis application.
- **Live deployed application:** [Open the Uber Fare Prediction Model](https://uber-fare-prediction-model.streamlit.app/)

**⭐ If this repository is useful to you, consider giving it a star.**

## Dataset vs Assignment

The assignment describes a Kaggle-style Uber Fares dataset with approximately 200,000 rows and a `passenger_count` field. The actual instructor-provided CSV used here has **50,000 rows and 14 columns** and does **not** contain `passenger_count`.

Therefore:
- passenger-count distribution = **N/A**
- fare vs passenger count = **N/A**
- passenger-count significance/business conclusion = **N/A**

The project does not fabricate, randomly generate, or infer passenger counts.

## Data Quality Summary

| Check | Result |
| --- | ---: |
| Raw rows | 50,000 |
| Raw columns | 14 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate trip IDs | 0 |
| Valid cleaned rows | 49,997 |
| Valid completed rides used for modeling | 42,538 |
| Haversine vs supplied distance correlation | ~0.00069 |
| Haversine within 0.1 km of supplied distance | ~1.50% |
| Fare vs supplied `distance_km` correlation | ~0.871 |

## Coordinate-Derived Trip Distance

The assignment asks for trip distance to be calculated from pickup/drop-off coordinates. This project **does calculate Haversine distance as a validation diagnostic**.

However, the coordinate-derived distance is internally inconsistent with the supplied `distance_km`. On the final completed modeling rows:
- correlation with supplied distance ≈ **0.00069**
- only about **1.50%** of rows are within 0.1 km

Because of that inconsistency, pickup/drop coordinates are **not prediction features**. The supplied `distance_km` remains the active route-distance field because it is the field actually associated with fare in this educational dataset.

## Active Prediction Features

- `city`
- `payment_method`
- `distance_km`
- `pickup_year`
- `pickup_month`
- `pickup_day`
- `pickup_hour`
- `day_of_week`
- `is_weekend`
- `is_rush_hour`

Excluded from active prediction:
- pickup/drop latitude and longitude
- `passenger_count` (not present)
- drop time / actual duration
- status
- trip, rider, and driver IDs
- target-derived fields

## Modeling Methodology

1. Validate and clean the instructor dataset.
2. Keep valid **Completed** rides.
3. Create one **80:20 train/test split** (`random_state=42`).
4. Keep the test partition untouched during model selection.
5. Compare **Linear Regression, Random Forest, and Gradient Boosting** using **5-fold KFold CV on the training partition only**.
6. Fit preprocessing inside each model pipeline/fold.
7. Select the model with the lowest mean CV RMSE.
8. Refit the selected pipeline on the full training partition.
9. Evaluate once on the untouched holdout.
10. Save/reload the final pipeline and verify prediction consistency.

## Final Model Comparison

| Model | CV MAE Mean | CV RMSE Mean | CV R² Mean |
| --- | ---: | ---: | ---: |
| **Linear Regression** | **2.4740** | **3.0868** | **0.7590** |
| Gradient Boosting | 2.4761 | 3.0911 | 0.7583 |
| Random Forest | 2.5272 | 3.1774 | 0.7447 |

### Final Untouched-Test Result

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Linear Regression** | **2.473577** | **9.566736** | **3.093014** | **0.754249** |

> R² is not percentage accuracy. MAE and RMSE are regression-error measures in fare units.

## Business Questions — Final Answers

**1. What factors most strongly influence Uber fares?**  
The EDA and final model show supplied `distance_km` as the strongest supported observed signal. Model coefficients are influence diagnostics, not causal proof.

**2. How does trip distance affect fare?**  
Fare has a strong positive relationship with supplied `distance_km` (Pearson correlation ≈ **0.871**).

**3. Does passenger count significantly affect fare?**  
Cannot be determined because `passenger_count` is absent from the supplied CSV.

**4. Which hours have higher average fares?**  
**06:00** has the highest average fare in the valid completed-ride data.

**5. Which model performs best?**  
**Linear Regression**, selected by the lowest mean training-only 5-fold CV RMSE.

**6. How accurate is the final model?**  
Untouched holdout: MAE **2.474**, RMSE **3.093**, R² **0.754**.

**7. Can the model provide a reasonable fare estimate for a new trip?**  
Yes. The Streamlit **Predict Fare** tab loads the final saved pipeline and estimates fare from supported pre-trip inputs.

## 🚀 Live Streamlit Application

**Try the deployed application:**  
👉 [Open Uber Fare Prediction Model](https://uber-fare-prediction-model.streamlit.app/)

Run locally:

```powershell
python -m streamlit run app/app.py
```

The app contains:
- **Predict Fare** — aligned prediction form and trip summary
- **Business Dashboard** — filters for city, status, payment, fare, distance, and pickup date
- **Model Performance** — final metrics, CV comparison, feature influence, Actual vs Predicted
- **Data Quality** — dataset and coordinate-distance diagnostics
- **About** — scope, model contract, and limitations

## Python Version Strategy

The project deliberately separates **training stability** from **deployment compatibility**:

- **Python 3.12** is the authoritative model-training and serialization environment.
- The committed model metadata records the exact training environment.
- **Python 3.14.7** is used in a separate GitHub Actions job to verify that the committed model loads, predicts, and that Streamlit starts successfully.
- Full model retraining is not forced under Python 3.14.7 because the current hosted scientific stack produced a native segmentation fault during training even though Python and dependencies installed successfully.

This avoids pretending a crashing training environment is production-ready while still testing forward compatibility.

## Install

Runtime / training dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional notebook tooling:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

## Reproduce Training

Use Python 3.12 for the authoritative training run:

```powershell
python train.py --output-dir work/run_001
```

A training run exports:
- `uber_fare_model.pkl`
- `model_metadata.json`
- `cross_validation_results.csv`
- `model_comparison.csv`
- `final_test_metrics.csv`
- `example_input.csv`
- `example_prediction.json`

## Repository Structure

```text
Uber-Fare-Prediction-Model/
├── .github/
│   └── workflows/
│       └── python-checks.yml      # Python 3.12 training + Python 3.14.7 compatibility CI
├── app/
│   └── app.py                     # Streamlit application
├── dataset/
│   ├── uber_trips_dataset_50k.csv
│   └── uber_trips_dataset_50k_cleaned.csv
├── doc/
│   ├── ASSIGNMENT_COVERAGE.md
│   ├── PASSENGER_COUNT_NOTE.txt
│   └── README.md
├── images/
│   └── charts/
├── models/
│   ├── uber_fare_model.pkl
│   ├── model_metadata.json
│   ├── cross_validation_results.csv
│   ├── model_comparison.csv
│   ├── final_test_metrics.csv
│   ├── example_input.csv
│   └── example_prediction.json
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
├── train.py
├── requirements.txt
├── requirements-notebook.txt
└── README.md
```

The former `uber_trips_completed_training_ready.csv` was removed because it duplicated data that `train.py` can reproduce from the authoritative raw source and was not required by the app.

## CI Checks

GitHub Actions now runs two independent jobs:

1. **Python 3.12 training reproducibility** — recompiles source, installs runtime dependencies, retrains into a temporary directory, reloads the model, and verifies a finite prediction.
2. **Python 3.14.7 compatibility** — compiles the code, loads the committed Python-3.12-trained model, verifies prediction compatibility, and runs a Streamlit smoke test.

CI does **not** modify or commit repository files.

## Limitations

- `passenger_count` is unavailable in the supplied dataset.
- Coordinates are range-valid but internally inconsistent with supplied `distance_km`.
- The dataset is educational/synthetic.
- `payment_method` is assumed to be known at prediction time for this educational workflow.
- Real-world fares may additionally depend on ride category, surge, traffic, tolls, weather, local demand, and other unavailable variables.

## Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • Digital Marketing
