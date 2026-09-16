# 🚕 Uber Fare Prediction Model

An end-to-end regression portfolio project built from the **instructor-provided 50,000-trip educational dataset**. The project covers data validation, EDA, feature engineering, leakage-safe model selection, final holdout evaluation, model persistence, business interpretation, and Streamlit deployment.

🌐 **Live demo:** https://uber-fare-prediction-model.streamlit.app/

## Business Objective

Estimate `fare_amount` before the trip is completed using historical ride information that is genuinely available in the supplied 50K dataset.

## Key Findings / Business Impact

- **Supplied trip distance is the strongest observed fare driver**; fare vs `distance_km` Pearson correlation is approximately **0.871** on valid completed rides.
- **Linear Regression** is selected by the lowest mean **5-fold CV RMSE** on the training partition.
- Final untouched-test performance: **MAE 2.474**, **MSE 9.567**, **RMSE 3.093**, **R² 0.754**.
- In the full valid completed-ride data, **06:00** has the highest average fare by pickup hour and **Monday** has the highest average fare by day.
- The Streamlit app provides aligned filters, business EDA, model evidence, data-quality diagnostics, and a new-trip fare estimate.

> These findings are specific to the supplied educational dataset and are not claims about Uber pricing in the real world.

## Dataset-vs-Assignment Note

The assignment document references a Kaggle-style Uber Fares dataset with roughly 200,000 rows and a `passenger_count` column. The actual instructor-provided CSV used here has **50,000 rows and 14 columns** and does **not** contain `passenger_count`.

Therefore:
- passenger-count distribution = **N/A**
- fare vs passenger count = **N/A**
- passenger-count significance/business conclusion = **N/A**

The project does not fabricate or infer passenger counts.

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
| Haversine vs supplied distance correlation | 0.0016 |
| Haversine within 0.1 km of supplied distance | 1.50% |
| Fare vs supplied `distance_km` correlation | 0.871 |

## Coordinate-Derived Trip Distance

The assignment specifically asks for `trip_distance` to be calculated from pickup/drop-off coordinates. This project **does calculate Haversine distance as a validation diagnostic**.

However, the coordinate-derived straight-line distance is internally inconsistent with the supplied `distance_km`:
- correlation with supplied distance ≈ **0.0016**
- only **1.50%** of rows are within 0.1 km

Because of that inconsistency, pickup/drop coordinates are **not used as prediction features**. The supplied `distance_km` remains the active route-distance feature because it is the distance field actually associated with fare.

## Feature Engineering

Created from `pickup_time`:
- `pickup_year`
- `pickup_month`
- `pickup_day`
- `pickup_hour`
- `day_of_week`
- `is_weekend`
- `is_rush_hour`

Active prediction features:
- `city`
- `payment_method`
- `distance_km`
- pickup year/month/day/hour
- day of week
- weekend indicator
- rush-hour indicator

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
4. Keep the test set untouched during model selection.
5. Compare **Linear Regression, Random Forest, and Gradient Boosting** using **5-fold KFold CV on the training partition only**.
6. Fit preprocessing inside each pipeline/fold.
7. Select the lowest mean CV RMSE.
8. Refit the selected pipeline on the full training partition.
9. Evaluate once on the untouched holdout.
10. Save/reload the final pipeline and verify identical predictions.

## Final Model Comparison

| Model             |   CV_MAE_Mean |   CV_MAE_Std |   CV_RMSE_Mean |   CV_RMSE_Std |   CV_R2_Mean |   CV_R2_Std |
|:------------------|--------------:|-------------:|---------------:|--------------:|-------------:|------------:|
| Linear Regression |       2.474   |     0.011815 |        3.08676 |      0.009508 |     0.759023 |    0.003091 |
| Gradient Boosting |       2.47614 |     0.009475 |        3.09112 |      0.007752 |     0.758341 |    0.00302  |
| Random Forest     |       2.52722 |     0.012693 |        3.17742 |      0.01373  |     0.744655 |    0.003768 |

### Final Untouched-Test Result

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| **Linear Regression** | **2.473577** | **9.566736** | **3.093014** | **0.754249** |

> R² is not percentage accuracy. MAE and RMSE are regression-error measures in fare units.

## Final Business Questions — Answers

**1. What factors most strongly influence Uber fares?**  
The model influence view and EDA show `distance_km` as the dominant supported numeric signal. Time and categorical effects are also represented, but coefficient magnitude is not interpreted as causation.

**2. How does trip distance affect fare?**  
Fare rises strongly with the supplied trip-distance field; Pearson correlation is approximately **0.871**.

**3. Does passenger count significantly affect fare?**  
This cannot be determined from the instructor-provided CSV because `passenger_count` is absent. No values are fabricated.

**4. Which hours have higher average fares?**  
In the full valid completed-ride dataset, **06:00** has the highest average fare (with nearby hours very close).

**5. Which model performs best?**  
**Linear Regression**, selected using the lowest mean training-only 5-fold CV RMSE.

**6. How accurate is the final model?**  
On the untouched holdout: MAE **2.474**, RMSE **3.093**, R² **0.754**.

**7. Can the model provide a reasonable fare estimate for a new trip?**  
Yes. The Streamlit **Predict Fare** tab loads the final saved pipeline and estimates fare from supported pre-trip inputs.

## Streamlit Application

Run locally:

```powershell
python -m streamlit run app/app.py
```

The app contains:
- **Predict Fare** — aligned two-column input form and summary metrics
- **Business Dashboard** — sidebar filters for city, status, payment, fare, distance, and pickup-date range
- **Model Performance** — final metrics, CV comparison, feature influence, Actual vs Predicted
- **Data Quality** — assignment diagnostics including Haversine-vs-supplied-distance validation
- **About** — scope and limitations

## Reproduce Training

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --output-dir work/run_001
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
├── .github/                      # CI checks
├── app/
│   └── app.py                    # Streamlit application
├── dataset/                      # Raw, cleaned, and training-ready data
├── doc/                          # Assignment coverage and validation notes
├── images/                       # Project visuals
├── models/
│   ├── uber_fare_model.pkl       # Final coordinate-safe pipeline
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
└── README.md
```

## Limitations

- `passenger_count` is unavailable in the supplied 50K dataset.
- Coordinates are range-valid but inconsistent with `distance_km`; they are diagnostic-only.
- The dataset is educational/synthetic.
- Real-world fares may additionally depend on ride type, surge, traffic, tolls, weather, and local demand.

## Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • Digital Marketing
