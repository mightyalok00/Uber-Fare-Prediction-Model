# Uber Fare Prediction — Current Technical & Business Report

## Executive summary

This project analyzes an instructor-provided 50,000-trip educational Uber-style dataset and builds a reproducible fare-prediction workflow. After validation, 49,997 clean rows remain; 42,538 valid Completed trips are used for modeling.

The current evaluation design separates model selection from final testing. An 80/20 split creates 34,030 training rows and 8,508 untouched holdout rows. Linear Regression, Random Forest and Gradient Boosting are compared using 5-fold KFold cross-validation on the training partition only. The model with the lowest mean CV RMSE is then fitted on all training data and evaluated once on the final holdout set.

## Key findings

- Supplied trip distance is the strongest observed fare driver in this dataset, with Pearson correlation approximately 0.871 on valid Completed rides.
- The historical packaged fixed-split baseline identifies Linear Regression as the strongest of the three compared models, with MAE approximately 2.474, RMSE approximately 3.093 and R² approximately 0.754.
- The dataset shows time-based fare patterns, including the highest average fare at 06:00, on Monday and in January for the currently analyzed Completed-trip data.
- The Streamlit application supports both interactive business analysis and fare estimation for a new trip.

Historical baseline values are retained for reproducibility. Fresh current CV and final-test values should be generated with `train.py` and used when reporting a new training run.

## Dataset integrity

- Original rows: 50,000
- Clean rows retained: 49,997
- Completed modeling rows: 42,538
- Training rows: 34,030
- Holdout test rows: 8,508
- Missing values in original source: 0
- Duplicate rows in original source: 0
- Duplicate trip IDs in original source: 0
- Invalid/non-positive distance rows removed: 3

The supplied `distance_km` does not correspond closely to Haversine distance calculated from the pickup/drop-off coordinates. Therefore the supplied `distance_km` is retained as the source trip-distance feature, while coordinate-derived Haversine distance is used only for audit/reference.

## Passenger-count limitation

The instructor-provided CSV does not contain `passenger_count`. Passenger count cannot be reliably reconstructed from fare, distance, coordinates, duration, city, payment method or time fields. Therefore:

- passenger-count distribution is not applicable;
- fare vs passenger-count analysis is not applicable;
- no passenger-count business conclusion is made;
- no synthetic or random passenger labels are created.

If passenger-count analysis is mandatory, a labeled dataset containing genuine passenger counts is required.

## Features used for fare prediction

The current model feature set includes:

- city
- payment method
- pickup latitude/longitude
- drop-off latitude/longitude
- supplied `distance_km`
- pickup year
- pickup month
- pickup day
- pickup hour
- day of week
- weekend indicator
- rush-hour indicator

Post-trip/leakage-prone fields such as actual trip duration, drop time, status, identifiers and fare-derived fields are excluded from the prediction feature set.

## Model-training methodology

1. Validate the training-ready dataset against the raw source.
2. Create one 80/20 train/test split with `random_state=42`.
3. Keep the holdout test partition untouched during model selection.
4. Build separate scikit-learn pipelines for Linear Regression, Random Forest and Gradient Boosting.
5. Perform 5-fold KFold cross-validation on the training partition only with `shuffle=True` and `random_state=42`.
6. Compare CV mean and standard deviation for MAE, RMSE and R².
7. Select the model with the lowest mean CV RMSE.
8. Fit that selected pipeline on the full training partition.
9. Evaluate it once on the holdout test set.
10. Save and reload the pipeline, then verify prediction consistency.

This design avoids repeatedly using the final test set for model selection.

## Historical packaged baseline

The retained fixed-split baseline results are:

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 2.473802 | 9.566163 | 3.092921 | 0.754264 |
| Gradient Boosting | 2.475648 | 9.595738 | 3.097699 | 0.753504 |
| Random Forest | 2.511922 | 9.981314 | 3.159322 | 0.743600 |

These values are historical baseline results, not training-only cross-validation scores.

## Business interpretation

For this educational dataset, distance is the dominant fare signal. Time features show smaller but visible differences across pickup hour, weekday and month. The Streamlit dashboard helps communicate those patterns, while the prediction form demonstrates how the fitted pipeline can estimate fare from pre-trip inputs.

R² should not be described as percentage accuracy. MAE and RMSE quantify prediction error in fare units, while R² measures explained variance relative to a baseline.

## Limitations

- Educational/synthetic dataset, not production Uber data.
- Passenger count is unavailable.
- Coordinates are not internally consistent with supplied route distance.
- Payment method is assumed known at prediction time.
- Real-world pricing may depend on surge, traffic, tolls, ride category, weather, demand and other unavailable variables.
- Production deployment would require temporal/geographic validation, monitoring, drift detection, governance, security and automated CI/CD.

## Reproducibility

Run current training with:

```powershell
python train.py --output-dir work/run_001
```

Run the application with:

```powershell
python -m streamlit run app/app.py
```

Use a new or empty output directory for every training run.

## Documentation precedence

For the current project state, use:

1. `train.py`
2. `README.md`
3. `doc/ASSIGNMENT_COVERAGE.md`
4. `doc/VALIDATION_REPORT.md`
5. this report

Older imported DOCX/PDF reports are preserved as historical artifacts and should not override the current methodology above.
