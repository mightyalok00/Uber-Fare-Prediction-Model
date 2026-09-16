# Documentation Guide

This folder intentionally contains only the documentation that is still useful for the final project.

## Source of Truth

| File | Purpose |
| --- | --- |
| `../README.md` | Main project overview, findings, limitations, environment strategy and usage |
| `../train.py` | Authoritative cleaning, feature engineering, model selection, evaluation and export code |
| `ASSIGNMENT_COVERAGE.md` | Requirement-by-requirement assignment mapping and final answers |
| `PASSENGER_COUNT_NOTE.txt` | Explains why passenger-count analysis is unavailable and must not be fabricated |

Older overlapping reports were removed because some contained historical metrics, local machine paths, or superseded feature descriptions.

## Current Modeling Methodology

1. Start from the instructor-provided 50,000-row educational CSV.
2. Validate fare, distance, timestamps, coordinates and duplicate records.
3. Use 42,538 valid Completed rides for modeling.
4. Split once into 34,030 training rows and 8,508 untouched holdout rows (`test_size=0.20`, `random_state=42`).
5. Compare Linear Regression, Random Forest and Gradient Boosting with 5-fold KFold cross-validation on the training partition only.
6. Select the lowest mean CV RMSE.
7. Fit the selected pipeline on the complete training partition.
8. Evaluate it once on the untouched holdout.
9. Save/reload the pipeline and verify prediction consistency.

## Runtime Strategy

- **Python 3.12:** authoritative model training and serialization environment.
- **Python 3.14.7:** compatibility environment for loading the committed model and starting the Streamlit app.
- GitHub Actions checks both environments separately.

This split is deliberate: Python 3.14.7 and the current native scientific stack installed successfully on GitHub Actions, but full model training crashed at native-library level with exit code 139. The project therefore keeps training on the stable Python 3.12 environment while still testing Python 3.14.7 deployment compatibility.

## Dataset Limitations

- `passenger_count` is not present and cannot be reliably reconstructed.
- Passenger-count distribution, fare-vs-passenger analysis and passenger-count significance are therefore N/A.
- Haversine distance is calculated for assignment/data-quality validation only.
- The supplied `distance_km` is retained as the model distance feature because the coordinates are internally inconsistent with it.
- Results apply to this educational/synthetic dataset and should not be presented as real-world Uber pricing claims.

## Metric Interpretation

MAE and RMSE are regression error measures in fare units. R² is a coefficient of determination and must not be described as percentage accuracy.
