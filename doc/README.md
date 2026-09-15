# Documentation Guide

This folder contains the current documentation for the Uber Fare Prediction Model.

## Source-of-truth documents

| File | Purpose |
| --- | --- |
| `../README.md` | Main project overview, findings, limitations and usage |
| `../train.py` | Authoritative training and evaluation implementation |
| `ASSIGNMENT_COVERAGE.md` | Requirement-by-requirement assignment mapping |
| `CURRENT_REPORT.md` | Current technical and business report |
| `VALIDATION_REPORT.md` | Validation, provenance and methodology checks |
| `PROJECT_SUMMARY.txt` | Concise project summary |
| `PASSENGER_COUNT_NOTE.txt` | Explains why passenger count is unavailable and cannot be reconstructed |
| `uber_dataset_quality_report.txt` | Dataset quality and distance/coordinate integrity findings |

## Current modeling methodology

1. Start from the instructor-provided 50,000-row educational CSV.
2. Validate and clean the source data.
3. Use 42,538 valid Completed rides for modeling.
4. Split once into 34,030 training rows and 8,508 holdout test rows (`test_size=0.20`, `random_state=42`).
5. Compare Linear Regression, Random Forest and Gradient Boosting using 5-fold KFold cross-validation on the training partition only (`shuffle=True`, `random_state=42`).
6. Select the model with the lowest mean CV RMSE.
7. Fit the selected pipeline on the complete training partition.
8. Evaluate it once on the untouched holdout test set.
9. Save and reload the pipeline and verify prediction consistency.

## Dataset limitations

- `passenger_count` is not present in the supplied dataset and cannot be reliably reconstructed from the available fields.
- Passenger-count distribution, fare-vs-passenger-count analysis and passenger-count business conclusions are therefore not applicable to this dataset.
- The supplied `distance_km` is retained as the source distance feature because it is strongly related to fare.
- Haversine distance derived from coordinates is used only for audit/reference because the coordinates are not internally consistent with the supplied route-distance field.
- Results apply to this educational/synthetic dataset and should not be presented as real-world Uber pricing claims.

## Metric interpretation

MAE and RMSE are regression error measures in fare units. R² is a coefficient of determination and must not be described as percentage accuracy.
