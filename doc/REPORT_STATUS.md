# Report Status

The repository contains current text documentation plus older imported binary reports.

## Current and synchronized

- `README.md`
- `train.py`
- `doc/ASSIGNMENT_COVERAGE.md`
- `doc/VALIDATION_REPORT.md`
- `doc/PROJECT_SUMMARY.txt`
- `doc/PASSENGER_COUNT_NOTE.txt`
- `doc/uber_dataset_quality_report.txt`
- `doc/CURRENT_REPORT.md`
- `doc/README.md`

## Historical binary artifacts

- `doc/Uber Prediction.docx`
- `doc/Uber_Fare_Prediction_Business_Analysis_Report.docx`
- `doc/Uber_Fare_Prediction_Business_Analysis_Report.pdf`

These binary files are retained for history and may reflect an earlier evaluation workflow. They are not the authoritative current report. If a statement in them conflicts with the current files above, use `train.py`, the repository `README.md`, `ASSIGNMENT_COVERAGE.md`, `VALIDATION_REPORT.md`, and `CURRENT_REPORT.md` as the source of truth.

## Current evaluation statement

The current workflow performs 5-fold cross-validation on the training partition only, selects the model using mean CV RMSE, and evaluates the selected model once on the untouched holdout test set.

Passenger count is not available in the instructor-provided dataset and is not reconstructed or fabricated.
