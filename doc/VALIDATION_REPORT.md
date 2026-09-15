# Local validation — 8 September 2026

## Inspection and provenance

The requested `D:\Project\_4_Uber Fare Prediction Model` path does not exist. The matching original project is `D:\Project_4_Uber Fare Prediction Model` and contains the original CSV and assignment brief. The latest complete project was found at `C:\Users\Alok Agarwal\OneDrive\Desktop\Uber_Fare_Prediction`; it contains the cleaned/training-ready datasets and Linear Regression model described in the referenced conversation. The older Downloads ZIP contains a different model and was not used.

The original D: CSV and imported raw CSV have matching SHA-256 hashes. The original D: files, Desktop project, downloaded archives and saved model were preserved. The organized repository is a new `Uber_Fare_Prediction` child folder in the matching D: project. Original versions of edited imported files are in `doc/original_import` where applicable.

## Executed checks

- Latest training-ready CSV: **42,538 rows**, all Completed; no missing fields, duplicate records or duplicate trip IDs.
- Cross-checked training-ready rows and original columns against the raw CSV after the documented cleaning and Completed filter. The cleaned CSV truncates drop timestamps to whole seconds, so timestamp comparison uses that precision. The raw file remains unchanged.
- Verified all pickup calendar, weekend and rush-hour features against pickup timestamps.
- Random split: `test_size=0.20`, `random_state=42`; **34,030 training** and **8,508 test** rows, with no shared row indices or trip IDs.
- Loaded the existing joblib `.pkl` and predicted all test rows successfully.
- Cloned and refitted the existing Linear Regression pipeline. Maximum prediction difference from the saved model: `0.0`.
- Retrained Linear Regression, Random Forest and Gradient Boosting using separate cloned preprocessors.
- Saved and reloaded the selected pipeline and compared predictions with strict numerical tolerance: **PASS**.
- Example reloaded prediction: `20.5073937314` fare units; example input and target are included in `models/`.
- Streamlit AppTest: startup, predict button, changed city/payment inputs, and empty dashboard filters all pass.

## Current model-selection methodology

The current repository uses a leakage-aware evaluation design implemented in `train.py`:

1. Create one 80/20 train/test split.
2. Keep the test partition untouched during model selection.
3. Compare **Linear Regression, Random Forest and Gradient Boosting** with **5-fold KFold cross-validation on the training partition only** (`shuffle=True`, `random_state=42`).
4. Report CV mean ± standard deviation for **MAE, RMSE and R²**.
5. Select the model with the lowest mean CV RMSE.
6. Fit the selected pipeline on the complete training partition.
7. Evaluate it once on the untouched final test set.
8. Save/reload the selected pipeline and verify prediction consistency.

This replaces the earlier workflow in which the same test split was used for model comparison. Historical baseline metrics are retained only for reproducibility and comparison; they are not presented as training-only CV results.

## Historical packaged baseline

The previously verified fixed-split baseline produced:

- **Linear Regression** — MAE `2.4738024474`, MSE `9.5661626072`, RMSE `3.0929213710`, R² `0.7542642126`
- **Gradient Boosting** — MAE `2.4756482748`, MSE `9.5957380045`, RMSE `3.0976988240`, R² `0.7535044791`
- **Random Forest** — MAE `2.5119221563`, MSE `9.9813137222`, RMSE `3.1593217187`, R² `0.7435998020`

These values are **historical baseline results**. Run `train.py` into a new/empty output directory to generate the current cross-validation and final-test artifacts for a fresh training run.

> R² is not percentage accuracy. MAE and RMSE are regression error measures in fare units.

## Corrections and maintainability

- Updated deprecated Streamlit width arguments.
- Added latitude/longitude bounds and numerical clipping to Haversine calculation.
- Removed automatic substitution of coordinate distance for supplied training distance. Planned trip distance is explicitly entered; coordinate distance is reference only.
- Dashboard uses the latest cleaned CSV.
- Prediction input columns are ordered using model metadata.
- Added `train.py` to audit the data, perform training-only cross-validation, evaluate the selected model once on the holdout test set, compute regression metrics and verify serialization.
- Notebook uses the shared audited training routine and supports execution from the root or notebooks directory; every new run has a distinct output folder.
- Pinned installed versions, including the previously omitted notebook dependencies.
- Backed up edited imported artifacts; retained historical packaged-model evidence without presenting it as current CV evidence.

## Interpretation limits

- The supplied dataset is educational/synthetic and does not establish real-world Uber pricing performance.
- The supplied `distance_km` field is strongly related to fare but is not consistent with straight-line Haversine distance calculated from the coordinates. It is therefore kept as the source trip-distance feature, while Haversine distance is used only for reference/audit.
- `passenger_count` is absent from the supplied CSV and cannot be reliably reconstructed from the available columns. No passenger counts are fabricated.
- Payment method is assumed to be known at prediction time for this educational workflow.
- Real-world fares may depend on ride category, surge pricing, traffic, tolls, weather, demand and other variables not present here.

## Current source of truth

For the latest methodology and project scope, use these files in this order:

1. `train.py`
2. `README.md`
3. `doc/ASSIGNMENT_COVERAGE.md`
4. this `doc/VALIDATION_REPORT.md`
5. current generated training artifacts from a fresh `train.py` run

Older imported business-report DOCX/PDF files should be treated as historical artifacts when their wording conflicts with the current files above.

Notebook execution: **PASS** — all 10 code cells completed in a fresh Jupyter kernel.
