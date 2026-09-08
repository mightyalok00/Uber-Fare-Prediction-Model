# Local validation — 8 September 2026

## Inspection and provenance

The requested D:\Project\_4_Uber Fare Prediction Model path does not exist. The matching original project is D:\Project_4_Uber Fare Prediction Model and contains the original CSV and assignment brief. The latest complete project was found at C:\Users\Alok Agarwal\OneDrive\Desktop\Uber_Fare_Prediction; it contains the cleaned/training-ready datasets and Linear Regression model described in the referenced conversation. The older Downloads ZIP contains a different model and was not used.

The original D: CSV and imported raw CSV have matching SHA-256 hashes. The original D: files, Desktop project, downloaded archives and saved model were preserved. The organized repository is a new Uber_Fare_Prediction child folder in the matching D: project. Original versions of edited imported files are in doc/original_import.

## Executed checks

- Latest training-ready CSV: 42,538 rows, all Completed; no missing fields, duplicate records or duplicate trip IDs.
- Cross-checked training-ready rows and original columns against the raw CSV after the documented cleaning and Completed filter. The cleaned CSV truncates drop timestamps to whole seconds, so timestamp comparison uses that precision. The raw file remains unchanged.
- Verified all pickup calendar, weekend and rush-hour features against pickup timestamps.
- Random split: test_size=0.20, random_state=42; 34,030 training and 8,508 test rows, no shared row indices or trip IDs.
- Loaded the existing joblib .pkl and predicted all test rows successfully.
- Cloned and refitted the existing Linear Regression pipeline. Maximum prediction difference from the saved model: 0.0.
- Retrained Linear Regression, Random Forest and Gradient Boosting using the notebook's hyperparameters and separate cloned preprocessors. Full results are in models/model_comparison.csv.
- Saved and reloaded the selected pipeline and compared every test prediction with strict numerical tolerance: PASS.
- Example reloaded prediction: 20.5073937314 fare units; example input and target are included in models/.
- Streamlit AppTest: startup, predict button, changed city/payment inputs, and empty dashboard filters all pass.

## Corrections and maintainability

- Updated deprecated Streamlit width arguments.
- Added latitude/longitude bounds and numerical clipping to Haversine calculation.
- Removed automatic substitution of coordinate distance for supplied training distance. Planned trip distance is explicitly entered; coordinate distance is a reference only.
- Dashboard uses the latest cleaned CSV.
- Prediction input columns are ordered using model metadata.
- Added train.py to audit the data, fit all three models, compute all four metrics and verify serialization. Each run requires a new/empty destination and does not overwrite the packaged model.
- Notebook uses the shared audited training routine and supports execution from the root or notebooks directory; every new run has a distinct output folder.
- Pinned installed versions, including the previously omitted notebook dependencies.
- Backed up edited imported artifacts; refreshed model metrics and metadata without replacing the original .pkl.

## Interpretation limits

Linear Regression: MAE 2.4738024474, MSE 9.5661626072, RMSE 3.0929213710, R2 0.7542642126. R2 is not percentage accuracy. Model selection uses this same test split; a separate final benchmark or nested validation is needed for an unbiased selection-independent estimate. The dataset does not establish real-world pricing performance. Distance/coordinates are inconsistent, payment method is assumed known at booking, and passenger count is absent. No passenger counts were fabricated. Original business reports are historical imported artifacts; this validation report and current metadata take precedence for run details.

The installed Python 3.14 environment successfully imported and executed the required packages. Its package manager reports unrelated leftover distribution warnings; no global packages were changed. README documents an isolated environment setup using the verified versions.

Notebook execution: PASS, all 10 code cells completed in a fresh Jupyter kernel.
