# Uber Fare Prediction

Predict `fare_amount` from the supplied 50,000-trip dataset using a saved preprocessing and regression pipeline.

## Verified results

Latest training-ready data: 42,538 Completed trips. Reproducible random 80:20 split (`random_state=42`): 34,030 training / 8,508 test rows; trip IDs do not overlap. Preprocessing is fitted only on training rows, with an independent preprocessing instance for each model.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 2.473802 | 9.566163 | 3.092921 | 0.754264 |
| Gradient Boosting | 2.475648 | 9.595738 | 3.097699 | 0.753504 |
| Random Forest | 2.511922 | 9.981314 | 3.159322 | 0.743600 |

The original saved Linear Regression model loads and predicts successfully. Refit predictions match exactly. Save/reload predictions also match. MAE and RMSE are in fare units; MSE is squared fare units. R² is not percentage accuracy. These scores also drive model selection and are not an independent final benchmark.

## Windows setup and use

From this repository folder, with Python 3.14 installed:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Reproduce all training, metrics, dataset checks and model persistence checks:

```powershell
.\.venv\Scripts\python.exe train.py --output-dir work/run_001
```

Choose a new output folder for each run; existing nonempty folders are rejected. The original model is never overwritten. To open the notebook:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/Uber_Fare_Prediction.ipynb
```

Load the packaged model and example input directly:

```python
from pathlib import Path
import joblib
import pandas as pd
root = Path.cwd()  # run from the repository root
model = joblib.load(root / 'models/uber_fare_model.pkl')
trip = pd.read_csv(root / 'models/example_input.csv')
print(model.predict(trip))  # approximately 20.5074
```

The `.pkl` is a joblib artifact containing preprocessing and the estimator. Use `joblib.load` with the pinned package versions. Only load trusted serialized models.

## Structure

```text
Uber_Fare_Prediction/
  dataset/           Original, cleaned and Completed-trip training-ready CSVs
  notebooks/         Executable analysis and training notebook
  models/            Saved pipeline, metadata, metrics and prediction examples
  app/               Streamlit prediction app and business dashboard
  doc/               Reports, validation notes and original imported versions
  images/charts/     Existing analysis charts
  train.py           Reproducible training and verification entry point
  requirements.txt
  README.md
  .gitignore
  PROJECT_SUMMARY.txt
```

## Data and interpretation

- Three invalid zero-distance trips are excluded; 49,997 cleaned records remain. Training keeps only the 42,538 Completed trips. No target-percentile trimming is applied.
- The supplied `distance_km` does not align with coordinate-derived Haversine distance. The app requires planned route distance and displays straight-line distance only as a reference. This educational dataset cannot validate real-world Uber pricing.
- Passenger count is absent and is not fabricated. Actual duration, drop time, fare-derived columns, IDs and status are excluded from model inputs. Payment method is assumed known at booking.
- The cleaned export truncates drop timestamps to seconds; the raw CSV retains subsecond precision. Drop time is not a predictive feature.
- `uber_trips_dataset_50k_cleaned.csv` is the canonical dashboard dataset; `uber_trips_completed_training_ready.csv` is the training source. `uber_fare_cleaned.csv` is retained for provenance.
- Original reports are retained as supplied. See `doc/VALIDATION_REPORT.md` for the current verification and changes.
- Git is local only. Nothing has been published or pushed to GitHub.
