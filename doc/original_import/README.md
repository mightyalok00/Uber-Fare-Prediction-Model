# Uber Fare Prediction Model

Portfolio machine-learning project built from the supplied **50,000-trip synthetic Uber-style dataset**.

## Objective
Predict an expected ride fare before the trip is completed using trip distance, pickup/drop-off coordinates, city, payment method, and pickup-time features.

## Dataset
- Rows: 50,000
- Clean rows: 49,997
- Completed trips used for ML: 42,538
- Target: `fare_amount`
- Important source limitation: `passenger_count` is not present and was not fabricated.

## Best model
**Linear Regression**

| Metric | Score |
|---|---:|
| MAE | 2.474 |
| RMSE | 3.093 |
| R² | 0.7543 |

## Project structure
```text
Uber_Fare_Prediction/
├── dataset/
│   ├── uber_trips_dataset_50k.csv
│   └── uber_fare_cleaned.csv
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
├── models/
│   ├── uber_fare_model.pkl
│   ├── model_metadata.json
│   └── model_comparison.csv
├── app/
│   └── app.py
├── doc/
│   └── Uber_Fare_Prediction_Business_Analysis_Report.docx
├── images/charts/
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Streamlit features
- Fare prediction form
- City and payment inputs
- Pickup/drop-off coordinates
- Date/time-derived model features
- Adjustable trip distance
- Business dashboard filters: city, status, payment method, distance, fare, pickup hour, date range
- KPI cards and charts
- Model comparison and performance tab
- Filtered CSV download

## Modeling choices
Only **Completed** trips are used for fare model training. Identifier columns (`trip_id`, `driver_id`, `rider_id`) and post-trip fields (`status`, actual drop time, actual duration) are excluded from prediction to reduce leakage and memorization.

## Limitation
The supplied dataset appears synthetic. City labels and coordinates are not always geographically consistent, so geographic conclusions should be interpreted cautiously.
