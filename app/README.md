# `app/` — Streamlit Application

This folder contains the interactive portfolio application.

## Files

- `app.py` — professional Streamlit dashboard, business filters, fare simulator, model-performance views, and data-quality diagnostics.

## Code organization

`app.py` is intentionally divided into commented sections:

1. configuration and constants;
2. visual/CSS system;
3. cached data/model loaders;
4. saved-model validation;
5. shared plotting/helper functions;
6. page header and sidebar filters;
7. Predict Fare tab;
8. Business Dashboard tab;
9. Model Performance tab;
10. Data Quality tab;
11. About tab.

## Important model rule

The app must continue using the feature contract recorded in `models/model_metadata.json`. Do not add `passenger_count` or pickup/drop coordinates to predictions unless the model is deliberately retrained on valid labeled data.

## Run locally

```powershell
python -m streamlit run app/app.py
```
