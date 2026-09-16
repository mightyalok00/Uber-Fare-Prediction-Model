# `images/charts/` — Static Analysis Charts

These PNG files are exported visual summaries for README/portfolio use.

## Chart meanings

- `fare_distribution.png` — how fare values are distributed.
- `fare_vs_distance.png` — observed relationship between supplied `distance_km` and fare.
- `avg_fare_by_hour.png` — average fare by pickup hour.
- `avg_fare_by_city.png` — average fare by city.
- `status_distribution.png` — count of trip statuses.
- `actual_vs_predicted.png` — final holdout actual fares versus model predictions.

## Maintenance note

If the dataset, train/test split, or final model changes, regenerate these charts so static images remain consistent with the current notebook, model artifacts, and Streamlit dashboard.
