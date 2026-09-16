# `images/` — Portfolio Visual Assets

This folder stores static charts used by the repository documentation and portfolio presentation.

## `charts/`

Current chart assets include:

- `actual_vs_predicted.png` — holdout prediction quality.
- `avg_fare_by_city.png` — average fare comparison by city.
- `avg_fare_by_hour.png` — hourly fare pattern.
- `fare_distribution.png` — fare distribution.
- `fare_vs_distance.png` — fare relationship with supplied trip distance.
- `status_distribution.png` — trip-status distribution.

## Guidance

These images are presentation artifacts. The interactive Streamlit dashboard recreates analysis from data at runtime, so static charts should not be treated as the primary source of current metrics if the dataset or model changes.
