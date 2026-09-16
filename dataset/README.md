# `dataset/` — Source and Cleaned Data

This folder contains the instructor-provided trip data used by the project.

## Files

- `uber_trips_dataset_50k.csv` — authoritative raw 50,000-row source file.
- `uber_trips_dataset_50k_cleaned.csv` — cleaned version used by the Streamlit dashboard for faster analysis.

## Data rules

- The raw CSV is the source of truth for reproducible training.
- Cleaning removes invalid/non-positive distances, invalid time order, invalid coordinates, and duplicates.
- `passenger_count` is not present and must not be fabricated.
- The supplied `distance_km` is kept as the route-distance feature because the pickup/drop coordinates are not internally consistent with it.

## Reproducibility

`train.py` starts from the raw CSV and recreates the Completed-trip modeling dataset during each training run. The cleaned CSV is a convenience artifact for dashboard analytics, not a replacement for the raw source.
