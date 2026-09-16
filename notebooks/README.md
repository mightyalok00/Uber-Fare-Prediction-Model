# `notebooks/` — Exploratory Analysis and Business Interpretation

This folder contains the recruiter/assignment-facing Jupyter analysis.

## Files

- `Uber_Fare_Prediction.ipynb` — end-to-end walkthrough of data understanding, cleaning, feature engineering, EDA, exported model evidence, holdout visualization, coefficient interpretation, and final business answers.

## Commenting standard

Every code cell begins with comments describing:

- the cell purpose;
- its main inputs;
- its expected outputs;
- any important data-science decision or limitation.

Additional inline comments explain non-obvious transformations, validation rules, sampling choices, and model-evidence logic.

## Source of truth

The notebook explains and visualizes the workflow, but authoritative model training remains in `../train.py`. Final metrics and selected-model evidence are loaded from `../models/` so the notebook does not create a competing training result.
