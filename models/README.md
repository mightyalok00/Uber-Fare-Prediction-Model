# `models/` — Final Model and Evaluation Evidence

This folder stores the committed model bundle and the evidence needed to verify it.

## Files

- `uber_fare_model.pkl` — final serialized scikit-learn Pipeline used by Streamlit.
- `model_metadata.json` — feature contract, selected model, environment versions, row counts, diagnostics, and persistence status.
- `cross_validation_results.csv` — training-only 5-fold CV results for compared models.
- `model_comparison.csv` — recruiter-friendly alias of the CV comparison table.
- `final_test_metrics.csv` — untouched holdout MAE, MSE, RMSE, and R².
- `example_input.csv` — one supported prediction input example.
- `example_prediction.json` — saved-model prediction for the example input.

## Model-selection logic

1. Split once into train and holdout test sets.
2. Compare Linear Regression, Random Forest, and Gradient Boosting using 5-fold CV on training data only.
3. Select the lowest mean CV RMSE.
4. Fit the selected model on all training rows.
5. Evaluate once on the untouched holdout.
6. Save/reload the model and verify prediction consistency.

## Important

Do not manually edit binary model files or metadata values. Regenerate model artifacts through `train.py` so the files remain internally consistent.
