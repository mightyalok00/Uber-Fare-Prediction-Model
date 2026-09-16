# Assignment Coverage

This project is evaluated against the instructor-provided **50,000-row Uber trip dataset**. The supplied CSV is the authoritative source for implementation.

## Coverage Summary

| Requirement / Question | Status | Evidence |
| --- | --- | --- |
| Load dataset and inspect structure | Complete | Notebook |
| Shape, columns, dtypes, missing, duplicates, descriptive statistics | Complete | Notebook |
| Missing/invalid coordinates | Complete | Notebook / `train.py` |
| Invalid passenger counts | Not applicable | `passenger_count` absent from supplied CSV |
| Negative/non-positive fares | Complete | `train.py` validation |
| Invalid latitude/longitude | Complete | `train.py` validation |
| Duplicate records | Complete | Notebook / `train.py` |
| pickup year/month/day/hour/day-of-week | Complete | Notebook / `train.py` |
| Coordinate-derived `trip_distance` | Complete as diagnostic | Haversine distance calculated and compared with supplied `distance_km` |
| Fare distribution | Complete | Notebook / Streamlit |
| Passenger-count distribution | N/A | Source column absent |
| Fare vs distance | Complete | Notebook / Streamlit |
| Fare vs passenger count | N/A | Source column absent |
| Fare by hour/day/month | Complete | Notebook / Streamlit |
| Trip-distance distribution | Complete | Notebook / Streamlit |
| Correlation matrix | Complete | Notebook / Streamlit |
| Linear Regression | Complete | `train.py` |
| Random Forest | Complete | `train.py` |
| Gradient Boosting | Complete | `train.py` |
| MAE/MSE/RMSE/R² | Complete | `models/final_test_metrics.csv` |
| Compare models | Complete | Training-only 5-fold CV |
| Best model | Complete | Linear Regression |
| Feature influence | Complete | Streamlit Model Performance |
| Actual vs Predicted | Complete | Saved final model + fixed untouched holdout |
| New-trip fare estimate | Complete | Streamlit Predict Fare |

## Verified Final Results

- Raw rows: **50,000**
- Valid cleaned rows: **49,997**
- Completed modeling rows: **42,538**
- Training rows: **34,030**
- Holdout rows: **8,508**
- Best model by training-only 5-fold CV RMSE: **Linear Regression**
- Final holdout MAE: **2.474**
- Final holdout MSE: **9.567**
- Final holdout RMSE: **3.093**
- Final holdout R²: **0.754**

## Business Questions — Final Answers

1. **What factors most strongly influence Uber fares?**  
   Supplied trip distance is the strongest supported observed signal. The app also displays transformed model coefficients as influence diagnostics, without claiming causality.

2. **How does trip distance affect fare?**  
   Fare and supplied `distance_km` have a strong positive relationship (Pearson correlation ≈ **0.871** on valid completed rides).

3. **Does passenger count significantly affect fare?**  
   **Cannot be determined** because `passenger_count` is absent from the supplied dataset.

4. **Which hours have higher average fares?**  
   **06:00** has the highest average fare in the full valid completed-ride data.

5. **Which model performs best?**  
   **Linear Regression**, selected using the lowest mean 5-fold CV RMSE on the training partition.

6. **How accurate is the final model?**  
   Untouched holdout: MAE **2.474**, RMSE **3.093**, R² **0.754**.

7. **Can the model provide a reasonable fare estimate for a new trip?**  
   **Yes.** The Streamlit app loads the final saved model and predicts a new fare from supported pre-trip inputs.

## Passenger-Count Limitation

The assignment describes a Kaggle-style schema containing `passenger_count`, but the actual instructor-provided 50K CSV does not contain it. The project therefore does not fabricate, infer, or randomly generate passenger counts.

## Coordinate-Distance Decision

Haversine distance is calculated to fulfill the assignment's coordinate-engineering requirement as a diagnostic. On the final completed modeling rows, its correlation with supplied `distance_km` is approximately **0.00069**, and only about **1.50%** of rows are within 0.1 km. The supplied `distance_km` is retained for prediction because it is the route-distance field associated with fare in this educational dataset.

## Runtime / Reproducibility Strategy

- The committed model is trained and serialization-tested under **Python 3.12**.
- GitHub Actions independently reproduces training under Python 3.12.
- A separate CI job validates that the committed model loads, predicts, and that Streamlit starts under **Python 3.14.7**.
- Full model retraining is intentionally not performed under Python 3.14.7 because the current native scientific stack produced a segmentation fault during training on the hosted runner.
