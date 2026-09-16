# Assignment Coverage

This project is evaluated against the instructor-provided **50,000-row Uber trip dataset**. The supplied CSV is the authoritative source for implementation.

## Coverage Summary

| Requirement / Question | Status | Evidence |
| --- | --- | --- |
| Load dataset and inspect structure | Complete | Notebook |
| Shape, columns, dtypes, missing, duplicates, descriptive statistics | Complete | Notebook / validation |
| Missing/invalid coordinates | Complete | Validation and Haversine diagnostic |
| Invalid passenger counts | Not applicable | `passenger_count` absent from supplied CSV |
| Negative/extreme fares | Complete | Cleaning/EDA |
| Invalid latitude/longitude | Complete | Training validation |
| Duplicate records | Complete | Validation |
| pickup year/month/day/hour/day-of-week | Complete | Notebook / `train.py` |
| Coordinate-derived `trip_distance` | Complete as diagnostic | Haversine distance calculated and compared with supplied `distance_km` |
| Fare distribution | Complete | Notebook / Streamlit |
| Passenger-count distribution | N/A | Source column absent |
| Fare vs distance | Complete | Notebook / Streamlit |
| Fare vs passenger count | N/A | Source column absent |
| Fare by hour/day/month | Complete | Notebook / Streamlit |
| Trip-distance distribution | Complete | Notebook / Streamlit |
| Correlation matrix | Complete | Notebook / Streamlit |
| Linear Regression | Complete | Training pipeline |
| Random Forest | Complete | Training pipeline |
| Gradient Boosting | Complete | Training pipeline |
| MAE/MSE/RMSE/R² | Complete | Final metrics |
| Compare models | Complete | Training-only 5-fold CV |
| Best model | Complete | Linear Regression |
| Feature influence | Complete | Streamlit Model Performance |
| Actual vs Predicted | Complete | Untouched holdout predictions |
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
   Fare and supplied `distance_km` have a strong positive relationship (Pearson correlation ≈ **0.871**).

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

Haversine distance is calculated to fulfill the assignment's coordinate-engineering requirement as a diagnostic. It is not used for prediction because its correlation with supplied `distance_km` is only **0.0016**, with just **1.50%** of rows within 0.1 km. The supplied `distance_km` is retained for prediction because it is the supported route-distance field associated with fare.
