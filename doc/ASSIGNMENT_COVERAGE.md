# Assignment Coverage

This project is evaluated against the instructor-provided **50,000-row Uber trip dataset**. The supplied CSV is the authoritative source for the project.

## Coverage summary

| Requirement / question | Status | Evidence |
| --- | --- | --- |
| Load and understand the dataset | Complete | Notebook, validation report, cleaned datasets |
| Check shape, columns, dtypes, missing values and duplicates | Complete | Notebook and validation report |
| Handle invalid fares, distances, coordinates and timestamps | Complete | Cleaning workflow and `train.py` validation |
| Create pickup year/month/day/hour/day-of-week features | Complete | Training-ready dataset and `train.py` |
| Analyze fare distribution | Complete | Notebook and Streamlit Business Dashboard |
| Analyze trip-distance distribution | Complete | Notebook and Streamlit Business Dashboard |
| Analyze fare vs distance | Complete | Notebook and Streamlit Business Dashboard |
| Analyze fare by pickup hour | Complete | Notebook and Streamlit Business Dashboard |
| Analyze fare by day of week | Complete | Streamlit Business Dashboard |
| Analyze fare by month | Complete | Streamlit Business Dashboard |
| Correlation analysis | Complete | Streamlit Business Dashboard |
| Linear Regression | Complete | Notebook, `train.py`, packaged model evidence |
| Random Forest Regressor | Complete | Notebook and `train.py` |
| Gradient Boosting Regressor | Complete | Notebook and `train.py` |
| MAE, MSE, RMSE and R² | Complete | Model metadata and Streamlit Model Performance tab |
| Compare models and identify the best model | Complete | Model comparison and training workflow |
| Estimate fare for a new trip | Complete | Streamlit Predict Fare tab |
| Show model influence / feature importance | Complete | Streamlit Model Performance tab |
| Actual vs Predicted visualization | Complete | Streamlit Model Performance tab |
| Passenger-count distribution | Not applicable to supplied data | `passenger_count` is absent from the instructor-provided CSV |
| Fare vs passenger count | Not applicable to supplied data | No ground-truth passenger-count field exists |
| Passenger-count business conclusion | Not applicable to supplied data | No supported conclusion can be made without labels |

## Business findings from the supplied data

Using valid **Completed** rides from the supplied dataset:

- **42,538** completed rides are available for modeling after validation.
- Fare and supplied trip distance have a strong positive relationship; the observed Pearson correlation is approximately **0.871** on the completed valid rides.
- In the current full completed-ride dataset, **06:00** has the highest average fare among pickup hours.
- **Monday** has the highest average fare among days of the week in the current data.
- **January** has the highest average fare among months in the current data.
- The packaged baseline comparison identifies **Linear Regression** as the best of the three compared models by RMSE, with MAE ≈ **2.474**, RMSE ≈ **3.093**, and R² ≈ **0.754**.
- The Streamlit app can generate a fare estimate for a new trip from pre-trip inputs.

These are dataset-specific findings, not claims about Uber pricing in the real world.

## Passenger-count limitation

The instructor-provided 50K CSV does not contain a `passenger_count` column. The project therefore does **not** fabricate, randomly generate, or infer passenger counts. Any passenger-count analysis would require a different labeled source dataset and would no longer be an analysis of the instructor-provided data.

## Distance note

The supplied dataset already contains `distance_km`, which is used as the planned route-distance feature. Coordinate-derived Haversine distance may be used for reference or validation, but it is not presented as a replacement for the supplied route-distance field.
