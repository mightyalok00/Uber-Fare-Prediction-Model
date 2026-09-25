# 🚕 Uber Fare Prediction Model

<p align="center">
  <strong>End-to-end machine learning project for Uber fare estimation</strong><br>
  Data validation • Feature engineering • Model selection • Evaluation • Streamlit deployment
</p>

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open the live Streamlit application">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn 1.8.0">
  <img src="https://img.shields.io/github/actions/workflow/status/mightyalok00/Uber-Fare-Prediction-Model/python-checks.yml?label=CI&logo=githubactions&style=for-the-badge" alt="GitHub Actions CI">
</p>

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/"><strong>🚀 Open Live Application</strong></a>
  &nbsp; • &nbsp;
  <a href="#-model-performance"><strong>📊 View Results</strong></a>
  &nbsp; • &nbsp;
  <a href="#-documentation"><strong>📚 Documentation</strong></a>
</p>

---

## 📌 Project at a Glance

| Item | Details |
|---|---|
| **Task** | Regression |
| **Target** | `fare_amount` |
| **Dataset** | 50,000 trip records |
| **Completed rides modeled** | 42,538 |
| **Models compared** | Linear Regression, Random Forest, Gradient Boosting |
| **Validation** | 5-fold cross-validation on training data |
| **Final evaluation** | 20% untouched holdout |
| **Selected model** | Linear Regression |
| **Holdout MAE** | **2.474** |
| **Holdout RMSE** | **3.093** |
| **Holdout R²** | **0.754** |
| **Deployment** | Streamlit |

## 🎯 Overview

This project builds an end-to-end regression pipeline for estimating Uber trip fares from the supplied educational dataset.

The workflow covers:

- Data validation and cleaning
- Exploratory and business-focused analysis
- Time-based feature engineering
- Leakage-safe preprocessing
- 5-fold cross-validation
- Regression model comparison
- Untouched holdout evaluation
- Model serialization and reload validation
- Interactive Streamlit deployment
- Automated GitHub Actions validation

> **Scope:** All reported results are specific to the supplied educational 50,000-row dataset. They should not be interpreted as a description of Uber's real-world pricing system.

## 🚀 Live Application

**Try the deployed application:**

### 👉 [uber-fare-prediction-model.streamlit.app](https://uber-fare-prediction-model.streamlit.app/)

The Streamlit application provides:

| App section | What it does |
|---|---|
| **Predict Fare** | Estimates a fare for a new trip |
| **Business Dashboard** | Explores fares, distance, payment methods, cities and ride status |
| **Model Performance** | Displays model metrics, comparison results and predictions |
| **Data Quality** | Shows validation and coordinate-distance diagnostics |
| **About** | Explains methodology, scope and limitations |

## 📊 Model Performance

### Cross-validation comparison

The models were compared using **5-fold cross-validation on the training partition only**.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| **Linear Regression** | **2.4740** | **3.0868** | **0.7590** |
| Gradient Boosting | 2.4761 | 3.0911 | 0.7583 |
| Random Forest | 2.5272 | 3.1774 | 0.7447 |

### CV RMSE

~~~mermaid
xychart-beta
    title "Mean 5-Fold CV RMSE — Lower is Better"
    x-axis ["Linear Regression", "Gradient Boosting", "Random Forest"]
    y-axis "RMSE" 3.0 --> 3.3
    bar [3.0868, 3.0911, 3.1774]
~~~

**Model selection:** Linear Regression was selected using the predefined criterion of lowest mean cross-validation RMSE.

### Final untouched holdout

| Metric | Result |
|---|---:|
| **MAE** | **2.474** |
| **MSE** | **9.567** |
| **RMSE** | **3.093** |
| **R²** | **0.754** |

> R² is a regression metric, not percentage prediction accuracy. MAE and RMSE represent error in the dataset's fare units.

## 🧠 Methodology

The project follows a leakage-aware training workflow:

~~~mermaid
flowchart TD
    A[Raw Dataset] --> B[Data Validation]
    B --> C[Completed Rides]
    C --> D[Feature Engineering]
    D --> E[80/20 Train-Holdout Split]
    E --> F[5-Fold Cross-Validation]
    F --> G[Linear Regression]
    F --> H[Random Forest]
    F --> I[Gradient Boosting]
    G --> J[Select Lowest CV RMSE]
    H --> J
    I --> J
    J --> K[Fit Final Pipeline]
    K --> L[Untouched Holdout Evaluation]
    L --> M[Serialized Model]
    M --> N[Streamlit Application]
~~~

### Leakage prevention

- The holdout partition is not used for model selection.
- Preprocessing is contained inside the scikit-learn pipeline.
- Cross-validation fits transformations within each training fold.
- The selected pipeline is evaluated once on the untouched holdout.
- The saved model is reloaded and checked for prediction consistency.

## 📚 Dataset

The supplied dataset contains **50,000 trip records**. After validation and cleaning:

- **49,997** rows remain valid.
- **42,538** completed rides are used for modeling.
- The target variable is `fare_amount`.

### Dataset dictionary

| Field / feature | Description | Project role |
|---|---|---|
| `key` | Trip identifier | Excluded from modeling |
| `fare_amount` | Recorded trip fare | **Target** |
| `pickup_datetime` | Pickup date and time | Source for time features |
| `pickup_longitude` / `pickup_latitude` | Pickup coordinates | Validation / Haversine diagnostic |
| `dropoff_longitude` / `dropoff_latitude` | Drop-off coordinates | Validation / Haversine diagnostic |
| `city` | City/category | Model feature |
| `payment_method` | Payment method/category | Model feature |
| `distance_km` | Supplied trip distance | **Model feature** |
| `pickup_year` | Pickup year | Engineered feature |
| `pickup_month` | Pickup month | Engineered feature |
| `pickup_day` | Pickup day | Engineered feature |
| `pickup_hour` | Pickup hour | Engineered feature |
| `day_of_week` | Day of week | Engineered feature |
| `is_weekend` | Weekend indicator | Engineered feature |
| `is_rush_hour` | Rush-hour indicator | Engineered feature |

> **Passenger count:** The assignment references `passenger_count`, but the supplied 50K CSV does not contain that field. The project does not fabricate or infer passenger counts.

## 🔎 Key Findings

### Distance is strongly associated with fare

The supplied `distance_km` has a Pearson correlation of approximately **0.871** with fare on valid completed rides.

### Coordinate-derived distance is treated as a diagnostic

The project calculates Haversine distance from pickup and drop-off coordinates to satisfy the coordinate-engineering requirement and validate the supplied distance field.

However:

- Correlation with supplied `distance_km`: approximately **0.00069**
- Rows within 0.1 km of supplied distance: approximately **1.50%**

Because of this discrepancy, Haversine distance is **not used as a prediction feature**. The supplied `distance_km` field is retained.

### Time patterns

In the valid completed-ride data, **06:00** has the highest average fare.

These findings describe this dataset and are not causal claims about Uber's pricing system.

## 🤖 Prediction Features

The final model uses:

~~~text
city
payment_method
distance_km
pickup_year
pickup_month
pickup_day
pickup_hour
day_of_week
is_weekend
is_rush_hour
~~~

The model excludes identifiers, target-derived fields, passenger count, coordinate fields, and other information not supported by the final prediction contract.

## 📁 Repository Structure

~~~text
Uber-Fare-Prediction-Model/
│
├── app/                    # Streamlit application
├── dataset/                # Raw and cleaned datasets
├── models/                 # Trained model and evaluation outputs
├── notebooks/              # EDA and analysis
├── images/                 # Charts and project visuals
├── doc/                    # Detailed documentation
├── .github/workflows/      # Continuous integration
│
├── train.py                # Reproducible model-training pipeline
├── requirements.txt        # Runtime dependencies
├── requirements-notebook.txt
└── README.md
~~~

## 💻 Run Locally

### 1. Clone

~~~bash
git clone https://github.com/mightyalok00/Uber-Fare-Prediction-Model.git
cd Uber-Fare-Prediction-Model
~~~

### 2. Create a virtual environment

**Windows**

~~~powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
~~~

**macOS / Linux**

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Launch the application

~~~bash
python -m streamlit run app/app.py
~~~

## 🏋️ Reproduce Training

The authoritative training environment is **Python 3.12**.

~~~bash
python train.py --output-dir work/run_001
~~~

The training pipeline exports:

- `uber_fare_model.pkl`
- `model_metadata.json`
- `cross_validation_results.csv`
- `model_comparison.csv`
- `final_test_metrics.csv`
- `example_input.csv`
- `example_prediction.json`

The exported model is reloaded after serialization and checked for prediction consistency.

## 🤖 Continuous Integration

GitHub Actions validates the project in two paths:

**Python 3.12**
- Compiles the project
- Installs runtime dependencies
- Retrains the model
- Reloads the exported model
- Verifies a finite prediction

**Python 3.14.7**
- Compiles the source
- Loads the committed Python 3.12-trained model
- Verifies prediction compatibility
- Runs a Streamlit smoke test

CI does not modify or commit repository files.

## ⚠️ Limitations

- The dataset is educational rather than a production Uber pricing dataset.
- `passenger_count` is unavailable in the supplied CSV.
- Coordinate-derived distance is inconsistent with supplied `distance_km`.
- `payment_method` is assumed to be available for this educational prediction workflow.
- Real-world fares may depend on ride category, surge, traffic, tolls, weather, demand and other variables not represented here.
- Model performance should not be interpreted as guaranteed real-world pricing accuracy.

## 📚 Documentation

For deeper project details:

- [Assignment coverage](doc/ASSIGNMENT_COVERAGE.md)
- [Passenger-count note](doc/PASSENGER_COUNT_NOTE.txt)
- [Project documentation](doc/README.md)
- [EDA and analysis notebook](notebooks/Uber_Fare_Prediction.ipynb)
- [Model comparison results](models/model_comparison.csv)
- [Final holdout metrics](models/final_test_metrics.csv)

## 👤 Author

**Alok Agarwal**

Data Analytics • Data Science • Machine Learning • Digital Marketing

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/"><strong>🚀 Try the Live Demo</strong></a>
  &nbsp; • &nbsp;
  <a href="https://github.com/mightyalok00/Uber-Fare-Prediction-Model"><strong>⭐ View the Repository</strong></a>
</p>

---

<p align="center">
  <sub>Built as an educational machine-learning project with reproducibility, evaluation and deployment in mind.</sub>
</p>
