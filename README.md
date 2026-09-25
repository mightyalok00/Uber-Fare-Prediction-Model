# Uber Fare Prediction Model

<p align="center">
  <strong>End-to-end machine learning project for Uber fare estimation</strong><br>
  Data validation • Feature engineering • Model selection • Evaluation • Streamlit deployment
</p>

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge" alt="Educational Project">
</p>

<p align="center">
  <a href="https://uber-fare-prediction-model.streamlit.app/"><strong>🚀 Open the Live Application</strong></a>
</p>

---

## 📌 Overview

This project is an end-to-end regression workflow for estimating **Uber trip fares** from historical ride data.

It demonstrates a production-minded machine learning workflow covering:

- Data validation and cleaning
- Exploratory and business-focused analysis
- Feature engineering
- Leakage-safe preprocessing
- Cross-validation and model comparison
- Untouched holdout evaluation
- Model serialization and reload validation
- Interactive Streamlit deployment
- Automated GitHub Actions checks

> **Dataset scope:** The results in this repository are specific to the supplied educational 50,000-trip dataset. They should not be interpreted as a description of real-world Uber pricing.

## 🚀 Live Demo

**Try the deployed Streamlit application:**

### 👉 [uber-fare-prediction-model.streamlit.app](https://uber-fare-prediction-model.streamlit.app/)

The application includes:

| Section | Purpose |
|---|---|
| **Predict Fare** | Estimate fare for a new trip using supported inputs |
| **Business Dashboard** | Explore fares, distance, payment methods, cities and trip status |
| **Model Performance** | Review evaluation metrics, model comparison and predictions |
| **Data Quality** | Inspect dataset and coordinate-distance diagnostics |
| **About** | Review scope, methodology and project limitations |

---

## 🎯 Business Objective

Estimate `fare_amount` before a trip is completed using information supported by the supplied dataset.

The active model uses:

- City
- Payment method
- Trip distance
- Pickup year
- Pickup month
- Pickup day
- Pickup hour
- Day of week
- Weekend indicator
- Rush-hour indicator

The model intentionally excludes fields that are unavailable at prediction time, target-derived information, identifiers, and coordinate fields that were found to be inconsistent with the supplied `distance_km`.

---

## 📊 Key Results

### Data quality

| Metric | Result |
|---|---:|
| Raw rows | 50,000 |
| Raw columns | 14 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Valid cleaned rows | 49,997 |
| Completed rides used for modeling | 42,538 |
| Fare vs. supplied distance correlation | ~0.871 |

### Model selection

Three regression approaches were compared using **5-fold cross-validation on the training partition only**.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| **Linear Regression** | **2.4740** | **3.0868** | **0.7590** |
| Gradient Boosting | 2.4761 | 3.0911 | 0.7583 |
| Random Forest | 2.5272 | 3.1774 | 0.7447 |

The final pipeline uses **Linear Regression**, selected according to the project's predefined model-selection criterion: lowest mean CV RMSE.

### Final untouched holdout

| Metric | Result |
|---|---:|
| MAE | **2.474** |
| MSE | **9.567** |
| RMSE | **3.093** |
| R² | **0.754** |

> R² is a regression metric, not percentage prediction accuracy. MAE and RMSE represent prediction error in fare units.

---

## 🔎 Important Data Finding

The assignment asks for distance to be calculated from pickup and drop-off coordinates.

This project calculates **Haversine distance** as a validation diagnostic. However, the coordinate-derived distance is internally inconsistent with the supplied `distance_km`:

- Correlation with supplied distance: approximately **0.00069**
- Rows within 0.1 km of supplied distance: approximately **1.50%**

Because of this discrepancy, coordinate-derived distance is **not used as a prediction feature**.

The supplied `distance_km` field is retained because it is the distance field supported by the educational dataset and shows a strong observed relationship with fare.

---

## 🧠 Machine Learning Workflow

The training pipeline follows this sequence:

```text
Raw Dataset
    │
    ▼
Data Validation & Cleaning
    │
    ▼
Completed Rides
    │
    ▼
Feature Engineering
    │
    ├── Time features
    ├── Weekend indicator
    └── Rush-hour indicator
    │
    ▼
80/20 Train-Test Split
    │
    ├───────────────┐
    ▼               ▼
Training Set     Untouched Test Set
    │
    ▼
5-Fold Cross-Validation
    │
    ├── Linear Regression
    ├── Random Forest
    └── Gradient Boosting
    │
    ▼
Model Selection
    │
    ▼
Final Pipeline Fit
    │
    ▼
Holdout Evaluation
    │
    ▼
Serialized Model
    │
    ▼
Streamlit Application
```

### Leakage prevention

The project deliberately keeps the final test partition untouched during model selection.

Preprocessing is contained inside the scikit-learn pipeline, so transformations are fitted within each training fold rather than on the complete dataset before cross-validation.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core development and model training |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Preprocessing, regression and evaluation |
| **Joblib** | Model serialization |
| **Matplotlib** | Visualization |
| **Streamlit** | Interactive web application |
| **GitHub Actions** | Automated CI checks |

---

## 📁 Repository Structure

```text
Uber-Fare-Prediction-Model/
│
├── .github/
│   └── workflows/
│       └── python-checks.yml
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── uber_trips_dataset_50k.csv
│   └── uber_trips_dataset_50k_cleaned.csv
│
├── doc/
│   ├── ASSIGNMENT_COVERAGE.md
│   ├── PASSENGER_COUNT_NOTE.txt
│   └── README.md
│
├── images/
│   └── charts/
│
├── models/
│   ├── uber_fare_model.pkl
│   ├── model_metadata.json
│   ├── cross_validation_results.csv
│   ├── model_comparison.csv
│   ├── final_test_metrics.csv
│   ├── example_input.csv
│   └── example_prediction.json
│
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
│
├── train.py
├── requirements.txt
├── requirements-notebook.txt
└── README.md
```

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mightyalok00/Uber-Fare-Prediction-Model.git
cd Uber-Fare-Prediction-Model
```

### 2. Create a virtual environment

**Windows:**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional notebook dependencies:

```bash
pip install -r requirements-notebook.txt
```

### 4. Launch the Streamlit app

```bash
python -m streamlit run app/app.py
```

The application will be available at the local Streamlit address shown in your terminal.

---

## 🏋️ Reproduce Training

The authoritative model-training environment is **Python 3.12**.

Run:

```bash
python train.py --output-dir work/run_001
```

The training script produces:

- `uber_fare_model.pkl`
- `model_metadata.json`
- `cross_validation_results.csv`
- `model_comparison.csv`
- `final_test_metrics.csv`
- `example_input.csv`
- `example_prediction.json`

The pipeline is reloaded after serialization and its predictions are checked against the original fitted pipeline.

---

## 🔬 Model Contract

### Included features

```text
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
```

### Excluded features

- Pickup/drop-off latitude and longitude
- `passenger_count` — not present in the supplied CSV
- `drop_time`
- `status`
- Trip, rider and driver identifiers
- Target-derived fields

This keeps the prediction interface aligned with the information available to the application.

---

## 📈 Business Insights

Based on the supplied dataset:

1. **Trip distance is the strongest supported observed fare signal.**
2. Fare has a strong positive relationship with the supplied `distance_km`.
3. **06:00** has the highest average fare among valid completed rides in this dataset.
4. Passenger-count impact cannot be evaluated because `passenger_count` is absent.
5. The model provides fare estimates through the Streamlit prediction interface.

These are dataset-specific observations and are not causal claims about Uber's real-world pricing system.

---

## 🤖 Continuous Integration

GitHub Actions contains two independent validation paths:

### Python 3.12
- Compiles the project
- Installs runtime dependencies
- Retrains the model
- Reloads the exported model
- Verifies a finite prediction

### Python 3.14.7
- Compiles the source
- Loads the committed Python 3.12-trained model
- Verifies prediction compatibility
- Runs a Streamlit smoke test

CI does not modify or commit repository files.

---

## ⚠️ Limitations

This project has several important limitations:

- The dataset is educational and should not be treated as a production Uber pricing dataset.
- `passenger_count` is unavailable.
- Coordinate-derived distance is inconsistent with supplied `distance_km`.
- `payment_method` is assumed to be available for this educational prediction workflow.
- Real-world fares can depend on variables not represented here, including ride category, surge, traffic, tolls, weather and local demand.
- Model performance should not be interpreted as guaranteed real-world pricing accuracy.

---

## 📚 Documentation

Additional project documentation is available in the repository:

- `doc/ASSIGNMENT_COVERAGE.md`
- `doc/PASSENGER_COUNT_NOTE.txt`
- `doc/README.md`
- `notebooks/Uber_Fare_Prediction.ipynb`

---

## 👤 Author

**Alok Agarwal**

Data Analytics • Data Science • Machine Learning • Digital Marketing

---

## ⭐ Support the Project

If you find this project useful for learning or reference, consider **starring the repository**.

<p align="center">
  <a href="https://github.com/mightyalok00/Uber-Fare-Prediction-Model">
    <strong>⭐ View the GitHub Repository</strong>
  </a>
  &nbsp; • &nbsp;
  <a href="https://uber-fare-prediction-model.streamlit.app/">
    <strong>🚀 Try the Live Demo</strong>
  </a>
</p>
