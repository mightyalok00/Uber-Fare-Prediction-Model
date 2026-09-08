# 🚕 Uber Fare Prediction Model

Predict `fare_amount` from a supplied **50,000-trip dataset** using a saved preprocessing and regression pipeline.

🌐 **Live Streamlit App:** https://uber-fare-prediction-model.streamlit.app/

> 💡 Enter trip details, generate an estimated fare, and explore the project dashboard directly in the deployed app.

---

## 🎯 Project Objective

Build a reproducible machine-learning regression workflow that predicts expected Uber fare while keeping preprocessing, model training, validation, persistence, and deployment consistent.

The project includes:

- 🧹 Data cleaning and validation
- 🛠️ Feature preprocessing
- 🤖 Multiple regression models
- 📊 Model evaluation
- 💾 Saved preprocessing + prediction pipeline
- 🔁 Reproducible training
- ✅ Persistence verification
- 📓 Jupyter Notebook analysis
- 🌐 Streamlit deployment

---

## 📊 Verified Model Results

Latest training-ready dataset:

- 🚘 **Completed trips:** 42,538
- 🏋️ **Training rows:** 34,030
- 🧪 **Test rows:** 8,508
- 🔀 **Split:** 80:20
- 🎲 **Random state:** `42`
- ✅ Trip IDs do not overlap between train and test sets
- ✅ Preprocessing is fitted only on training data
- ✅ Each model uses an independent preprocessing instance

| 🤖 Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| 🥇 **Linear Regression** | **2.473802** | **9.566163** | **3.092921** | **0.754264** |
| 🥈 Gradient Boosting | 2.475648 | 9.595738 | 3.097699 | 0.753504 |
| 🥉 Random Forest | 2.511922 | 9.981314 | 3.159322 | 0.743600 |

### 🏆 Selected Model

**Linear Regression** achieved the best results among the tested models and is used as the packaged prediction model.

The original saved model:

- ✅ Loads successfully
- ✅ Predicts successfully
- ✅ Produces predictions matching a fresh refit exactly
- ✅ Produces identical predictions after save/reload

> 📌 **Important:** MAE and RMSE are measured in fare units. MSE is measured in squared fare units. R² is **not percentage accuracy**.

The same test scores are used for model comparison and selection, so they should not be interpreted as an independent final benchmark.

---

## 🚀 Live Demo

Try the deployed application here:

### 🌐 [Uber Fare Prediction Streamlit App](https://uber-fare-prediction-model.streamlit.app/)

The app allows users to:

- 📍 Enter trip information
- 🛣️ Provide planned route distance
- 💳 Select booking-time information
- 🤖 Generate a fare prediction
- 📊 Explore project-level data insights
- 📐 Compare route information with straight-line distance references

---

## ⚙️ Windows Setup

From the repository root, with **Python 3.14** installed:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

---

## 🔁 Reproduce Training & Validation

Run:

```powershell
.\.venv\Scripts\python.exe train.py --output-dir work/run_001
```

This reproduces:

- 🧹 Dataset checks
- ✂️ Train/test split
- 🛠️ Preprocessing
- 🤖 Model training
- 📊 Evaluation metrics
- 💾 Model persistence
- 🔄 Save/reload prediction checks
- ✅ Refit prediction verification

> ⚠️ Choose a new output directory for every run. Existing non-empty output folders are rejected to prevent accidental overwriting.

The original packaged model is never overwritten.

---

## 📓 Open the Jupyter Notebook

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/Uber_Fare_Prediction.ipynb
```

The notebook contains the project analysis, preprocessing workflow, training process, evaluation, and model verification.

---

## 💾 Load the Saved Model

```python
from pathlib import Path
import joblib
import pandas as pd

root = Path.cwd()  # run from repository root

model = joblib.load(
    root / "models/uber_fare_model.pkl"
)

trip = pd.read_csv(
    root / "models/example_input.csv"
)

prediction = model.predict(trip)

print(prediction)
# approximately 20.5074
```

The `.pkl` file is a **joblib artifact containing both preprocessing and the regression estimator**.

> 🔐 Only load serialized model files from trusted sources.

Use the package versions pinned in `requirements.txt` for maximum compatibility.

---

## 📁 Project Structure

```text
Uber_Fare_Prediction/
│
├── 📂 dataset/
│   └── Original, cleaned and Completed-trip training-ready CSVs
│
├── 📂 notebooks/
│   └── Executable analysis and training notebook
│
├── 📂 models/
│   └── Saved pipeline, metadata, metrics and prediction examples
│
├── 📂 app/
│   └── Streamlit prediction app and business dashboard
│
├── 📂 doc/
│   └── Reports, validation notes and imported versions
│
├── 📂 images/charts/
│   └── Existing analysis charts
│
├── 🐍 train.py
├── 📦 requirements.txt
├── 📖 README.md
├── 🙈 .gitignore
└── 📝 PROJECT_SUMMARY.txt
```

---

## 🧹 Data Processing

The original dataset contains approximately **50,000 trip records**.

Cleaning and training preparation include:

- ❌ 3 invalid zero-distance trips removed
- ✅ 49,997 cleaned records retained
- 🚘 Only **Completed** trips used for model training
- ✅ 42,538 Completed trips available for training
- 🚫 No target-percentile trimming applied

---

## 🧠 Feature & Modeling Decisions

Several precautions were taken to avoid leakage and fabricated features.

### 🚫 Excluded Features

The following are not used as model inputs:

- Trip IDs
- Trip status
- Actual duration
- Drop time
- Fare-derived columns
- Other post-trip information

### 👥 Passenger Count

Passenger count does not exist in the supplied dataset.

It is therefore:

**Not fabricated and not estimated.**

### 💳 Payment Method

Payment method is assumed to be known at booking time and is therefore allowed as an input feature.

---

## 📍 Distance Interpretation

The supplied `distance_km` does not reliably align with coordinate-derived Haversine distance.

For this reason:

- 🛣️ Planned route distance is used as the prediction input
- 📐 Straight-line Haversine distance is shown only as a reference
- ⚠️ The educational dataset cannot be used to validate real-world Uber pricing behavior

---

## 🕐 Timestamp Handling

The cleaned dataset truncates drop timestamps to seconds.

The original CSV retains subsecond timestamp precision.

Drop time is not used as a predictive feature.

---

## 📦 Dataset Files

### 📊 Dashboard Dataset

`uber_trips_dataset_50k_cleaned.csv`

Used as the canonical dataset for dashboard exploration.

### 🧠 Training Dataset

`uber_trips_completed_training_ready.csv`

Used as the canonical training source.

### 🗃️ Provenance Dataset

`uber_fare_cleaned.csv`

Retained for provenance and traceability.

---

## ✅ Validation

The project includes explicit validation checks for:

- Data integrity
- Train/test separation
- Preprocessing leakage
- Model persistence
- Prediction reproducibility
- Saved model compatibility

For the detailed verification process, see:

`doc/VALIDATION_REPORT.md`

---

## 🌐 Deployment

The project is deployed using **Streamlit Community Cloud**.

### 🚀 Live Application

👉 https://uber-fare-prediction-model.streamlit.app/

---

## 🛠️ Tech Stack

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 💾 Joblib
- 📊 Matplotlib
- 📓 Jupyter Notebook
- 🌐 Streamlit
- 🐙 GitHub

---

## 📌 Project Highlights

- ✅ Reproducible ML workflow
- ✅ Leakage-aware preprocessing
- ✅ Independent preprocessing per model
- ✅ Multiple regression models compared
- ✅ Saved preprocessing + estimator pipeline
- ✅ Exact persistence verification
- ✅ Streamlit deployment
- ✅ Clean repository structure
- ✅ Documented validation methodology
- ✅ No fabricated passenger-count feature
- ✅ Reproducible train/test split

---

## 👨‍💻 Author

**Alok Agarwal**

📊 Data Science | 🤖 Machine Learning | 📈 Digital Marketing

---

### ⭐ If you find this project useful

Consider giving the repository a **star ⭐** and exploring the live application.

🚕 **Try the model:** https://uber-fare-prediction-model.streamlit.app/
