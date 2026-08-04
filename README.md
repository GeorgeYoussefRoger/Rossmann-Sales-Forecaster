# 📈 Rossmann Sales Forecaster

An end-to-end Machine Learning project that forecasts daily store sales using a complete workflow, from feature engineering and model comparison to hyperparameter tuning and deployment through a FastAPI backend and Streamlit interface.

> Dataset: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)

## 🚀 Features

- End-to-End ML Pipeline (Preprocessing -> Feature Engineering -> Training -> Tuning)
- Time Series Forecasting with lag & rolling features
- MLflow experiment tracking
- Hyperparameter tuning with Optuna
- FastAPI inference API
- Streamlit interactive frontend
- Deployment-ready Scikit-learn pipeline

## 📦 Installation & Usage

- Prerequisites
  - Python 3.12+

1. Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Rossmann-Sales-Forecaster.git
cd Rossmann-Sales-Forecaster
```

2. Create a Virtual Environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run API

```
uvicorn api.api:app
```

5. Run UI

```
streamlit run ui/app.py
```

6. Access:
   - UI -> http://localhost:8501
   - API Docs -> http://localhost:8000/docs

## 🧠 Training

- Install dependencies

```
pip install -r requirements.txt
```

- Train Models

```
python -m src.main
```

- View MLflow experiments

```
mlflow server --backend-store-uri sqlite:///mlruns.db
```

## 🤖 Model Details

- XGBoost outperformed ElasticNet and LightGBM in RMSPE after tuning.

- Test Set Metric:
  - RMSPE (Primary Metric): 0.141
- Notes:
  - RMSPE was used as it was the official metric in the Rossmann Kaggle competition
  - Target variable was log-transformed `log1p` to stabilize variance
  - Three baseline models were compared using MLflow
  - The best-performing baseline model was selected for Optuna hyperparameter tuning before deployment
  - Time-based split was used to prevent data leakage
- Limitations:
  - The model was trained on data from 2013–2015
  - Predictions are limited to this period to avoid distribution shift
  - Lag and rolling inputs must be provided manually in the UI

## 📜 License

This project is licensed under the MIT License.
