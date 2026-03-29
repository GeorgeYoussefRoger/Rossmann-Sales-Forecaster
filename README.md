# 📈 Rossmann Sales Forecaster

An end-to-end Time Series Machine Learning system that forecasts daily store sales using a production-style pipeline, with feature engineering, model comparison, and an interactive UI for predictions.

> Built on the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales)

## 🚀 Features

- End-to-End ML Pipeline (Preprocessing -> Feature Engineering -> Training -> Tuning)
- Time Series Forecasting with lag & rolling features
- Custom evaluation metric (RMSPE)
- Experiment Tracking with MLflow
- Hyperparameter tuning with Optuna
- FastAPI for real-time predictions
- Streamlit UI for interaction
- Dockerized services (API + UI)
- CI/CD with GitHub Actions
- Multi-service architecture (separate API & UI)

## 📦 Installation & Usage

### Prerequisites

- Python 3.12+

### Run Locally

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

4. Train Model

```
python -m src.pipeline
```

5. Run API

```
pip install -r api/requirements.txt
uvicorn api.main:app
```

6. Run UI

```
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

7. Access:
   - UI -> http://localhost:8501
   - API Docs -> http://localhost:8000/docs

### Run with Docker (Recommended)

1. Build Docker images (API + UI)

```
docker-compose up --build
```

2. Access:

- UI -> http://localhost:8501
- API Docs -> http://localhost:8000/docs

## 📊 Model Performance

- XGBoost outperformed Ridge and LightGBM in RMSPE after tuning.

- Test Set Metric:
  - RMSPE (Primary Metric): 0.145
- Notes:
  - RMSPE was used as it was the official metric in the Rossmann Kaggle competition
  - Target variable was log-transformed (log1p) to stabilize variance
  - Ridge struggled because it cannot capture non-linear interactions
  - Tuning efforts were focused on boosting models (XGBoost, LightGBM) where gains were significant
  - Time-based split was used to prevent data leakage
- Limitations:
  - The model was trained on data from 2013–2015
  - Predictions are limited to this period to avoid distribution shift
  - Lag and rolling inputs must be provided manually in the UI
  - Lag and rolling features are computed using past values only via shifting, ensuring no future data leakage.

## 📂 Project Structure

```
Rossmann-Sales-Forecaster/
├── .github/workflows/     # GitHub Actions CI/CD
├── api/                   # FastAPI
├── data/                  # Rossmann Dataset
├── models/                # Trained Models
├── notebooks/             # Exploration Notebook
├── src/                   # ML Pipeline
├── ui/                    # Streamlit UI
└── requirements.txt       # Training Requirements
```

## 📜 License

- This project is licensed under the MIT License.
- See the [LICENSE](LICENSE) file for more details.
