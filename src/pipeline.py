import mlflow
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import *
from src.load_data import load_data
from src.preprocess import preprocess
from src.features import FeatureBuilder, add_lag_rolling_features
from src.train import train
from src.tune import tune

def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['a', 'b', 'c']]), ['Assortment']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['StateHoliday', 'StoreType', 'DayOfWeek'])
    ], remainder='passthrough')

    pipeline = Pipeline([
        ("features", FeatureBuilder()), 
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def run_pipeline():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Rossmann-Sales-Forecasting")

    df = load_data('data')
    df = preprocess(df)
    df = add_lag_rolling_features(df)

    df = df.sort_values("Date").reset_index(drop=True)
    X = df.drop(['Sales', 'Customers', 'Open'], axis=1)
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

    baseline_scores = {}
    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        score = train(X_train, y_train, pipeline, name)
        baseline_scores[name] = score

    top2 = sorted(baseline_scores, key=baseline_scores.get)[:2]

    best_score = float('inf')
    best_pipeline = None
    best_pipeline_name = None
    for name in top2:
        model = MODELS[name]
        pipeline = build_pipeline(model)
        score, tuned_pipeline = tune(X_train, X_test, y_train, y_test, pipeline, name)
        if score < best_score:
            best_score = score
            best_pipeline = tuned_pipeline
            best_pipeline_name = name

    print(f"Best model: {best_pipeline_name} with Test RMSPE: {best_score:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, os.path.join('models', 'final_model.pkl'))

if __name__ == "__main__":
    run_pipeline()