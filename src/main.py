import mlflow
import os
import joblib
from sklearn.model_selection import train_test_split

from src.config import *
from src.preprocess import preprocess, create_preprocessor
from src.features import add_date_features, add_promo_features, add_lag_rolling_features
from src.models import models, build_pipeline
from src.train import train
from src.tune import tune

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = preprocess()
    df = add_date_features(df)
    df = add_promo_features(df)
    df = add_lag_rolling_features(df)

    df = df.sort_values("Date").reset_index(drop=True)
    X = df.drop(['Sales', 'Customers', 'Open', 'Date', 'PromoInterval'], axis=1)
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    baseline_scores = {}
    for name, model in models.items():
        score = train(X_train, y_train, X_test, y_test, model, name)
        baseline_scores[name] = score

    best_model_name = min(baseline_scores, key=baseline_scores.get)
    best_model = models[best_model_name]

    tuned_model = tune(X_train, X_test, y_train, y_test, best_model, best_model_name)
    pipeline = build_pipeline(preprocessor, tuned_model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'final_pipeline.pkl'))

if __name__ == "__main__":
    main()