import numpy as np
import mlflow
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer

from src.metrics import rmspe

def train(X_train, y_train, pipeline, name):
    """
    Train the model and log results to MLflow.
    """
    with mlflow.start_run(run_name=f'{name}_Baseline'):
        print(f"Training {name}...") 

        tscv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(rmspe, greater_is_better=False)
        score = -cross_val_score(pipeline, X_train, np.log1p(y_train), cv=tscv, scoring=scorer).mean()

        pipeline.fit(X_train, np.log1p(y_train))  
        mlflow.log_metric('CV RMSPE', score)
        mlflow.sklearn.log_model(pipeline, name="model")
        print(f"{name} CV RMSPE: {score:.4f}")

        return score