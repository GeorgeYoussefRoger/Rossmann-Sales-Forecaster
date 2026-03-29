import numpy as np
import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.base import clone

from src.metrics import rmspe

def tune(X_train, X_test, y_train, y_test, pipeline, name):
    """
    Tune the model using Optuna and log results to MLflow.
    """
    def objective(trial):
        if name == 'Ridge':
            params = {
                'model__alpha': trial.suggest_float('model__alpha', 0.1, 10.0, log=True)
            }
        elif name == 'XGBoost':
            params = {
                'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 500),
                'model__max_depth': trial.suggest_int('model__max_depth', 4, 8),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 0.03, 0.2),
                'model__subsample': trial.suggest_float('model__subsample', 0.5, 1.0),
                'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.5, 1.0)
            }
        elif name == 'LightGBM':
            params = {
                'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 500),
                'model__num_leaves': trial.suggest_int('model__num_leaves', 31, 63),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 0.03, 0.2),
                'model__feature_fraction': trial.suggest_float('model__feature_fraction', 0.5, 1.0),
                'model__bagging_fraction': trial.suggest_float('model__bagging_fraction', 0.5, 1.0)
            }
        trial_pipeline = clone(pipeline)
        trial_pipeline.set_params(**params)
        tscv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(rmspe, greater_is_better=False)
        return -cross_val_score(trial_pipeline, X_train, np.log1p(y_train), cv=tscv, scoring=scorer).mean()

    with mlflow.start_run(run_name=f'{name}_Tuned'):
        print(f"Tuning {name} with Optuna...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        tuned_pipeline = clone(pipeline)
        tuned_pipeline.set_params(**best_params)
        tuned_pipeline.fit(X_train, np.log1p(y_train))
        test_rmspe = rmspe(y_test, np.expm1(tuned_pipeline.predict(X_test)))

        mlflow.log_metric("CV RMSPE", study.best_value)
        mlflow.log_metric("Test RMSPE", test_rmspe)
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(tuned_pipeline, name="model")
        
        print(f"{name} CV RMSPE: {study.best_value:.4f}")
        print(f"{name} Test RMSPE: {test_rmspe:.4f}")
        
        return test_rmspe, tuned_pipeline