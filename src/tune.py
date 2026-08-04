import numpy as np
import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer

from src.evaluate import rmspe

def tune(X_train, X_test, y_train, y_test, model, name):
    def objective(trial):
        if name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.05, 0.95)
            }
        elif name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        elif name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }

        model.set_params(**params)
        tscv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(rmspe, greater_is_better=False)
        return -cross_val_score(model, X_train, np.log1p(y_train), cv=tscv, scoring=scorer).mean()

    with mlflow.start_run(run_name=f'{name}_tuned'):
        print(f"Tuning {name}...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        tuned_model = model.set_params(**best_params)
        tuned_model.fit(X_train, np.log1p(y_train))
        test_rmspe = rmspe(y_test, np.expm1(tuned_model.predict(X_test)))

        mlflow.log_metrics({
            "CV RMSPE": study.best_value,
            "Test RMSPE": test_rmspe
        })

        mlflow.log_params(best_params)
        print(f"Best parameters for {name}: {best_params}")
        print(f"{name} CV RMSPE: {study.best_value:.4f}")
        print(f"{name} Test RMSPE: {test_rmspe:.4f}")
        
        return tuned_model