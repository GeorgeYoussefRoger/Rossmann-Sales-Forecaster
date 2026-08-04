import numpy as np
import mlflow

from src.evaluate import rmspe

def train(X_train, y_train, X_test, y_test, model, name):
    with mlflow.start_run(run_name=f'{name}'):
        print(f"Training {name}...") 
        model.fit(X_train, np.log1p(y_train))  

        score = rmspe(y_test, np.expm1(model.predict(X_test)))
        mlflow.log_metric('Test RMSPE', score)
        
        print(f"{name} Test RMSPE: {score:.4f}")
        return score