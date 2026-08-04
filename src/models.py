from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import RANDOM_STATE

models = {
    'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
    'XGBoost': XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE),
    'LightGBM': LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE)
}

def build_pipeline(preprocessor, model):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline