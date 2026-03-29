from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2

MONTH_MAP = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
    7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'
}

MODELS = {
    'Ridge': Ridge(),
    'XGBoost': XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE),
    'LightGBM': LGBMRegressor(n_jobs=-1, random_state=RANDOM_STATE)
}