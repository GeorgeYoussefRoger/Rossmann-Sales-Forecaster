import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import DATA_DIR

def preprocess():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    store = pd.read_csv(os.path.join(DATA_DIR, 'store.csv'))
    df = pd.merge(train, store, on='Store', how='left')

    df = df[df['Open'] == 1]
    df['Date'] = pd.to_datetime(df['Date'])

    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)

    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0).astype(int)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)
    df['PromoInterval'] = df['PromoInterval'].fillna('0')
    
    return df

def create_preprocessor():
    return ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['a', 'b', 'c']]), ['Assortment']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['StateHoliday', 'StoreType', 'DayOfWeek'])
    ], remainder='passthrough')