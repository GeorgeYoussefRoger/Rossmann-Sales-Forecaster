import os
import pandas as pd

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the train and store data from a CSV file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data files not found in: {data_path}")
    
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    store = pd.read_csv(os.path.join(data_path, 'store.csv'))
    
    return pd.merge(train, store, on='Store', how='left')