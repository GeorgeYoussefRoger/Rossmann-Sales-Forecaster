import pandas as pd

def preprocess(df):
    """
    Cleaning for the Rossmann Store Sales dataset:
    - Remove closed stores
    - Convert the 'Date' column to datetime format
    - Handle missing values
    - Convert the filled numeric columns to integer
    """
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