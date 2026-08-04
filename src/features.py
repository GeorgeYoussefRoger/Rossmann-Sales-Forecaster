from src.config import MONTH_MAP

def add_date_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    return df

def add_promo_features(df):
    df['IsPromo2Month'] = [
        1 if str(pi) != '0' and MONTH_MAP[m] in str(pi) else 0 
        for m, pi in zip(df['Month'], df['PromoInterval'])
    ]

    return df

def add_lag_rolling_features(df):
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["sales_lag_14"] = df.groupby("Store")["Sales"].shift(14)
    df["rolling_mean_7"] = df.groupby("Store")["Sales"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_mean_30"] = df.groupby("Store")["Sales"].transform(lambda x: x.shift(1).rolling(30).mean())
    df.dropna(inplace=True)

    return df