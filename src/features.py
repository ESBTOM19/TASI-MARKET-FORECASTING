import pandas as pd#loads the pandas library for data manpulation

def add_lagged_features(df: pd.DataFrame, ticker: str, lags: list[int] = [1, 3, 5]) -> pd.DataFrame:
    """
    Adds lagged %change features for a given ticker.
    Example: Aramco_Change_pct_lag_1, Al_Rajhi_Change_pct_lag_3

    Parameters:
        df (pd.DataFrame): DataFrame with Change % column
        ticker (str): Ticker name for column prefix
        lags (list[int]): Number of days to lag

    Returns:
        pd.DataFrame: Original df with new lagged features added
    """
    col_name = f"{ticker}_Change_pct"#creates the column name based on the ticker
    for lag in lags:#iterates through each value in the lags list(1,3,5) 
        df[f"{col_name}_lag_{lag}"] = df[col_name].shift(lag)#creates a new column for each lag value
    return df


def add_rolling_stats(df: pd.DataFrame, ticker: str, window: int = 5) -> pd.DataFrame:
    """
    Adds rolling volatility and trend features for a given ticker.
    Example: Aramco_Volatility_5, Al_Rajhi_Trend_5

    Parameters:
        df (pd.DataFrame): DataFrame with Change % column
        ticker (str): Ticker name for column prefix
        window (int): Rolling window size in days

    Returns:
        pd.DataFrame: df with rolling std and mean columns added
    """
    col_name = f"{ticker}_Change_pct"
    df[f"{ticker}_Volatility_{window}"] = df[col_name].rolling(window).std()#calculates the rolling standard deviation over a specified window
    df[f"{ticker}_Trend_{window}"] = df[col_name].rolling(window).mean()#calculate the rolling mean over a specified window
    return df


def drop_feature_na(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Drops rows where lagged or rolling features are NaN for a given ticker.

    Parameters:
        df (pd.DataFrame): Feature-enriched DataFrame
        ticker (str): Ticker to filter feature columns

    Returns:
        pd.DataFrame: Cleaned DataFrame with valid feature rows
    """
    feature_cols = [
        col for col in df.columns
        if ticker in col and ("lag_" in col or "Volatility" in col or "Trend" in col)
    ]
    df_cleaned = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df_cleaned#returns the cleaned dataframe with rows containing NAN values of the specified feature columns dropped