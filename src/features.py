import pandas as pd

def add_lagged_features(df: pd.DataFrame, ticker: str, lags: list[int] = [1, 3, 5]) -> pd.DataFrame:
    """
    Adds lagged % change features for a given ticker.
    Example: Aramco_Change_pct_lag_1, Al_Rajhi_Change_pct_lag_3

    Parameters:
        df (pd.DataFrame): DataFrame with Change % column
        ticker (str): Ticker name for column prefix
        lags (list[int]): Number of days to lag

    Returns:
        pd.DataFrame: Original df with new lagged features added
    """
    col_name = f"{ticker}_Change_pct"
    for lag in lags:
        df[f"{col_name}_lag_{lag}"] = df[col_name].shift(lag)
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
    df[f"{ticker}_Volatility_{window}"] = df[col_name].rolling(window).std()
    df[f"{ticker}_Trend_{window}"] = df[col_name].rolling(window).mean()
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
    return df_cleaned