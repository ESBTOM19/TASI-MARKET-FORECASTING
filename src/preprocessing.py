import pandas as pd#pandas is a data manipulation and analysis library
from pathlib import Path#pathlib handles file paths in a platform-independent way
from features import add_lagged_features, add_rolling_stats, drop_feature_na#imports custom feature functions

def load_ticker(path: str) -> pd.DataFrame:#location of the ticker data file and reurns a dataframe
    """
    Loads a single ticker's data from CSV, ensuring datetime format.
    """
    df = pd.read_csv(path, parse_dates=["Date"])#reads the CSV file and parses the 'Date' column as datetime
    df = df.sort_values("Date").reset_index(drop=True)#sorts the dataframe by 'Date' and resets the index
    return df#returns the cleaned dataframe whrere 'Date' is the index and sorted in ascending order


def standardize_change_pct_column(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Renames the % change column to the expected format for feature functions.
    Handles cases like 'Change %', 'Change_pct', etc.
    """
    expected_col = f"{ticker}_Change_pct"
    raw_candidates = ["Change %", "Change_pct", "% Change"]

    for col in df.columns:
        if col in raw_candidates:
            df.rename(columns={col: expected_col}, inplace=True)
            break

    if expected_col not in df.columns:
        raise KeyError(f"Expected column '{expected_col}' not found after standardization.")
    
    return df#returns the dataframe with the standardied change percentage column


def preprocess_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:#pipeline to preprocess a single ticker's data
    """
    Applies lagged and rolling features, then drops NaNs.
    """
    df = standardize_change_pct_column(df, ticker)
    df = add_lagged_features(df, ticker)
    df = add_rolling_stats(df, ticker)
    df = drop_feature_na(df, ticker)
    return df


def align_on_date(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merges multiple DataFrames on common Date column.
    """
    base = dfs[0][["Date"]].copy()
    for df in dfs:
        base = base.merge(df, on="Date", how="inner")#merges each dataframe on the dates column, keeping dates present in all dataframes
    return base#returns the merged dataframe containing only rows with dates present in all input dataframes


def run_preprocessing() -> pd.DataFrame:#main function to runt the whole preprocessing pipeline
    """
    Loads, processes, and aligns all ticker datasets.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]#gets the root directory of the project

    path_Aramco     = PROJECT_ROOT / "data" / "Aramco.csv"#path to Aramco data file
    path_Al_Rajhi   = PROJECT_ROOT / "data" / "Al_Rajhi.csv"
    path_tasi       = PROJECT_ROOT / "data" / "TASI_cleaned.csv"

    df_Aramco   = preprocess_ticker(load_ticker(path_Aramco), "Aramco")#loads and preprocesses the Aramco data
    df_Al_Rajhi = preprocess_ticker(load_ticker(path_Al_Rajhi), "Al_Rajhi")
    df_tasi     = preprocess_ticker(load_ticker(path_tasi), "TASI")

    df_final = align_on_date(df_Aramco, df_Al_Rajhi, df_tasi)#align the three dataframes on their common dates
    return df_final#returns the final preprocessed and alligned dataframe


if __name__ == "__main__":
    df = run_preprocessing()
    output_path = Path("data/model_input/merged_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Features saved to {output_path}")