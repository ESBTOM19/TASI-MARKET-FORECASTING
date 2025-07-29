import yfinance as yf
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# List tickers for Aramco & Al Rajhi
tickers = ["2222.SR", "1120.SR"]#2222.SR is Aramco, 1120.SR is Al Rajhi Bank

# Download OHLCV data
df = yf.download(tickers, start="2020-01-01", end="2024-12-31", group_by="ticker")

# Process each ticker and export separately
for ticker in tickers:
    try:
        ohlcv = df[ticker][["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        ohlcv["Change %"] = ohlcv["Close"].pct_change() * 100
        ohlcv["Ticker"] = ticker
        ohlcv.reset_index(inplace=True)

        name = "Aramco" if ticker == "2222.SR" else "Al_Rajhi"
        ohlcv.to_csv(f"data/{name}.csv", index=False)
        print(f"Saved: data/{name}.csv")

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

# Convert numeric suffixes (e.g. 12M → 12,000,000)
def parse_numeric(value):
    try:
        if isinstance(value, str):
            value = value.strip().upper()
            if value.endswith("B"):
                return float(value[:-1]) * 1_000_000_000
            elif value.endswith("M"):
                return float(value[:-1]) * 1_000_000
            elif value.endswith("K"):
                return float(value[:-1]) * 1_000
        return float(value)
    except Exception as e:
        print(f"Could not parse value: {value}. Error: {e}")
        return None

# Locate raw TASI CSV
def locate_tasi_file(filename="Tadawul All Share Historical Data.csv"):
    local_path = os.path.join("data", filename)
    downloads_path = os.path.expanduser(os.path.join("~", "Downloads", filename))
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(downloads_path):
        return downloads_path
    else:
        return None

# Load & clean TASI file
tasi_csv_path = locate_tasi_file()
try:
    if not tasi_csv_path:
        raise FileNotFoundError("CSV not found in 'data/' or Downloads.")

    tasi_df = pd.read_csv(tasi_csv_path, parse_dates=["Date"])
    
    #Rename 'Price' to 'Close' for compatibility with other tickers
    if "Price" in tasi_df.columns and "Close" not in tasi_df.columns:
     tasi_df.rename(columns={"Price": "Close"}, inplace=True)



    # Clean numeric columns
    for col in ["Close", "Open", "High", "Low"]:
        if col in tasi_df.columns:
            tasi_df[col] = tasi_df[col].astype(str).str.replace(",", "", regex=False).astype(float)

    # Parse volume
    vol_col = "Vol." if "Vol." in tasi_df.columns else "Volume"
    if vol_col in tasi_df.columns:
        tasi_df[vol_col] = tasi_df[vol_col].apply(parse_numeric)
        tasi_df.rename(columns={vol_col: "Volume"}, inplace=True)

    # Parse change %
    if "Change %" in tasi_df.columns:
        tasi_df["Change %"] = (
            tasi_df["Change %"].astype(str).str.replace("%", "", regex=False).astype(float)
        )

    # Set index and save cleaned TASI
    tasi_df.set_index("Date", inplace=True)
    tasi_df.sort_index(inplace=True)
    cleaned_path = "data/TASI_cleaned.csv"
    tasi_df.to_csv(cleaned_path)
    print(f"Saved: {cleaned_path}")
    print("TASI columns:", tasi_df.columns.tolist())

except Exception as e:
    print(f"TASI cleaning failed: {e}")

# Merge Aramco + Al Rajhi + TASI
try:
    Aramco_df = pd.read_csv("data/Aramco.csv", parse_dates=["Date"])
    Al_Rajhi_df = pd.read_csv("data/Al_Rajhi.csv", parse_dates=["Date"])
    tasi_df = pd.read_csv("data/TASI_cleaned.csv", parse_dates=["Date"])
    tasi_df["Ticker"] = "TASI"

    # Align columns and merge
    all_frames = [Aramco_df, Al_Rajhi_df, tasi_df]
    common_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Change %", "Ticker"]
    merged_df = pd.concat([df[common_cols] for df in all_frames], axis=0)
    merged_df.sort_values("Date", inplace=True)

    # Save final merged file
    merged_df.to_csv("data/all_merged.csv", index=False)
    print("Saved: data/all_merged.csv")

except Exception as e:
    print(f"Merge failed: {e}")