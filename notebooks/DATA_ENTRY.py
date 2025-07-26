import yfinance as yf
import pandas as pd
import os

#create data directory if it doesnt exist
os.makedirs("data", exist_ok=True)

#List of tickers to download
tickers=