
Forecasting fluctuations of the Tadawul All Share Index(TASI), the primary stock market index of Saudi Arabia
# TASI Market Forecasting with Random Forest  
**Forecasting fluctuations of the Tadawul All Share Index (TASI), the primary stock market of Saudi Arabia, using Aramco & Al Rajhi Bank stock data.**  

---
## Financial Literacy & Time-Series Awareness

This project applies key financial data analysis concepts:

- **Time-Series Forecasting**: Stock prices are sequential, so lagged features (1, 3, 5 days) were engineered to capture momentum and autocorrelation.
- **Volatility Tracking**: Rolling standard deviation over 5 days was used as a measure of market volatility.
- **Trend Indicators**: Rolling mean over 5 days captured short-term market trends.
- **Market Data Structure**: OHLCV (Open, High, Low, Close, Volume) and % change were cleaned and aligned per ticker for reliable modeling.
- **Avoiding Lookahead Bias**: Training/testing was split chronologically, with the last 20% reserved for testing future predictions.


## Project Workflow  

### 1. Data Entry  
- Imported data from **Yahoo Finance** (`yfinance`).  
- Cleaned and aligned the following columns:  
  - **OHLCV** (Open, High, Low, Close, Volume)  
  - **CHANGE%**  
  - **TICKER**  
- Merged all cleaned data into a single DataFrame: `all_merged`.  

---

### 2. Feature Engineering  
- **Lagged Features**  
  - Added lag features to track the past **1, 3, and 5 days** of stock returns.  
  - These features capture short-term memory of market behavior.  

- **Rolling Statistics**  
  - Added **5-day rolling mean** (trend feature).  
  - Added **5-day rolling standard deviation** (volatility feature).  

- **Handling Missing Data**  
  - Dropped rows containing `NaN` values introduced by lag and rolling features.  
  - Ensured a clean dataset for modeling.  

---

### 3. Preprocessing  
- Loaded and cleaned the feature set.  
- Ensured alignment across all tickers with common columns and consistent rows.  
- Removed rows with missing values to avoid overfitting.  
- Exported a **final preprocessed CSV** to be fed directly into `TRAIN.py`.  

---

### 4. Training Model  
- Added `src/` directory to Python’s import path for modularity.  
- Imported custom modules:  
  - **`evaluate.py`** → Evaluation functions & plotting utilities.  
  - **`features.py`** → Functions for feature generation.  
  - **`preprocessing.py`** → Pipeline for preparing features for the model.  

- Workflow:  
  1. Split dataset into **training (80%)** and **testing (20%)**.  
  2. Trained a **Random Forest Regressor** on the training set.  
  3. Evaluated predictions using the test set.  
  4. Visualized results with plots from `evaluate.py`.  

---

### 5. Evaluation  
- Compared **Actual vs Predicted** values of TASI index and decides which one influences the TASI market more than the other.  
- Plotted performance metrics and prediction graphs.  
- Key outputs:  
  - Forecast accuracy plots.  
  - Feature importance visualization (impact of Aramco vs Al Rajhi).  

---

## Tech Stack  
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, yfinance  
- **Model**: Random Forest Regressor  
- **Structure**: Modular (with `src/` for preprocessing, features, evaluation)  

---

## How to Run the Project  

### Clone Repository  
```bash
git clone https://github.com/ESBTOM19/TASI-MARKET-FORECASTING.git
cd TASI-MARKET-FORECASTING
