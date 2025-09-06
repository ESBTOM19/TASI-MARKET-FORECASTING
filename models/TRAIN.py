import sys
from pathlib import Path

# Add 'src/' to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))#allows importing from the src directory

from evaluate import evaluate_predictions, plot_actual_vs_pred, plot_residuals

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(path: str) -> pd.DataFrame:#function to load the merged features dataset
    """
    Loads the merged feature dataset.
    """
    df = pd.read_csv(path, parse_dates=["Date"])#reads the CSV file() and parses the 'Date' column as datetime
    return df


def define_target(df: pd.DataFrame, ticker: str = "TASI") -> tuple[pd.DataFrame, pd.Series]:#function to define the target variable and feature set
    """
    Separates features from target Change % for the given ticker.
    """
    target = f"{ticker}_Change_pct"
    Aramco_col="Aramco_Change_pct"
    Al_Rajhi_col="Al_Rajhi_Change_pct"
    
    feature_cols=[Aramco_col, Al_Rajhi_col]
    X=df[feature_cols].rename(columns={Aramco_col:"Aramco", Al_Rajhi_col:"Al_Rajhi"})
    y=df[target].rename("TASI")
    return X, y#returns the feature dataframe and target series


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:#function to train the random forest model and make predictions
    """
    Splits data, trains Random Forest, and evaluates performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )#splits the data into training and testing sets, using the last 20% of the data for testing without shuffling date order

    model = RandomForestRegressor(n_estimators=100, random_state=42)#initializes the random forest regressor with 100 trees and a fixed random state for reproductibility
    model.fit(X_train, y_train)#trains the model on the data X_train being the features and y_train being the target variable

    y_pred = model.predict(X_test)#makes predictions from the trained model using the test features
    return model, X_test, y_test, y_pred#returns the trained model, test features, true test target values, and predictions


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> None:#function to evaluate the model's performance using basic metrics like MSE and R²
    """
    Prints evaluation metrics for model performance.
    """
    mse = mean_squared_error(y_test, y_pred)#THIS calculates the mean squared error between the true and predicted values
    r2 = r2_score(y_test, y_pred)#this calculates R², a statistical measure of how close the data are to the fitted prediction lines
    print(f"Basic Evaluation:\nMSE: {mse:.4f} | R²: {r2:.4f}")


if __name__ == "__main__":
    # Load feature dataset
    df = load_dataset("data/model_input/merged_features.csv")

    # Define predictors and target
    X, y = define_target(df)

    # Train model and predict
    model, X_test, y_test, y_pred = train_model(X, y)

    # Print core metrics
    evaluate_model(y_test, y_pred)

    # Extended evaluation
    try:#tries to import the evaluate module for extended evaluation and plotting
        from evaluate import evaluate_predictions, plot_actual_vs_pred, plot_residuals

        metrics = evaluate_predictions(y_test, y_pred)#evaluates the predictions using additional metrics
        print("Detailed Evaluation:")
        for k, v in metrics.items():#iterates through the metrics dictionary and prints each metric name and value formatted to 4 decimal places
            print(f"{k}: {v:.4f}")

        plot_actual_vs_pred(y_test, y_pred)#generates a plot comparing actual and predicted values
        plot_residuals(y_test, y_pred)#generates a plot of the residuals (differences between actual and predicted values)

    except ImportError:
        print("Extended evaluation skipped — 'evaluate' not found.")

    print("Training & evaluation complete.")