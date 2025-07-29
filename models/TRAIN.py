import sys
from pathlib import Path

# Add 'src/' to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import evaluate_predictions, plot_actual_vs_pred, plot_residuals

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the merged feature dataset.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def define_target(df: pd.DataFrame, ticker: str = "TASI") -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates features from target Change % for the given ticker.
    """
    target = f"{ticker}_Change_pct"
    feature_cols = [
        col for col in df.columns
        if "Change_pct" in col and ticker not in col
    ]
    X = df[feature_cols]
    y = df[target]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Splits data, trains Random Forest, and evaluates performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Prints evaluation metrics for model performance.
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
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
    try:
        from evaluate import evaluate_predictions, plot_actual_vs_pred, plot_residuals

        metrics = evaluate_predictions(y_test, y_pred)
        print("Detailed Evaluation:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        plot_actual_vs_pred(y_test, y_pred)
        plot_residuals(y_test, y_pred)

    except ImportError:
        print("Extended evaluation skipped — 'evaluate' not found.")

    print("Training & evaluation complete.")