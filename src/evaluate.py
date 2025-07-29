import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Computes key evaluation metrics.

    Parameters:
        y_true (pd.Series): Actual target values
        y_pred (pd.Series): Predicted target values

    Returns:
        dict: Dictionary with MSE and R²
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": round(mse, 4), "R²": round(r2, 4)}


def plot_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plots true vs predicted target values.

    Parameters:
        y_true (pd.Series): Actual target values
        y_pred (pd.Series): Model predictions
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", linewidth=2)
    plt.plot(y_true.index, y_pred, label="Predicted", linestyle="--", color="blue")
    plt.title("Actual vs Predicted TASI Change %")
    plt.xlabel("Index")
    plt.ylabel("Change %")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plots distribution of residuals.

    Parameters:
        y_true (pd.Series): Actual target values
        y_pred (pd.Series): Predicted target values
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=30, kde=True, color="skyblue")
    plt.title("Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()