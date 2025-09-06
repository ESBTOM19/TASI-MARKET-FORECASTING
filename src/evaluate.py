import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score#imports necessary libraries for data manipulation, plotting and evaluation metrics

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:#function to evaluate model predictions using MSE and R²
    """
    Computes key evaluation metrics.

    Parameters:
        y_true (pd.Series): Actual target values
        y_pred (pd.Series): Predicted target values

    Returns:
        dict: Dictionary with MSE and R²
    """
    mse = mean_squared_error(y_true, y_pred)#calculates the mean squared error between the true and predicted values
    r2 = r2_score(y_true, y_pred)#calculates the statistical measure of how close the data are to the fitted prediction lines
    return {"MSE": round(mse, 4), "R²": round(r2, 4)}#returns a dictionary containing the MSE and R² values rounded to 4 decimal places


def plot_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plots true vs predicted target values.

    Parameters:
        y_true (pd.Series): Actual target values
        y_pred (pd.Series): Model predictions
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", color="green", linewidth=2)
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
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true.index, residuals, alpha=0.6, color="red", edgecolor="k")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residual Distribution")
    plt.xlabel("Index")
    plt.ylabel("Residuals(Actual-Predicted)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()