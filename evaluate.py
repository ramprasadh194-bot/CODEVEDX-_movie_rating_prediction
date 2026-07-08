import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import ensure_dir, logger, save_json

sns.set(style="whitegrid", palette="muted")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, n_features: int) -> dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "adjusted_r2": float(adjusted_r2),
    }


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Rating")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Ratings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logger.info("Saved residual plot to %s", output_path)


def plot_prediction_vs_actual(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Predicted vs Actual Rating")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logger.info("Saved prediction vs actual plot to %s", output_path)


def plot_error_distribution(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Prediction Error")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logger.info("Saved error distribution plot to %s", output_path)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.array(model.coef_))
    else:
        logger.warning("Unable to extract feature importance from model type %s", type(model))

    if importance is None:
        return

    importance = importance.flatten()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    ensure_dir(output_path.parent)
    results = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": results.importances_mean,
            "importance_std": results.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_path, index=False)
    logger.info("Saved permutation importance to %s", output_path)
    return importance_df


def save_evaluation_report(
    metrics: dict[str, float],
    report_path: Path,
    plots: dict[str, Path],
) -> None:
    ensure_dir(report_path.parent)
    lines = ["Model evaluation report", "======================", ""]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}")
    lines.append("")
    for title, path in plots.items():
        lines.append(f"{title}: {path}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    save_json(report_path.with_suffix(".json"), metrics)
    logger.info("Saved evaluation report to %s", report_path)
