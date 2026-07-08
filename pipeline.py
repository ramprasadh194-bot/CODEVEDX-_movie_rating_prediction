from pathlib import Path
from typing import Optional

from .data_loader import load_dataset
from .evaluate import (
    compute_metrics,
    compute_permutation_importance,
    plot_error_distribution,
    plot_feature_importance,
    plot_prediction_vs_actual,
    plot_residuals,
    save_evaluation_report,
)
from .train import save_training_artifacts, train_model
from .utils import ensure_dir, logger


def run_full_pipeline(
    dataset_path: Optional[str] = None,
    model_dir: str = "models",
    report_dir: str = "reports",
) -> dict[str, Path]:
    df = load_dataset(dataset_path) if dataset_path else load_dataset()
    logger.info("Starting model training pipeline.")
    artifacts = train_model(df)
    saved = save_training_artifacts(artifacts, output_dir=model_dir)

    evaluation_dir = ensure_dir(report_dir)
    y_test = artifacts["y_test"]
    y_pred = artifacts["best_model"].predict(artifacts["X_test"])
    metrics = compute_metrics(y_test, y_pred, artifacts["X_test"].shape[1])

    plot_residuals(y_test, y_pred, evaluation_dir / "residual_plot.png")
    plot_prediction_vs_actual(y_test, y_pred, evaluation_dir / "prediction_vs_actual.png")
    plot_error_distribution(y_test, y_pred, evaluation_dir / "error_distribution.png")
    plot_feature_importance(
        artifacts["best_model"],
        artifacts["feature_engineer"].feature_names_,
        evaluation_dir / "feature_importance.png",
    )
    compute_permutation_importance(
        artifacts["best_model"],
        artifacts["X_test"],
        y_test,
        evaluation_dir / "permutation_importance.csv",
    )
    save_evaluation_report(
        metrics,
        evaluation_dir / "evaluation_report.txt",
        {
            "Residual Plot": evaluation_dir / "residual_plot.png",
            "Prediction vs Actual": evaluation_dir / "prediction_vs_actual.png",
            "Error Distribution": evaluation_dir / "error_distribution.png",
            "Feature Importance": evaluation_dir / "feature_importance.png",
        },
    )
    logger.info("Pipeline completed. Metrics: %s", metrics)
    return {
        **saved,
        "evaluation_report": evaluation_dir / "evaluation_report.txt",
        "metrics_json": evaluation_dir / "evaluation_report.json",
    }
