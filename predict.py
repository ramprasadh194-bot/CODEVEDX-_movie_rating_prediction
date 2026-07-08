from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from .data_loader import load_dataset
from .feature_engineering import MovieFeatureEngineer
from .preprocessing import clean_movie_dataset
from .utils import load_json, load_pickle, logger


@dataclass
class PredictionResult:
    predicted_rating: float
    confidence_score: float
    explanation: dict[str, float]


def load_artifacts(
    model_path: str | Path,
    feature_engineer_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[RegressorMixin, MovieFeatureEngineer, dict[str, Any]]:
    model = load_pickle(model_path)
    feature_engineer = load_pickle(feature_engineer_path)
    metadata = {}
    if metadata_path is not None:
        try:
            metadata = load_json(metadata_path)
        except FileNotFoundError:
            metadata = {}
    return model, feature_engineer, metadata


def _build_input_frame(movie_info: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([movie_info])
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return clean_movie_dataset(df)


def _calculate_confidence(model: Any, X: pd.DataFrame, training_rmse: float | None = None) -> float:
    if hasattr(model, "predict") and training_rmse is not None:
        predictions = model.predict(X)
        if len(predictions) > 0:
            return float(max(0.0, min(1.0, 1 - training_rmse / (np.std(predictions) + 1e-6))))
    return 0.75


def _feature_contributions(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    base_prediction = None
    contributions: dict[str, float] = {}
    try:
        import shap

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        contributions = {
            fname: float(shap_values.values[0, idx])
            for idx, fname in enumerate(feature_names)
        }
        return contributions
    except Exception:
        if hasattr(model, "feature_importances_"):
            importance = np.array(model.feature_importances_).flatten()
            contributions = {
                fname: float(value)
                for fname, value in zip(feature_names, importance)
            }
        elif hasattr(model, "coef_"):
            importance = np.abs(np.array(model.coef_).flatten())
            contributions = {
                fname: float(value)
                for fname, value in zip(feature_names, importance)
            }
        else:
            contributions = {fname: 0.0 for fname in feature_names}
    return contributions


def predict_movie_rating(
    movie_info: dict[str, Any],
    model_path: str | Path = "models/best_model.pkl",
    feature_engineer_path: str | Path = "models/feature_engineer.pkl",
    metadata_path: str | Path = "models/training_metadata.json",
    training_rmse: float | None = None,
) -> PredictionResult:
    model, feature_engineer, metadata = load_artifacts(
        model_path,
        feature_engineer_path,
        metadata_path,
    )
    cleaned = _build_input_frame(movie_info)
    features = feature_engineer.transform(cleaned)
    predicted_rating = float(model.predict(features)[0])
    if training_rmse is None:
        training_rmse = float(metadata.get("test_rmse", 0.0)) or None
    confidence = _calculate_confidence(model, features, training_rmse)
    contributions = _feature_contributions(model, features, feature_engineer.feature_names_)
    sorted_contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10])
    return PredictionResult(
        predicted_rating=predicted_rating,
        confidence_score=confidence,
        explanation=sorted_contributions,
    )


def predict_from_dataset_row(row_index: int) -> PredictionResult:
    df = load_dataset()
    if row_index < 0 or row_index >= len(df):
        raise IndexError("Row index is out of range.")
    row = df.iloc[row_index : row_index + 1].to_dict(orient="records")[0]
    return predict_movie_rating(row)
