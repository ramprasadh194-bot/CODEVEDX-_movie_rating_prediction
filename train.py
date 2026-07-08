import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from .feature_engineering import MovieFeatureEngineer
from .utils import ensure_dir, logger, save_json, save_pickle


def _build_candidate_models() -> dict[str, Any]:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        logger.warning("XGBoost is not installed; skipping XGBRegressor.")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(random_state=42, n_jobs=-1)
    except ImportError:
        logger.warning("LightGBM is not installed; skipping LGBMRegressor.")

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(random_state=42, verbose=0)
    except ImportError:
        logger.warning("CatBoost is not installed; skipping CatBoostRegressor.")

    return models


def _get_model_param_distributions() -> dict[str, dict[str, list[Any]]]:
    return {
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 50.0],
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 5.0],
        },
        "DecisionTree": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "XGBoost": {
            "n_estimators": [100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "LightGBM": {
            "n_estimators": [100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 70],
        },
        "CatBoost": {
            "iterations": [100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8],
        },
    }


def _resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    candidates = [
        target_column,
        target_column.lower(),
        target_column.upper(),
        target_column.capitalize(),
        target_column.replace("_", " ").title(),
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Target column '{target_column}' does not exist in dataset.")


def train_model(
    df: pd.DataFrame,
    target_column: str = "rating",
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 18,
) -> dict[str, Any]:
    df = df.copy()
    resolved_target = _resolve_target_column(df, target_column)

    df = df.dropna(subset=[resolved_target])
    y = df[resolved_target].astype(float)
    X = df.drop(columns=[resolved_target])

    feature_engineer = MovieFeatureEngineer()
    X_transformed = feature_engineer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state
    )

    model_candidates = _build_candidate_models()
    param_distributions = _get_model_param_distributions()
    results: list[dict[str, Any]] = []

    for name, estimator in model_candidates.items():
        params = param_distributions.get(name)
        if params is None or len(params) == 0:
            estimator.fit(X_train, y_train)
            best_estimator = estimator
            logger.info("Trained baseline model %s", name)
        else:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                n_iter=min(n_iter, len(params) * 3),
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                random_state=random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            logger.info(
                "Best %s parameters: %s", name, search.best_params_
            )

        cv_scores = cross_val_score(
            best_estimator,
            X_train,
            y_train,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        results.append(
            {
                "model_name": name,
                "estimator": best_estimator,
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
            }
        )

    best = sorted(results, key=lambda item: item["cv_r2_mean"], reverse=True)[0]
    best_model = best["estimator"]
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    residuals = y_test - predictions
    test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    training_results = {
        "best_model_name": best["model_name"],
        "best_model": best_model,
        "feature_engineer": feature_engineer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv_results": results,
        "residuals": residuals,
        "test_rmse": test_rmse,
    }
    return training_results


def save_training_artifacts(
    artifacts: dict[str, Any],
    output_dir: str | Path = "models",
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    model_path = output_dir / "best_model.pkl"
    feature_engineer_path = output_dir / "feature_engineer.pkl"
    training_path = output_dir / "training_metadata.json"

    save_pickle(model_path, artifacts["best_model"])
    save_pickle(feature_engineer_path, artifacts["feature_engineer"])
    save_json(
        training_path,
        {
            "best_model_name": artifacts["best_model_name"],
            "cv_results": [
                {
                    "model_name": r["model_name"],
                    "cv_r2_mean": r["cv_r2_mean"],
                    "cv_r2_std": r["cv_r2_std"],
                }
                for r in artifacts["cv_results"]
            ],
            "test_rmse": float(artifacts.get("test_rmse", 0.0)),
        },
    )

    logger.info("Saved best model to %s", model_path)
    logger.info("Saved feature engineer to %s", feature_engineer_path)
    logger.info("Saved training metadata to %s", training_path)
    return {
        "model_path": model_path,
        "feature_engineer_path": feature_engineer_path,
        "training_metadata_path": training_path,
    }
