import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("movie_rating_prediction")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = configure_logger()


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found at {path_obj}")

    errors: list[str] = []
    candidate_encodings = [None, "utf-8", "latin1", "ISO-8859-1", "cp1252"]

    for encoding in candidate_encodings:
        try:
            return pd.read_csv(path_obj, encoding=encoding, **kwargs)
        except Exception as exc:
            errors.append(f"{encoding}:{exc}")

    raise ValueError(
        f"Could not read CSV file at {path_obj}. Tried encodings: {candidate_encodings}."
        f" Errors: {errors}"
    )


def save_json(path: str | Path, data: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_pickle(path: str | Path, obj: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    joblib.dump(obj, path_obj)


def load_pickle(path: str | Path) -> Any:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Pickle file not found at {path_obj}")
    return joblib.load(path_obj)
