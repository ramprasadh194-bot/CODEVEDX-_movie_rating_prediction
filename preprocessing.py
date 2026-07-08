import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import logger


def _clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def _parse_year(value: Any) -> float:
    text = _clean_string(value)
    if not text:
        return np.nan
    match = re.search(r"(19\d{2}|20\d{2})", text)
    return float(match.group(0)) if match else np.nan


def _parse_duration(value: Any) -> float:
    text = _clean_string(value)
    if not text:
        return np.nan
    match = re.search(r"(\d{1,3})", text)
    return float(match.group(1)) if match else np.nan


def _parse_votes(value: Any) -> float:
    text = _clean_string(value)
    if not text:
        return np.nan
    cleaned = text.replace(",", "").replace(" ", "")
    return float(cleaned) if cleaned.isdigit() else np.nan


def _split_genres(value: Any) -> list[str]:
    text = _clean_string(value)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[,;/&]", text) if part.strip()]
    return parts


def _extract_primary_genre(value: Any) -> str:
    values = _split_genres(value)
    return values[0] if values else "Unknown"


def _bucket_runtime(minutes: float) -> str:
    if np.isnan(minutes):
        return "Unknown"
    minutes = float(minutes)
    if minutes <= 80:
        return "Short"
    if minutes <= 110:
        return "Medium"
    if minutes <= 150:
        return "Long"
    return "Epic"


def _clean_name(value: Any) -> str:
    text = _clean_string(value)
    return text if text else "Unknown"


def clean_movie_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    if "name" not in df.columns:
        logger.warning("Expected a Name column in the dataset.")
        df["name"] = "Unknown"

    df["name"] = df["name"].astype(str).apply(_clean_name)
    df["release_year"] = df.get("year", pd.Series(dtype="object")).apply(_parse_year)
    df["duration_minutes"] = df.get("duration", pd.Series(dtype="object")).apply(_parse_duration)
    df["vote_count"] = df.get("votes", pd.Series(dtype="object")).apply(_parse_votes)
    df["genre_list"] = df.get("genre", pd.Series(dtype="object")).apply(_split_genres)
    df["primary_genre"] = df.get("genre", pd.Series(dtype="object")).apply(_extract_primary_genre)
    df["num_genres"] = df["genre_list"].apply(len)
    current_year = datetime.now().year
    df["movie_age"] = df["release_year"].apply(
        lambda year: current_year - year if not np.isnan(year) else np.nan
    )
    df["movie_age"] = df["movie_age"].fillna(df["movie_age"].median())
    df["runtime_bucket"] = df["duration_minutes"].apply(_bucket_runtime)
    df["title_length"] = df["name"].astype(str).apply(lambda value: len(value))
    df["title_words"] = df["name"].astype(str).apply(lambda value: len(value.split()))

    for column in ["director", "actor_1", "actor_2", "actor_3"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = df[column].apply(lambda value: _clean_string(value) if not pd.isna(value) else "")
        df[column] = df[column].astype(object).replace("", np.nan)

    df["director_name"] = df["director"].fillna("Unknown")
    df["actor_1"] = df["actor_1"].fillna("Unknown")
    df["actor_2"] = df["actor_2"].fillna("Unknown")
    df["actor_3"] = df["actor_3"].fillna("Unknown")

    df["director_count"] = df.groupby("director_name")["director_name"].transform("count")
    df["actor_1_count"] = df.groupby("actor_1")["actor_1"].transform("count")
    df["actor_2_count"] = df.groupby("actor_2")["actor_2"].transform("count")
    df["actor_3_count"] = df.groupby("actor_3")["actor_3"].transform("count")

    total_rows = len(df)
    df["director_popularity"] = df["director_count"] / total_rows
    df["actor_popularity_mean"] = (
        df[["actor_1_count", "actor_2_count", "actor_3_count"]].replace(0, np.nan).mean(axis=1)
        / total_rows
    ).fillna(0)
    df["actor_popularity_max"] = (
        df[["actor_1_count", "actor_2_count", "actor_3_count"]].replace(0, np.nan).max(axis=1)
        / total_rows
    ).fillna(0)
    df["cast_size"] = df[["actor_1", "actor_2", "actor_3"]].apply(
        lambda row: sum(1 for value in row if str(value).strip().lower() not in {"", "unknown", "nan"}),
        axis=1,
    )
    df["has_votes"] = df["vote_count"].notna().astype(int)
    df["year_missing"] = df["release_year"].isna().astype(int)
    df["duration_missing"] = df["duration_minutes"].isna().astype(int)
    df["vote_count_missing"] = df["vote_count"].isna().astype(int)
    df["release_decade"] = df["release_year"].apply(
        lambda year: int(year // 10 * 10) if not np.isnan(year) else -1
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    df = df.copy()
    numeric = [
        "duration_minutes",
        "vote_count",
        "movie_age",
        "num_genres",
        "director_popularity",
        "actor_popularity_mean",
        "actor_popularity_max",
        "cast_size",
        "title_length",
        "title_words",
        "release_decade",
        "has_votes",
        "year_missing",
        "duration_missing",
        "vote_count_missing",
    ]
    categorical = ["primary_genre", "runtime_bucket", "director_name", "actor_1", "actor_2", "actor_3"]
    numeric = [col for col in numeric if col in df.columns]
    categorical = [col for col in categorical if col in df.columns]
    return numeric, categorical
