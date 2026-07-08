from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler

from .preprocessing import clean_movie_dataset, get_feature_columns
from .utils import logger


@dataclass
class MovieFeatureEngineer(BaseEstimator, TransformerMixin):
    top_genres: int = 12
    top_directors: int = 40
    top_actors: int = 80

    def __post_init__(self) -> None:
        self.genre_binarizer_: MultiLabelBinarizer | None = None
        self.primary_genre_encoder_: OrdinalEncoder | None = None
        self.runtime_encoder_: OrdinalEncoder | None = None
        self.director_encoder_: OrdinalEncoder | None = None
        self.actor_encoder_: OrdinalEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.numeric_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MovieFeatureEngineer:
        df = clean_movie_dataset(X)
        self._fit_encoders(df)
        features = self._build_feature_frame(df)
        self.numeric_features_ = [col for col in features.columns if features[col].dtype.kind in "fi"]
        self.scaler_ = StandardScaler().fit(features[self.numeric_features_])
        self.feature_names_ = list(features.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise ValueError("Feature engineer must be fit before transform.")

        df = clean_movie_dataset(X)
        features = self._build_feature_frame(df)
        features[self.numeric_features_] = self.scaler_.transform(features[self.numeric_features_])
        return features

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _top_categories(self, values: pd.Series, top_k: int) -> list[str]:
        counts = values.fillna("Unknown").value_counts()
        categories = counts.head(top_k).index.tolist()
        if "Unknown" not in categories:
            categories.append("Unknown")
        return categories

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        genres = df["genre_list"].tolist()
        flat_genres = [genre for row in genres for genre in row]
        genres = pd.Series(flat_genres).value_counts().head(self.top_genres).index.tolist()
        self.genre_binarizer_ = MultiLabelBinarizer(classes=genres)
        self.genre_binarizer_.fit(df["genre_list"])

        self.primary_genre_encoder_ = OrdinalEncoder(
            categories=[self._top_categories(df["primary_genre"], self.top_genres)],
            handle_unknown="use_encoded_value",
            unknown_value=len(self._top_categories(df["primary_genre"], self.top_genres)),
        )
        self.primary_genre_encoder_.fit(df[["primary_genre"]])

        runtime_categories = ["Short", "Medium", "Long", "Epic", "Unknown"]
        self.runtime_encoder_ = OrdinalEncoder(
            categories=[runtime_categories],
            handle_unknown="use_encoded_value",
            unknown_value=len(runtime_categories),
        )
        self.runtime_encoder_.fit(df[["runtime_bucket"]])

        self.director_encoder_ = OrdinalEncoder(
            categories=[self._top_categories(df["director_name"], self.top_directors)],
            handle_unknown="use_encoded_value",
            unknown_value=len(self._top_categories(df["director_name"], self.top_directors)),
        )
        self.director_encoder_.fit(df[["director_name"]])

        actor_candidates = list(
            set(
                self._top_categories(df["actor_1"], self.top_actors)
                + self._top_categories(df["actor_2"], self.top_actors)
                + self._top_categories(df["actor_3"], self.top_actors)
            )
        )
        actor_categories = [actor_candidates, actor_candidates, actor_candidates]
        self.actor_encoder_ = OrdinalEncoder(
            categories=actor_categories,
            handle_unknown="use_encoded_value",
            unknown_value=len(actor_candidates),
        )
        self.actor_encoder_.fit(df[["actor_1", "actor_2", "actor_3"]])

    def _build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        genre_features = self.genre_binarizer_.transform(df["genre_list"])
        genre_feature_names = [f"genre_{genre}" for genre in self.genre_binarizer_.classes_]
        genre_df = pd.DataFrame(genre_features, columns=genre_feature_names, index=df.index)

        primary_genre = self.primary_genre_encoder_.transform(
            df[["primary_genre"]]
        ).astype(int).flatten()
        runtime_bucket = self.runtime_encoder_.transform(df[["runtime_bucket"]]).astype(int).flatten()
        director_encoded = self.director_encoder_.transform(df[["director_name"]]).astype(int).flatten()
        actor_encoded = self.actor_encoder_.transform(df[["actor_1", "actor_2", "actor_3"]]).astype(int)
        actor_1_encoded = actor_encoded[:, 0].flatten()
        actor_2_encoded = actor_encoded[:, 1].flatten()
        actor_3_encoded = actor_encoded[:, 2].flatten()

        encoded_df = pd.DataFrame(
            {
                "primary_genre_encoded": primary_genre,
                "runtime_bucket_encoded": runtime_bucket,
                "director_encoded": director_encoded,
                "actor_1_encoded": actor_1_encoded,
                "actor_2_encoded": actor_2_encoded,
                "actor_3_encoded": actor_3_encoded,
            },
            index=df.index,
        )

        numeric_columns, _ = get_feature_columns(df)
        numeric_df = df[numeric_columns].copy()
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)

        output = pd.concat([numeric_df, encoded_df, genre_df], axis=1)
        output.columns = [str(col) for col in output.columns]
        return output
