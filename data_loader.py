from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import logger, safe_read_csv

DATA_FILENAME = "movie_dataset.csv"


def resolve_dataset_path(base_dir: Optional[Path] = None) -> Path:
    base_dir = base_dir or Path.cwd()
    candidates = [
        base_dir / "data" / DATA_FILENAME,
        base_dir / "IMDb Movies India.csv",
        base_dir / "data" / "IMDb Movies India.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Loading dataset from %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Unable to locate dataset. Place the file in the project root or data/ folder."
    )


def load_dataset(dataset_path: Optional[str] = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else resolve_dataset_path()
    df = safe_read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns=lambda col: str(col).strip())
    return df


def save_dataset(df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    output_path = Path(output_path or (Path.cwd() / "data" / DATA_FILENAME))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path
