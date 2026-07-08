from pathlib import Path

from src.pipeline import run_full_pipeline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    artifacts = run_full_pipeline(
        dataset_path=str(project_root / "data" / "movie_dataset.csv"),
        model_dir=str(project_root / "models"),
        report_dir=str(project_root / "reports"),
    )
    print("Pipeline complete. Artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
