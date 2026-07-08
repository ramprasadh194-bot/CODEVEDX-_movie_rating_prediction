# Movie Rating Prediction

This repository contains a portfolio-quality machine learning system for predicting IMDb-style movie ratings. It includes reproducible preprocessing, feature engineering, model selection, evaluation, and a Streamlit demo app.

## Project Structure

- `IMDb Movies India.csv` - original dataset available in the project root.
- `data/movie_dataset.csv` - working dataset copy used by the pipeline.
- `notebooks/Movie_Rating_Prediction.ipynb` - documented exploratory analysis and workflow.
- `src/` - reusable ML pipeline modules.
- `models/` - trained artifacts and serialized transformers.
- `reports/` - evaluation plots, reports, and metrics.
- `app.py` - Streamlit application for live prediction.
- `requirements.txt` - Python package dependencies.

## Features

- Safe dataset loading with encoding fallback.
- Adaptive cleaning for year, duration, votes, genre, and cast/director columns.
- Engineered features including genres, runtime buckets, director/actor encodings, and movie age.
- Multiple regression candidates with randomized hyperparameter search.
- Evaluation using MAE, MSE, RMSE, R², and adjusted R².
- Model explainability via feature importance and SHAP fallback.
- Simple prediction interface for movie metadata.
- Streamlit app for interactive use.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Optional libraries for better model performance and explanation:
>
> ```bash
> pip install xgboost lightgbm catboost shap
> ```

## Usage

### Train the model and generate reports

```bash
python run_pipeline.py
```

This command trains the best candidate model, saves artifacts under `models/`, and writes evaluation output under `reports/`.

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run from Python

```bash
python -c "from src.data_loader import load_dataset; from src.train import train_model, save_training_artifacts; df=load_dataset('data/movie_dataset.csv'); artifacts=train_model(df); save_training_artifacts(artifacts)"
```

## Dataset

The repository supports both `data/movie_dataset.csv` and `IMDb Movies India.csv` in the project root. The loader will automatically choose the available file.

## Notes

- Model artifacts are stored in `models/`.
- Evaluation plots and reports are stored in `reports/`.
- The app uses `models/best_model.pkl` and `models/feature_engineer.pkl` for prediction.

## Future Improvements

- Add notebook visualizations for SHAP explanations.
- Add dataset versioning with DVC or MLflow.
- Expand feature engineering with title/plot text embeddings.
- Deploy the Streamlit app to a cloud platform.
