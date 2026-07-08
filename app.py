import streamlit as st

from src.predict import predict_movie_rating
from src.utils import load_pickle

MODEL_PATH = "models/best_model.pkl"
FEATURE_ENGINEER_PATH = "models/feature_engineer.pkl"


def load_model_artifacts():
    try:
        model = load_pickle(MODEL_PATH)
        feature_engineer = load_pickle(FEATURE_ENGINEER_PATH)
        return model, feature_engineer
    except FileNotFoundError as exc:
        st.error(f"Model artifacts are missing: {exc}")
        st.stop()
        return None, None


def render_header() -> None:
    st.set_page_config(
        page_title="Movie Rating Predictor",
        page_icon="🎬",
        layout="wide",
    )
    st.title("Movie Rating Prediction System")
    st.markdown(
        "Predict IMDb-style movie ratings using a production-ready regression pipeline."
    )


def build_form() -> dict[str, str]:
    with st.form(key="movie_input_form"):
        st.markdown("## Simple movie input")
        st.write("Enter only the most essential movie details and the system will fill missing cast or crew values automatically.")
        name = st.text_input("Movie Title", "New Movie")
        year = st.text_input("Release Year", "2023")
        duration = st.text_input("Duration", "120 min")
        genre = st.text_input("Genre", "Drama, Romance")
        votes = st.text_input("Votes", "1000")

        with st.expander("Advanced inputs (optional)"):
            director = st.text_input("Director", "Unknown")
            actor_1 = st.text_input("Actor 1", "Unknown")
            actor_2 = st.text_input("Actor 2", "Unknown")
            actor_3 = st.text_input("Actor 3", "Unknown")

        submit = st.form_submit_button("Predict Rating")
        return {
            "name": name,
            "year": year,
            "duration": duration,
            "genre": genre,
            "votes": votes,
            "director": director if director else "Unknown",
            "actor_1": actor_1 if actor_1 else "Unknown",
            "actor_2": actor_2 if actor_2 else "Unknown",
            "actor_3": actor_3 if actor_3 else "Unknown",
            "submit": submit,
        }


def render_prediction(result):
    st.subheader("Prediction Result")
    st.metric("Predicted Rating", f"{result.predicted_rating:.2f} / 10")
    st.metric("Confidence Score", f"{result.confidence_score:.2f}")
    st.markdown("**Prediction details**")
    if result.predicted_rating >= 8.0:
        st.success("This title is likely to be rated very highly by viewers.")
    elif result.predicted_rating >= 6.0:
        st.info("This title is likely to receive a solid rating in the mid-range.")
    else:
        st.warning("This title is predicted to score below the strong rating threshold.")
    st.markdown("**Top feature contributions**")
    for feature, value in result.explanation.items():
        st.write(f"- **{feature}**: {value:.3f}")


def main() -> None:
    render_header()
    model, _ = load_model_artifacts()
    user_input = build_form()

    if user_input["submit"]:
        movie_info = {k: v for k, v in user_input.items() if k != "submit"}
        result = predict_movie_rating(movie_info)
        render_prediction(result)
        st.write("---")
        st.markdown(
            "This app uses a feature-engineered regression model trained on IMDb-style movie data."
        )


if __name__ == "__main__":
    main()
