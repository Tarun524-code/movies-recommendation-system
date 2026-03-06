import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from pathlib import Path


# Page configuration

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.markdown("Find movies similar to your favourites using collaborative filtering.")


# Load and cache data

@st.cache_data
def load_data():
    data_dir = Path("dataset")
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

@st.cache_data
def build_similarity_matrix(ratings, movies, min_ratings=20):
    # Merge ratings with movies
    data = pd.merge(ratings, movies, on="movieId")

    # Keep only movies with at least min_ratings
    rating_counts = data.groupby("movieId")["rating"].count()
    popular_movies = rating_counts[rating_counts >= min_ratings].index
    popular_data = data[data["movieId"].isin(popular_movies)]

    # Build user‑item matrix
    user_movie_matrix = popular_data.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # Mean‑center ratings
    user_means = user_movie_matrix.mean(axis=1)
    centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

    # Compute cosine similarity between movies
    movie_sim = cosine_similarity(centered.T)
    sim_df = pd.DataFrame(
        movie_sim,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    return sim_df, popular_movies

@st.cache_data
def get_movie_catalog(movies):
    # Return a list of (movieId, title) for all movies
    return [(row["movieId"], row["title"]) for _, row in movies.iterrows()]

@st.cache_data
def get_movie_metadata(movies):
    return movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")


# Load everything

ratings, movies = load_data()
sim_df, popular_movies = build_similarity_matrix(ratings, movies)
movie_catalog = get_movie_catalog(movies)
movie_metadata = get_movie_metadata(movies)


# Search function

def search_movie(query):
    query_lower = query.lower().strip()
    # 1. Exact match
    exact = [(mid, title) for mid, title in movie_catalog if title.lower() == query_lower]
    if exact:
        return exact[:10]
    # 2. Substring
    substr = [(mid, title) for mid, title in movie_catalog if query_lower in title.lower()]
    if substr:
        return substr[:10]
    # 3. Fuzzy
    titles_only = [title for _, title in movie_catalog]
    fuzzy_titles = difflib.get_close_matches(query, titles_only, n=10, cutoff=0.6)
    fuzzy = []
    for ftitle in fuzzy_titles:
        mid = next(mid for mid, title in movie_catalog if title == ftitle)
        fuzzy.append((mid, ftitle))
    return fuzzy


# Recommendation function

def recommend(movie_id, top_k=5):
    if movie_id not in sim_df.index:
        return []
    scores = sim_df[movie_id].values
    # Efficient top‑k (including the movie itself)
    top_indices = np.argpartition(scores, -(top_k + 1))[-(top_k + 1):]
    candidate_ids = sim_df.index[top_indices]
    results = []
    for mid in candidate_ids:
        if mid != movie_id:
            results.append((mid, sim_df.loc[movie_id, mid]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# Streamlit UI

user_input = st.text_input("🔍 Enter a movie name (e.g., 'Toy Story'):", "")

if user_input:
    matches = search_movie(user_input)
    if not matches:
        st.warning("No matching movies found. Try a different spelling.")
    else:
        # Create a list of display strings for the selectbox
        display_options = [f"{title} (ID: {mid})" for mid, title in matches]
        selected_display = st.selectbox("Select the correct movie:", display_options)

        # Extract the selected movieId
        selected_index = display_options.index(selected_display)
        selected_mid, selected_title = matches[selected_index]

        # Show movie info
        genres = movie_metadata[selected_mid]["genres"]
        st.markdown(f"**You selected:** {selected_title}")
        st.caption(f"Genres: {genres}")

        # Check if movie is in popular set
        if selected_mid not in sim_df.index:
            st.error("This movie does not have enough ratings (≥20) to generate recommendations. Try another one.")
        else:
            # Get recommendations
            recs = recommend(selected_mid, top_k=5)
            if recs:
                st.subheader("🎥 Top 5 similar movies")
                for i, (mid, score) in enumerate(recs, 1):
                    rec_title = movie_metadata[mid]["title"]
                    rec_genres = movie_metadata[mid]["genres"]
                    st.write(f"**{i}.** {rec_title}")
                    st.caption(f"   Genres: {rec_genres}  |  Similarity: {score:.3f}")
            else:
                st.info("No recommendations available.")