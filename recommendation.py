import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from pathlib import Path


# Load data from CSV files using pathlib for portability

data_dir = Path("C:/Users/tarun/Desktop/movie-recommendation-system/dataset")
ratings_file = data_dir / "ratings.csv"
movies_file = data_dir / "movies.csv"

ratings_df = pd.read_csv(ratings_file)
movies_df = pd.read_csv(movies_file)

# Combine ratings with movie titles (inner join on movieId)
merged_data = pd.merge(ratings_df, movies_df, on="movieId")


# Keep only movies that have received at least 20 ratings
# (reduces noise and speeds up computation)

ratings_per_movie = merged_data.groupby("movieId")["rating"].count()
popular_movie_ids = ratings_per_movie[ratings_per_movie >= 20].index

# Subset the data to these popular movies
filtered_data = merged_data[merged_data["movieId"].isin(popular_movie_ids)]


# Build a user‑item matrix (rows = users, columns = movies)
# Missing values are filled with 0 (assume no rating = 0)

user_item_matrix = filtered_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)


# Mean‑center each user's ratings to remove user bias
# (a user who rates everything 4 should not be considered similar
#  to one who rates everything 5 if their relative preferences align)

user_averages = user_item_matrix.mean(axis=1)
centered_matrix = user_item_matrix.sub(user_averages, axis=0).fillna(0)


# Compute cosine similarity between movies (item‑based)
# We transpose the matrix so that columns (movies) become rows.

item_similarity = cosine_similarity(centered_matrix.T)

# Convert the numpy array into a pandas DataFrame for easy lookup
similarity_frame = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)


# Prepare a mapping from movieId to title and genres for all movies,
# not only the popular ones. This will be used for searching.

movie_metadata = movies_df.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

# Create a list of (movieId, title) pairs for fast searching
movie_catalog = [(row["movieId"], row["title"]) for _, row in movies_df.iterrows()]


# Search function: given a user query, return up to 10 matching movies
# It uses three strategies in order: exact match, substring, fuzzy.

def search_movie(query):
    """
    Parameters
    ----------
    query : str
        The movie name entered by the user.

    Returns
    -------
    list of (movieId, title)
        A list of matching movies (max 10).
    """
    query_lower = query.lower().strip()

    # 1. Exact match (case‑insensitive)
    exact = [(mid, title) for mid, title in movie_catalog if title.lower() == query_lower]
    if exact:
        return exact[:10]

    # 2. Substring match
    substring = [(mid, title) for mid, title in movie_catalog if query_lower in title.lower()]
    if substring:
        return substring[:10]

    # 3. Fuzzy match using difflib
    all_titles = [title for _, title in movie_catalog]
    fuzzy_titles = difflib.get_close_matches(query, all_titles, n=10, cutoff=0.6)
    # Map back to movieId (assume titles are unique enough)
    fuzzy_results = []
    for ftitle in fuzzy_titles:
        # Find the first movieId that has this title
        movie_id = next(mid for mid, title in movie_catalog if title == ftitle)
        fuzzy_results.append((movie_id, ftitle))
    return fuzzy_results


# Recommendation function: returns top_k most similar movies

def recommend_similar(movie_id, top_k=5):
    """
    Parameters
    ----------
    movie_id : int
        The movieId of the selected movie.
    top_k : int
        Number of recommendations to return.

    Returns
    -------
    list of (movieId, similarity_score)
        The top_k most similar movies (excluding the input movie).
    """
    if movie_id not in similarity_frame.index:
        return []   # movie not in the popular set → no recommendations

    # Get similarity scores for the given movie
    scores = similarity_frame[movie_id].values

    # Efficiently find indices of the top scores (including the movie itself)
    # We use argpartition for O(n) complexity instead of full sort.
    top_indices = np.argpartition(scores, -(top_k + 1))[-(top_k + 1):]

    # Retrieve the corresponding movieIds
    candidate_ids = similarity_frame.index[top_indices]

    # Build list of (movieId, score) excluding the input movie
    candidates = []
    for mid in candidate_ids:
        if mid != movie_id:
            candidates.append((mid, similarity_frame.loc[movie_id, mid]))

    # Sort by descending similarity score and return top_k
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


# Main interaction loop

if __name__ == "__main__":
    print("=" * 50)
    print("      Welcome to the Movie Recommender System")
    print("=" * 50)
    print("You can enter any movie name (or part of it).")
    print("Type 'quit' to exit.\n")

    while True:
        user_query = input("Enter a movie name: ").strip()
        if user_query.lower() == "quit":
            print("Goodbye!")
            break

        if not user_query:
            print("Please type something.\n")
            continue

        # Find movies matching the query
        matches = search_movie(user_query)
        if not matches:
            print("Sorry, no movies found. Try a different spelling.\n")
            continue

        # Show the list of possible movies
        print("\nMovies matching your input:")
        for idx, (mid, title) in enumerate(matches, start=1):
            print(f"  {idx}. {title}")

        # Let the user pick one
        try:
            selection = int(input("\nSelect a movie by number: "))
            if selection < 1 or selection > len(matches):
                print("Invalid number. Please choose from the list.\n")
                continue
        except ValueError:
            print("Please enter a valid number.\n")
            continue

        chosen_id, chosen_title = matches[selection - 1]

        # Display the chosen movie's genre
        genre_info = movie_metadata[chosen_id]["genres"]
        print(f"\nYou selected: {chosen_title}  [Genre: {genre_info}]")

        # Check if the movie is in the similarity matrix (i.e., has enough ratings)
        if chosen_id not in similarity_frame.index:
            print("This movie does not have enough ratings (>=20) to generate recommendations.")
            print("Please choose another movie.\n")
            continue

        # Get and show recommendations
        recommendations = recommend_similar(chosen_id, top_k=5)
        if not recommendations:
            print("No recommendations available for this movie.\n")
        else:
            print("\nTop 5 similar movies:")
            for rec_id, sim_score in recommendations:
                rec_title = movie_metadata[rec_id]["title"]
                print(f"  • {rec_title}  (similarity: {sim_score:.3f})")
        print()   # blank line for readability