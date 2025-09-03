import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 1. Inject Custom CSS ------------------
def local_css():
    st.markdown(
        """
        <style>
        /* App background */
        .stApp {
            background-color: #f9fafb; /* light gray background */
            color: #111827;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        /* Titles */
        h1, h2, h3, h4 {
            color: #2563eb !important; /* blue accent */
        }

        /* Buttons */
        div.stButton > button {
            background: #2563eb;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 15px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background: #1d4ed8;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# ------------------ 2. Data Loading ------------------
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(
            '/Users/priyankamalavade/Desktop/movie_recommendation_system/ml-latest-small/movies.csv'
        )
        ratings = pd.read_csv(
            '/Users/priyankamalavade/Desktop/movie_recommendation_system/ml-latest-small/ratings.csv'
        )
    except FileNotFoundError:
        st.error("Missing 'ratings.csv' or 'movies.csv'. Please place them in the right folder.")
        return None, None

    # Merge movies and ratings
    df = ratings.merge(movies, on="movieId").drop(columns=["timestamp"])

    # Keep only movies with at least 50 ratings
    popular_titles = df['title'].value_counts()[df['title'].value_counts() > 50].index
    df_filtered = df[df['title'].isin(popular_titles)]

    return df, df_filtered


ratings_df, filtered_df = load_data()
if ratings_df is None:
    st.stop()

# ------------------ 3. Train/Test + Similarity Matrix ------------------
train_df, _ = train_test_split(filtered_df, test_size=0.2, random_state=42)

# Use pivot_table with mean to avoid duplicate index error
user_movie_matrix = train_df.pivot_table(
    index="userId", columns="title", values="rating", aggfunc="mean"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(
    user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index
)

# ------------------ 4. Recommendation Logic ------------------
def get_recommendations_from_user(user_id, n=10):
    watched = set(filtered_df.loc[filtered_df['userId'] == user_id, 'title'])
    scores = {}

    for other_user, sim in similarity_df[user_id].drop(user_id).items():
        if sim <= 0:
            continue
        other_ratings = user_movie_matrix.loc[other_user]
        for movie, rating in other_ratings.items():
            if rating > 0 and movie not in watched:
                scores[movie] = scores.get(movie, 0) + sim * rating

    if not scores:
        return pd.DataFrame(columns=["Movie", "Predicted Score"])

    recs = pd.DataFrame(scores.items(), columns=["Movie", "Predicted Score"])
    return recs.sort_values("Predicted Score", ascending=False).head(n)


def get_recommendations_from_favorites(favorite_movies, n=10):
    if not favorite_movies:
        return pd.DataFrame(columns=["Movie", "Predicted Score"])

    temp_scores = user_movie_matrix[favorite_movies].mean(axis=1)
    similar_users = temp_scores[temp_scores > 0].index
    scores = {}

    for u in similar_users:
        sim = temp_scores.loc[u]
        other_ratings = user_movie_matrix.loc[u]
        for movie, rating in other_ratings.items():
            if rating > 0 and movie not in favorite_movies:
                scores[movie] = scores.get(movie, 0) + sim * rating

    if not scores:
        return pd.DataFrame(columns=["Movie", "Predicted Score"])

    recs = pd.DataFrame(scores.items(), columns=["Movie", "Predicted Score"])
    return recs.sort_values("Predicted Score", ascending=False).head(n)

# ------------------ 5. Streamlit UI ------------------
st.title("🎬 Movie Recommendation Engine")
st.markdown("Pick movies you like, and we'll recommend others for you.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
num_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# Tabs
tab1, tab2 = st.tabs(["🎥 Single Movie Mode", "⭐ Favorite Movies Mode"])

with tab1:
    movie_list = sorted(ratings_df['title'].unique())
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    if st.button("🎯 Get Recommendations (Single Movie)"):
        user_id = ratings_df.loc[ratings_df['title'] == selected_movie, 'userId'].iloc[0]
        recommendations = get_recommendations_from_user(user_id, n=num_recs)

        st.subheader(f"Recommendations for a user who liked **{selected_movie}**")
        if recommendations.empty:
            st.info("No recommendations found.")
        else:
            st.dataframe(
                recommendations.style.background_gradient(
                    subset=["Predicted Score"], cmap="Blues"
                ),
                use_container_width=True,
            )
            st.bar_chart(recommendations.set_index("Movie")["Predicted Score"])

with tab2:
    movie_list = sorted(user_movie_matrix.columns.unique())
    fav_movies = st.multiselect("Pick your favorite movies:", movie_list)

    if st.button("🎯 Get Recommendations (Favorites)"):
        if not fav_movies:
            st.warning("⚠️ Please select at least one movie.")
        else:
            recommendations = get_recommendations_from_favorites(fav_movies, n=num_recs)
            st.subheader("Recommendations based on your favorites")
            if recommendations.empty:
                st.info("No recommendations found. Try different favorites.")
            else:
                st.dataframe(
                    recommendations.style.background_gradient(
                        subset=["Predicted Score"], cmap="YlOrRd"
                    ),
                    use_container_width=True,
                )
                st.bar_chart(recommendations.set_index("Movie")["Predicted Score"])

# ------------------ 6. Extra Insights ------------------
st.markdown("---")
with st.expander("📊 Explore Dataset Insights"):
    st.subheader("🔥 Most Popular Movies")
    popular_movies = ratings_df['title'].value_counts().head(10)
    st.bar_chart(popular_movies)

    st.subheader("👥 Ratings per User Distribution")
    user_ratings_count = ratings_df['userId'].value_counts()
    st.line_chart(user_ratings_count)

    st.subheader("🎭 Average Rating per Movie")
    avg_movie_ratings = ratings_df.groupby("title")['rating'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_movie_ratings)
