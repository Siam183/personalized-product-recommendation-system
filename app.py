import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models and dataset
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
product_profiles = joblib.load("models/product_profiles.pkl")
df = pd.read_csv("data/amazon_dataset.csv")

# Map product IDs to index
product_id_to_index = pd.Series(df.index, index=df['Uniq Id']).drop_duplicates()

def recommend_by_demo_and_content(age, gender, top_n=5):
    # Fake users for demo
    users_df = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(1, 101)],
        'age': np.random.randint(16, 60, size=100),
        'gender': np.random.choice(['Male', 'Female'], size=100)
    })

    interactions = []
    for user in users_df['user_id']:
        product_indices = np.random.choice(df.index, size=5, replace=False)
        for idx in product_indices:
            interactions.append({'user_id': user, 'product_id': df.at[idx, 'Uniq Id']})
    interactions_df = pd.DataFrame(interactions)

    # Find similar users
    age_range = (age - 5, age + 5)
    similar_users = users_df[(users_df['gender'] == gender) &
                             (users_df['age'].between(age_range[0], age_range[1]))]['user_id'].tolist()

    if not similar_users:
        return pd.DataFrame()

    relevant_interactions = interactions_df[interactions_df['user_id'].isin(similar_users)]
    product_ids = relevant_interactions['product_id'].unique()

    indices = [product_id_to_index.get(pid) for pid in product_ids if product_id_to_index.get(pid) is not None]
    if not indices:
        return pd.DataFrame()

    demo_profile_vector = product_profiles[indices].mean(axis=0)
    scores = cosine_similarity(demo_profile_vector, product_profiles).flatten()
    df['score'] = scores

    recommendations = df[~df['Uniq Id'].isin(product_ids)].sort_values(by='score', ascending=False).head(top_n)
    return recommendations[['Product Name', 'Category', 'Selling Price', 'score']]

# Streamlit UI
st.title("🛒 Personalized Product Recommendation System")
age = st.number_input("Enter your age:", min_value=16, max_value=80, value=25)
gender = st.selectbox("Select your gender:", ["Male", "Female"])
top_n = st.slider("Number of recommendations:", 1, 20, 5)

if st.button("Recommend"):
    results = recommend_by_demo_and_content(age, gender, top_n)
    if results.empty:
        st.warning("No recommendations found.")
    else:
        st.write(results)
