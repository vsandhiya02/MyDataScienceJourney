import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset once at start and cache it
@st.cache_data
def load_data():
    df = pd.read_csv('career.csv')
    # Combine relevant columns for vectorizing
    df['combined_text'] = df['Required_Skills'].fillna('') + ' ' + df['Related_Interests'].fillna('')
    return df

df = load_data()

# Prepare TF-IDF vectorizer and vectors; cache for performance
@st.cache_data
def prepare_vectors(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data['combined_text'])
    return vectorizer, vectors

vectorizer, career_vectors = prepare_vectors(df)

def recommend_careers(user_education, user_interests, top_n=5):
    user_text = user_education + ' ' + user_interests
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, career_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    recommendations = []
    for idx in top_indices:
        career = df.iloc[idx]
        recommendations.append({
            'Career_Title': career['Career_Title'],
            'Required_Skills': career['Required_Skills'],
            'Career_Path': career['Typical_Progression'],
            'Certifications/Courses': career['Certifications_Courses'],
            'Similarity_Score': similarities[idx]
        })
    return recommendations

# Streamlit UI
st.title("Career Recommendation System")

education = st.text_input("Enter your education background (e.g., biotechnology student):")
interests = st.text_input("Enter your interests (e.g., information technology, coding):")
top_n = st.slider("Number of recommendations to show:", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    if not education.strip() or not interests.strip():
        st.warning("Please enter both your education and interests.")
    else:
        results = recommend_careers(education.lower(), interests.lower(), top_n=top_n)

        st.subheader("Recommended Careers:")
        for res in results:
            st.markdown(f"**Career Title:** {res['Career_Title']}")
            st.markdown(f"**Required Skills:** {res['Required_Skills']}")
            st.markdown(f"**Career Path:** {res['Career_Path']}")
            st.markdown(f"**Certifications/Courses:** {res['Certifications/Courses']}")
            st.markdown(f"**Match Score:** {res['Similarity_Score']:.2f}")
            st.markdown("---")
