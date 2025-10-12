# app.py

import streamlit as st
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load models and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("student_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, kmeans, scaler

model, kmeans, scaler = load_assets()

# Define cluster names
cluster_info = {
    0: "Highly Motivated Achievers",
    1: "Balanced Performers",
    2: "Struggling & Unmotivated",
    3: "Extracurricular Focused"
}

# Feature names used for clustering
X_behavior_features = [
    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
    'Extracurricular_Activities', 'Sleep_Hours', 'Motivation_Level',
    'Tutoring_Sessions', 'Physical_Activity', 'Internet_Access'
]

# Encoding maps
encoding_maps = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Extracurricular_Activities": {"No": 0, "Yes": 1},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Internet_Access": {"No": 0, "Yes": 1},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "School_Type": {"Public": 0, "Private": 1},
    "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
    "Learning_Disabilities": {"Yes": 0, "No": 1},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
    "Gender": {"Male": 0, "Female": 1},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2}
}

# Recommendation logic
def generate_recommendation(row):
    rec = []

    # Hours studied
    if row['Hours_Studied'] < 2:
        rec.append("Try to dedicate at least 2–3 hours a day to focused study.")
    
    # Scores
    if row['Previous_Scores'] < 50:
        rec.append("Focus on foundational topics where you struggled before.")
    if row['Predicted_Exam_Score'] < 50:
        rec.append("Consider a study plan or tutor support to boost exam performance.")
    
    # Tutoring
    if row['Tutoring_Sessions'] < 1:
        rec.append("Attend tutoring sessions to clarify difficult topics.")
    
    # Attendance
    if row['Attendance'] < 75:
        rec.append("Attend classes regularly to avoid missing lessons.")
    
    # Sleep & physical activity
    if row['Sleep_Hours'] < 6:
        rec.append("Aim for 7–8 hours of sleep to improve memory and focus.")
    if row['Physical_Activity'] < 2:
        rec.append("Add physical activity 2–3 times a week for better focus.")
    
    # Motivation_Level: 0 = Low
    if row['Motivation_Level'] == 0:
        rec.append("Set small, achievable goals each week.")
    
    # Peer_Influence: 0 = Negative
    if row['Peer_Influence'] == 0:
        rec.append("Surround yourself with classmates who encourage your growth.")
    
    # Parental_Involvement: 0 = Low
    if row['Parental_Involvement'] == 0:
        rec.append("Encourage more parental involvement in your learning.")
    
    # Parental_Education_Level: 0 = High School
    if row['Parental_Education_Level'] == 0:
        rec.append("Leverage available school resources for extra support.")
    
    # Family_Income: 0 = Low
    if row['Family_Income'] == 0:
        rec.append("Explore school aid programs for low-income families.")
    
    # Learning_Disabilities: 0 = Yes
    if row['Learning_Disabilities'] == 0:
        rec.append("Ask for individualized learning plans if needed.")
    
    # Internet_Access: 1 = Yes
    if row['Internet_Access'] == 1:
        rec.append("Ensure stable internet for effective online learning.")
    
    # Access_to_Resources: 0 = Low
    if row['Access_to_Resources'] == 0:
        rec.append("Use school libraries, labs, and online tools.")
    
    # Distance_from_Home: 2 = Far
    if row['Distance_from_Home'] == 2:
        rec.append("Use commuting time for light study (audiobooks, flashcards).")
    
    # Extracurricular_Activities: 0 = No
    if row['Extracurricular_Activities'] == 0:
        rec.append("Balance extracurriculars and study time carefully.")
    
    # Teacher_Quality: 0 = Low
    if row['Teacher_Quality'] == 0:
        rec.append("Use trusted online content or study groups for extra help.")
    
    # School_Type: 0 = Public
    if row['School_Type'] == 0:
        rec.append("Use free educational apps to supplement learning.")
    
    if not rec:
        rec.append("You're on the right track! Keep up the good work.")
    
    return rec

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Student Performance Predictor")

with st.form("student_form"):
    st.subheader("Enter Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        Hours_Studied = st.number_input("Hours Studied", 0, 100, 10)
        Attendance = st.slider("Attendance (%)", 0, 100, 90)
        Parental_Involvement = st.selectbox("Parental Involvement", encoding_maps["Parental_Involvement"].keys())
        Access_to_Resources = st.selectbox("Access to Resources", encoding_maps["Access_to_Resources"].keys())
        Extracurricular_Activities = st.selectbox("Extracurricular Activities", encoding_maps["Extracurricular_Activities"].keys())
        Sleep_Hours = st.slider("Sleep Hours", 0, 12, 8)

    with col2:
        Previous_Scores = st.slider("Previous Scores", 0, 100, 90)
        Motivation_Level = st.selectbox("Motivation Level", encoding_maps["Motivation_Level"].keys())
        Internet_Access = st.selectbox("Internet Access", encoding_maps["Internet_Access"].keys())
        Tutoring_Sessions = st.slider("Tutoring Sessions", 0, 10, 2)
        Family_Income = st.selectbox("Family Income", encoding_maps["Family_Income"].keys())
        Teacher_Quality = st.selectbox("Teacher Quality", encoding_maps["Teacher_Quality"].keys())

    with col3:
        School_Type = st.selectbox("School Type", encoding_maps["School_Type"].keys())
        Peer_Influence = st.selectbox("Peer Influence", encoding_maps["Peer_Influence"].keys())
        Physical_Activity = st.slider("Physical Activity (hrs/week)", 0, 20, 3)
        Learning_Disabilities = st.selectbox("Learning Disabilities", encoding_maps["Learning_Disabilities"].keys())
        Parental_Education_Level = st.selectbox("Parental Education Level", encoding_maps["Parental_Education_Level"].keys())
        Distance_from_Home = st.selectbox("Distance from Home", encoding_maps["Distance_from_Home"].keys())
        Gender = st.selectbox("Gender", encoding_maps["Gender"].keys())

    submitted = st.form_submit_button("📊 Predict")

if submitted:
    # Encode inputs
    data = {
        "Hours_Studied": Hours_Studied,
        "Attendance": Attendance,
        "Parental_Involvement": encoding_maps["Parental_Involvement"][Parental_Involvement],
        "Access_to_Resources": encoding_maps["Access_to_Resources"][Access_to_Resources],
        "Extracurricular_Activities": encoding_maps["Extracurricular_Activities"][Extracurricular_Activities],
        "Sleep_Hours": Sleep_Hours,
        "Previous_Scores": Previous_Scores,
        "Motivation_Level": encoding_maps["Motivation_Level"][Motivation_Level],
        "Internet_Access": encoding_maps["Internet_Access"][Internet_Access],
        "Tutoring_Sessions": Tutoring_Sessions,
        "Family_Income": encoding_maps["Family_Income"][Family_Income],
        "Teacher_Quality": encoding_maps["Teacher_Quality"][Teacher_Quality],
        "School_Type": encoding_maps["School_Type"][School_Type],
        "Peer_Influence": encoding_maps["Peer_Influence"][Peer_Influence],
        "Physical_Activity": Physical_Activity,
        "Learning_Disabilities": encoding_maps["Learning_Disabilities"][Learning_Disabilities],
        "Parental_Education_Level": encoding_maps["Parental_Education_Level"][Parental_Education_Level],
        "Distance_from_Home": encoding_maps["Distance_from_Home"][Distance_from_Home],
        "Gender": encoding_maps["Gender"][Gender],
    }

    input_df = pd.DataFrame([data])

    # Predict cluster
    X_scaled = scaler.transform(input_df[X_behavior_features])
    cluster = kmeans.predict(X_scaled)[0]
    input_df['Cluster'] = cluster
    cluster_label = cluster_info.get(cluster, "Unknown")

    # Predict exam score
    predicted_score = model.predict(input_df)[0]
    input_df['Predicted_Exam_Score'] = predicted_score

    # Generate recommendations
    recommendations = generate_recommendation(input_df.iloc[0])

    # Display results
    st.success(f"🎯 Predicted Exam Score: {round(predicted_score, 2)}")
    st.info(f"🧠 Student Cluster: {cluster_label}")

    st.subheader("✅ Personalized Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")
