# main.py

# Importing libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import warnings
from typing import Any

warnings.filterwarnings('ignore')

# Create FastAPI object
app = FastAPI()

# Load models and scaler once at startup
model = joblib.load("student_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping cluster number to label
cluster_info = {
    0: {"name": "Struggling & Unmotivated"},
    1: {"name": "Highly Motivated Achievers"},
    2: {"name": "Balanced Performers"},
    3: {"name": "Extracurricular Focused"},
}

# Input schema using Pydantic
class StudentFeatures(BaseModel):
    Hours_Studied: int
    Attendance: int
    Parental_Involvement: int
    Access_to_Resources: int
    Extracurricular_Activities: int
    Sleep_Hours: int
    Previous_Scores: int
    Motivation_Level: int
    Internet_Access: int
    Tutoring_Sessions: int
    Family_Income: int
    Teacher_Quality: int
    School_Type: int
    Peer_Influence: int
    Physical_Activity: int
    Learning_Disabilities: int
    Parental_Education_Level: int
    Distance_from_Home: int
    Gender: int

# Define the behavior features used for clustering
X_behavior_features = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Sleep_Hours', 'Motivation_Level', 'Tutoring_Sessions', 'Physical_Activity', 'Internet_Access']

# Output schema
class Output(BaseModel):
    predicted_score: float
    cluster: int
    cluster_label: str

def generate_recommendation(row):
    rec = []

    # Academic Effort
    # Note: Assuming encoded values based on your notebook. Adjust thresholds if needed.
    if row['Hours_Studied'] < 2:
        rec.append("Try to dedicate at least 2–3 hours a day to focused study. Short, regular sessions are better than cramming.")
    if row['Previous_Scores'] < 50:
        rec.append("Focus on foundational topics where you struggled before. Use revision guides or get help from a teacher.")
    if row['Predicted_Exam_Score'] < 50:
        rec.append("Consider a study plan or tutor support to boost exam performance.")
    if row['Tutoring_Sessions'] < 1:
        rec.append("You may benefit from attending tutoring sessions to clarify difficult topics.")

    # Attendance
    if row['Attendance'] < 75:
        rec.append("Try to attend classes regularly to avoid missing important lessons and falling behind.")

    # Health & Well-being
    if row['Sleep_Hours'] < 6:
        rec.append("Getting at least 7–8 hours of sleep helps improve memory, focus, and overall well-being.")
    if row['Physical_Activity'] < 2:
        rec.append("Add some physical activity like walking, yoga, or sports 2–3 times a week to improve energy and reduce stress.")

    # Motivation & Social Environment
    # Assuming Motivation_Level mapping: 0=Low, 1=Medium, 2=High
    if row['Motivation_Level'] == 0:
        rec.append("Set small, achievable goals each week to build momentum and increase motivation.")
    # Assuming Peer_Influence mapping: 0=Negative, 1=Neutral, 2=Positive
    if row['Peer_Influence'] == 0:
         rec.append("Surround yourself with classmates who encourage and support your academic goals.")


    # Parental & Family Support
    # Assuming Parental_Involvement mapping: 0=Low, 1=Medium, 2=High
    if row['Parental_Involvement'] == 0:
        rec.append("More parental involvement — like reviewing homework or attending school meetings — can positively impact progress.")
    # Assuming Parental_Education_Level mapping: 0=High School, 1=College, 2=Postgraduate
    if row['Parental_Education_Level'] == 0:
        rec.append("Parents can support learning through school-provided materials, even without a formal education background.")
    # Assuming Family_Income is encoded numerically, adjust threshold if needed
    if row['Family_Income'] == 0: # Assuming 0 maps to 'Low'
        rec.append("Explore financial aid or school assistance programs to reduce stress related to resources.")


    # Learning Support
    if row['Learning_Disabilities'] == 0: # Assuming 0 maps to 'Yes'
        rec.append("You may benefit from individualized learning plans or accommodations — speak with a school counselor.")

    # Infrastructure & Access
    if row['Internet_Access'] == 1: # Assuming 1 maps to 'No'
        rec.append("Reliable internet access is essential. Ask your school about free or subsidized internet programs.")
    # Assuming Access_to_Resources mapping: 0=Low, 1=Medium, 2=High
    if row['Access_to_Resources'] == 0:
        rec.append("Make use of school libraries, labs, and online educational platforms for extra learning support.")
    # Assuming Distance_from_Home mapping: 0=Near, 1=Moderate, 2=Far
    if row['Distance_from_Home'] == 2:
        rec.append("Try to use commuting time for light revision like flashcards or audiobooks if travel is long.")

    # Balance with Extracurriculars
    if row['Extracurricular_Activities'] == 0: # Assuming 0 maps to 'Yes'
        rec.append("Make sure extracurriculars don't overwhelm study time — balance is key to success.")

    # Quality of Education
    # Assuming Teacher_Quality mapping: 0=Low, 1=Medium, 2=High
    if row['Teacher_Quality'] == 0:
        rec.append("If teaching support is limited, look for trusted online content or peer study groups to reinforce learning.")
    # Assuming School_Type mapping: 0=Public, 1=Private
    if row['School_Type'] == 0:
        rec.append("Make the most of online tools and free educational apps to supplement school learning.")


    # Default positive message
    if not rec:
        rec.append("You're on the right track! Keep up the consistent effort and continue building good habits.")

    return rec

@app.post("/predict/")
def predict_exam_score_and_recommendations(features: StudentFeatures):
    input_df = pd.DataFrame([features.dict()])

    # Scale the behavior features for clustering
    X_behavior_scaled = scaler.transform(input_df[X_behavior_features])

    # Predict the cluster
    predicted_cluster = kmeans.predict(X_behavior_scaled)[0]
    cluster_name = cluster_info.get(predicted_cluster, {}).get("name", "Unknown Cluster")

    # Add the cluster to the input data for exam score prediction
    input_df['Cluster'] = predicted_cluster

    # Predict the exam score using the best model (XGBoost)
    predicted_exam_score = model.predict(input_df)[0]

    # Add the predicted exam score to the input_df for recommendation generation
    input_df.loc[0, 'Predicted_Exam_Score'] = predicted_exam_score

    # Generate personalized recommendations
    recommendations = generate_recommendation(input_df.iloc[0])

    return {
        "predicted_exam_score": float(predicted_exam_score),
        "cluster_name": cluster_name,
        "recommendations": recommendations
    }