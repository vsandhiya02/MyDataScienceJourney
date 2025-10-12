import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Gallstone Predictor", layout="wide")
st.title("🩺 Gallstone Prediction Tool")
st.markdown("Enter patient info and test results below to predict gallstones.")

gender_map = {"Male": 0, "Female": 1}

# Load your model (make sure your .pkl file is in the right path)
model = joblib.load("model_gallstone.pkl")

# ---- Basic Patient Info ----
st.subheader("Patient Assessment")
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1, format="%d")
with col2:
    gender = st.selectbox("Gender", options=list(gender_map.keys()))
with col3:
    height = st.number_input("Height (cm)", min_value=00.0, max_value=300.0, step=0.1, format="%.1f")
with col4:
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, step=0.1, format="%.1f")
with col5:
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")

basic_inputs = [age, gender_map[gender], height, weight, bmi]

# ---- Medical Test Results ----
st.subheader("Medical Test Results")

ordered_features = [
    "C Reactive Protein CRP",
    "Vitamin D",
    "Lean Mass LM",
    "Extracellular Fluid Total Body Water ECF TBW",
    "Aspartat Aminotransferaz AST",
    "Total Body Water TBW",
    "Bone Mass BM",
    "Extracellular Water ECW",
    "Hemoglobin HGB",
    "Body Protein Content",
    "Obesity",
    "Total Body Fat Ratio TBFR",
    "Total Fat Content TFC",
    "High Density Lipoprotein HDL",
    "Glucose",
    "Visceral Fat Area VFA",
    "Muscle Mass MM",
    "Hepatic Fat Accumulation HFA",
    "Intracellular Water ICW",
    "Visceral Fat Rating",
    "Creatinine",
    "Triglyceride",
    "Visceral Muscle Area VMA",
    "Glomerular Filtration Rate GFR",
    "Alanin Aminotransferaz ALT",
    "Low Density Lipoprotein LDL",
    "Alkaline Phosphatase ALP",
    "Total Cholesterol TC",
    "Comorbidity"
]

cols = st.columns(3)
test_inputs = []
for i, feature in enumerate(ordered_features):
    with cols[i % 3]:
        val = st.number_input(feature, step=0.01, format="%.2f", key=feature)
        test_inputs.append(val)

# ---- Predict Button ----
if st.button("🔍 Predict"):
    try:
        input_array = np.array([basic_inputs + test_inputs])
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][1]
        if prediction == 1:
            st.error(f"🔴 Gallstone Likely (Confidence: {proba:.2%})")
        else:
            st.success(f"🟢 No Gallstone Detected (Confidence: {1 - proba:.2%})")
    except Exception as e:
        st.warning("⚠️ Error during prediction. Check your inputs.")
        st.exception(e)
