# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model = joblib.load("model_gallstone.pkl")  # Update the path if needed

app = FastAPI(title="Gallstone Prediction")

# Input schema using Pydantic
class PatientData(BaseModel):
    Age: int
    Gender: int
    Comorbidity: int
    Coronary_Artery_Disease_CAD: int
    Hypothyroidism: int
    Hyperlipidemia: int
    Diabetes_Mellitus_DM: int
    Height: float
    Weight: float
    Body_Mass_Index_BMI: float
    Total_Body_Water_TBW: float
    Extracellular_Water_ECW: float
    Intracellular_Water_ICW: float
    Extracellular_Fluid_Total_Body_Water_ECF_TBW: float
    Total_Body_Fat_Ratio_TBFR: float
    Lean_Mass_LM: float
    Body_Protein_Content: float
    Visceral_Fat_Rating: int
    Bone_Mass_BM: float
    Muscle_Mass_MM: float
    Obesity: float
    Total_Fat_Content_TFC: float
    Visceral_Fat_Area_VFA: float
    Visceral_Muscle_Area_VMA: float
    Hepatic_Fat_Accumulation_HFA: int
    Glucose: int
    Total_Cholesterol_TC: int
    Low_Density_Lipoprotein_LDL: int
    High_Density_Lipoprotein_HDL: int
    Triglyceride: int
    Aspartat_Aminotransferaz_AST: int
    Alanin_Aminotransferaz_ALT: int
    Alkaline_Phosphatase_ALP: int
    Creatinine: float
    Glomerular_Filtration_Rate_GFR: float
    C_Reactive_Protein_CRP: float
    Hemoglobin_HGB: float
    Vitamin_D: float

#to pass the output
class Output(BaseModel):
    gallstone_status: int

import numpy as np

@app.post("/predict")
def predict(data: PatientData):
    X_input = np.array([[
        data.Age,
        data.Gender,
        data.Comorbidity,
        data.Coronary_Artery_Disease_CAD,
        data.Hypothyroidism,
        data.Hyperlipidemia,
        data.Diabetes_Mellitus_DM,
        data.Height,
        data.Weight,
        data.Body_Mass_Index_BMI,
        data.Total_Body_Water_TBW,
        data.Extracellular_Water_ECW,
        data.Intracellular_Water_ICW,
        data.Extracellular_Fluid_Total_Body_Water_ECF_TBW,
        data.Total_Body_Fat_Ratio_TBFR,
        data.Lean_Mass_LM,
        data.Body_Protein_Content,
        data.Visceral_Fat_Rating,
        data.Bone_Mass_BM,
        data.Muscle_Mass_MM,
        data.Obesity,
        data.Total_Fat_Content_TFC,
        data.Visceral_Fat_Area_VFA,
        data.Visceral_Muscle_Area_VMA,
        data.Hepatic_Fat_Accumulation_HFA,
        data.Glucose,
        data.Total_Cholesterol_TC,
        data.Low_Density_Lipoprotein_LDL,
        data.High_Density_Lipoprotein_HDL,
        data.Triglyceride,
        data.Aspartat_Aminotransferaz_AST,
        data.Alanin_Aminotransferaz_ALT,
        data.Alkaline_Phosphatase_ALP,
        data.Creatinine,
        data.Glomerular_Filtration_Rate_GFR,
        data.C_Reactive_Protein_CRP,
        data.Hemoglobin_HGB,
        data.Vitamin_D
    ]])

    # Predict gallstone presence
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]


    # Return the result
    return{
        "prediction": int(prediction),
        "probability": float(probability) 
     } # returning the predicted value
