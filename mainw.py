# app_asd_ready.py
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained pipeline
pipeline = joblib.load("random_forest_pipeline.pkl")

# Page config
st.set_page_config(page_title="ASD Prediction App", page_icon="🧠", layout="centered")

# Header image (replace with your own image if you want)
st.image("why-doctors-changed-way-categorize-autism-alt-1440x810.webp", use_column_width=True)

st.title("🧠 ASD Prediction Web App")
st.markdown("""
Welcome! Enter the features below to predict **ASD Class** using our Random Forest model by Rana Basak.
""")

# Numeric columns with realistic defaults
numeric_columns = {
    'A1_Score': 0, 'A2_Score': 0, 'A3_Score': 0, 'A4_Score': 0, 'A5_Score': 0,
    'A6_Score': 0, 'A7_Score': 0, 'A8_Score': 0, 'A9_Score': 0, 'A10_Score': 0,
    'age': 10, 'result': 0
}

# Categorical columns with options
categorical_options = {
    'gender': ['male', 'female'],
    'ethnicity': ['White', 'Asian', 'Hispanic', 'Black', 'Other'],
    'jaundice': ['yes', 'no'],
    'austim': ['yes', 'no'],
    'contry_of_res': ['USA', 'UK', 'India', 'Other'],
    'used_app_before': ['yes', 'no'],
    'relation': ['Self', 'Parent', 'Relative', 'Guardian', 'Other']
}

input_data = {}

# Numeric inputs in an expander
with st.expander("Numeric Features (Scores & Age)"):
    for col, default in numeric_columns.items():
        input_data[col] = [st.number_input(f"Enter {col}", value=default, step=1)]

# Categorical inputs in another expander
with st.expander("Categorical Features"):
    for col, options in categorical_options.items():
        input_data[col] = [st.selectbox(f"Select {col}", options)]

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Predict button
if st.button("Predict ASD Class"):
    try:
        prediction = pipeline.predict(input_df)
        # Map 0/1 to readable output
        pred_label = "No ASD" if prediction[0] == 0 else "ASD"
        st.success(f"✅ Predicted Class/ASD: {pred_label}")
        
        # Display placeholder images based on prediction
        if prediction[0] == 1:
            st.image("engproc-59-00205-g001.png", caption="ASD Detected", use_column_width=True)
        else:
            st.image("nad-no-abnormality-detected-acronym-260nw-2365656621.webp", caption="No ASD Detected", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# Reset button
#if st.button("Reset Inputs"):
   # st.experimental_rerun()
