import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
try:
    with open('random_forestt_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("❌ Model file 'random_forest_model.pkl' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Function to predict rain
def predict_rain(PRECTOT, ws50m_range):
    input_data = pd.DataFrame({'PRECTOT': [PRECTOT], 'ws50m_range': [ws50m_range]})
    prediction = model.predict(input_data)[0]  # Extract single prediction
    probability = model.predict_proba(input_data)[:, 1][0]  # Extract probability for class 1
    return prediction, probability

# Streamlit UI
st.title('🌧️ Rain Prediction App')

st.sidebar.header('Input Parameters')
PRECTOT = st.sidebar.slider(
    'Total Precipitation (PRECTOT)', min_value=0.0, max_value=100.0, value=10.0, step=0.1
)
ws50m_range = st.sidebar.slider(
    'Wind Speed at 50m Range (ws50m_range)', min_value=0, max_value=30, value=10, step=1
)

if st.sidebar.button('Predict'):
    try:
        prediction, probability = predict_rain(PRECTOT, ws50m_range)
        st.markdown("---")
        if prediction == 1:
            st.success(f"🌧️ It is likely to rain with a probability of {probability:.2%}.")
        else:
            st.info(f"☀️ It is unlikely to rain with a probability of {(1 - probability):.2%}.")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
