import streamlit as st
import pandas as pd
import pickle

# Load the Random Forest model
try:
    with open('random_forestt_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("❌ Model file 'random_forest_model.pkl' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Define the function to predict rain
def predict_rain(PRECTOT, ws50m_range):
    # Create a DataFrame with the correct feature names
    input_data = pd.DataFrame({'PRECTOT': [PRECTOT], 'ws50m_range': [ws50m_range]})
    try:
        prediction = model.predict(input_data)[0]  # Ensure single value is extracted
        probability = model.predict_proba(input_data)[:, 1][0]  # Probability of class 1 (rain)
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

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
    prediction, probability = predict_rain(PRECTOT, ws50m_range)
    st.markdown("---")
    if prediction == 1:
        st.success('🌧️ It is likely to rain.')
    else:
        st.info('☀️ It is unlikely to rain.')
    st.write(f"**Probability of rain:** {probability:.2%}")

# Additional Information Section
st.markdown("---")
st.subheader("About This App")
st.write("""
This app predicts the likelihood of rain based on:
1. **Total Precipitation (PRECTOT)**: The amount of precipitation in mm.
2. **Wind Speed at 50m (ws50m_range)**: The average wind speed at a height of 50m.

The prediction is powered by a pre-trained Random Forestt model.
""")
