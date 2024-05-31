import streamlit as st
import pandas as pd
import pickle

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the function to predict rain
def predict_rain(PRECTOT, ws50m_range):
    input_data = pd.DataFrame({'PRECTOT': [PRECTOT], 'ws50m_range': [ws50m_range]})
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:,1]
    return prediction[0], probability[0]

# Streamlit UI
st.title('Rain Prediction')

st.sidebar.header('Input Parameters')
PRECTOT = st.sidebar.slider('Total Precipitation (PRECTOT)', min_value=0.0, max_value=100.0, value=10.0, step=1.0)
ws50m_range = st.sidebar.slider('Wind Speed at 50m Range (ws50m_range)', min_value=0, max_value=30, value=10, step=1)

if st.sidebar.button('Predict'):
    prediction, probability = predict_rain(PRECTOT, ws50m_range)
    if prediction == 1:
        st.write('It is likely to rain.')
        st.write('Probability of rain:', probability)
    else:
        st.write('It is unlikely to rain.')
        st.write('Probability of rain:', probability)