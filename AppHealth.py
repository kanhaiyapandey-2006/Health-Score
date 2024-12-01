import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add custom CSS for background image (optional, you can remove this section if you don't want any background image)
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvcm0zNzNiYXRjaDE1LWJnLTExLmpwZw.jpg');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
        }
        body {
            background-color: #f5f5f5;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
        }
        .stTitle {
            color: #2e3d49;
            font-family: 'Arial', sans-serif;
            font-size: 36px; /* Increased font size */
            font-style: italic; /* Italic font */
        }
        .stText {
            color: #4c4c4c;
            font-size: 20px; /* Increased font size */
            font-style: italic; /* Italic font */
        }
        .stSelectbox select, .stNumberInput input {
            font-size: 18px; /* Increased font size */
            padding: 5px;
            border-radius: 5px;
        }
        h3 {
            font-size: 24px;
            font-style: italic;
            color: #2E8B57;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
model = joblib.load('Health_Score.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler you used during training

# Title and description
st.title("🌱 Health Score Prediction")
st.markdown("""
    <h3 style='text-align: center;'>Enter Your Lifestyle Data to Predict Your Health Score</h3>
    <p style='text-align: center;'>This app predicts your health score based on factors such as age, BMI, diet, exercise, sleep, smoking, and alcohol consumption.</p>
""", unsafe_allow_html=True)

# Input fields for data collection
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
    bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    exercise_frequency = st.number_input('Exercise Frequency (hours per week)', min_value=0, max_value=20, value=3, step=1)

with col2:
    diet_quality = st.number_input('Diet Quality (1 to 10)', min_value=1, max_value=10, value=7, step=1)
    sleep_hours = st.number_input('Sleep Hours (per night)', min_value=4, max_value=12, value=7, step=1)
    smoking_status = st.selectbox('Smoking Status', ['Non-Smoker', 'Occasional Smoker', 'Regular Smoker'])
    alcohol_consumption = st.number_input('Alcohol Consumption (drinks per week)', min_value=0, max_value=20, value=2, step=1)

# Convert smoking status to numerical value
smoking_map = {'Non-Smoker': 0, 'Occasional Smoker': 1, 'Regular Smoker': 2}
smoking_status = smoking_map[smoking_status]

# Collect user input into a numpy array
user_input = np.array([age, bmi, exercise_frequency, diet_quality, sleep_hours, smoking_status, alcohol_consumption]).reshape(1, -1)

# Scale the input data using the loaded scaler
user_input_scaled = scaler.transform(user_input)

# Make prediction when the user clicks 'Predict'
if st.button('🔮 Predict Health Score'):
    with st.spinner('Making prediction...'):
        prediction = model.predict(user_input_scaled)
    st.success(f'Predicted Health Score: {prediction[0]:.2f}')

# Additional styling and info
st.write("""
    The health score is an estimate based on the data you provided. It combines factors like exercise, diet, 
    sleep, and lifestyle choices to predict your overall health status.
""")

# Add a "Creator" section below
st.markdown("""
    <h3 style='text-align: center;'>Created by: Kanhaiya Pandey </h3>
    <p style='text-align: center;'>Feel free to reach out for questions or feedback!</p>
""", unsafe_allow_html=True)
