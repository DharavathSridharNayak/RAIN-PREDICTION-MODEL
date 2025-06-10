import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

# Set page configuration
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: black;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: black;
        border-radius: 5px;
    }
    .css-1aumxhk {
        background-color: #e8f4fc;
        border-radius: 10px;
        padding: 20px;
    }
    .title {
        color: #1e3d6b;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌧️ Rainfall Prediction App")
st.markdown("""
    Predict the probability of rainfall tomorrow based on weather data.
    This machine learning model uses Random Forest Classifier for accurate predictions.
    """)

# Load sample data or create synthetic data
@st.cache_data
def load_data():
    # In a real app, you would load your actual dataset here
    # This is synthetic data for demonstration
    data = {
        'MinTemp': np.random.uniform(5, 25, 1000),
        'MaxTemp': np.random.uniform(15, 40, 1000),
        'Rainfall': np.random.uniform(0, 50, 1000),
        'Evaporation': np.random.uniform(0, 20, 1000),
        'Sunshine': np.random.uniform(0, 14, 1000),
        'Humidity9am': np.random.randint(20, 100, 1000),
        'Humidity3pm': np.random.randint(20, 100, 1000),
        'Pressure9am': np.random.uniform(990, 1040, 1000),
        'Pressure3pm': np.random.uniform(990, 1040, 1000),
        'Cloud9am': np.random.randint(0, 9, 1000),
        'Cloud3pm': np.random.randint(0, 9, 1000),
        'Temp9am': np.random.uniform(10, 35, 1000),
        'Temp3pm': np.random.uniform(15, 40, 1000),
        'RainToday': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'RainTomorrow': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)

df = load_data()

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to get user input
def user_input_features():
    min_temp = st.sidebar.slider('Minimum Temperature (°C)', 0.0, 30.0, 15.0)
    max_temp = st.sidebar.slider('Maximum Temperature (°C)', 10.0, 45.0, 25.0)
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 100.0, 0.0)
    humidity_9am = st.sidebar.slider('Humidity at 9am (%)', 0, 100, 60)
    humidity_3pm = st.sidebar.slider('Humidity at 3pm (%)', 0, 100, 50)
    pressure_9am = st.sidebar.slider('Pressure at 9am (hPa)', 990.0, 1040.0, 1015.0)
    pressure_3pm = st.sidebar.slider('Pressure at 3pm (hPa)', 990.0, 1040.0, 1015.0)
    cloud_9am = st.sidebar.slider('Cloud cover at 9am (oktas)', 0, 8, 3)
    cloud_3pm = st.sidebar.slider('Cloud cover at 3pm (oktas)', 0, 8, 4)
    rain_today = st.sidebar.selectbox('Rain Today?', ('No', 'Yes'))
    
    data = {
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'RainToday': 1 if rain_today == 'Yes' else 0
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.subheader("User Input Parameters")
st.write(input_df)

# Train model
@st.cache_resource
def train_model():
    # Prepare data
    X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 
            'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday']]
    y = df['RainTomorrow']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model()

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display results
st.subheader('Prediction')
rain_tomorrow = 'Yes' if prediction[0] == 1 else 'No'
st.markdown(f"<h3 style='text-align: center; color: {'red' if rain_tomorrow == 'Yes' else 'green'};'>"
            f"Will it rain tomorrow? {rain_tomorrow}</h3>", 
            unsafe_allow_html=True)

st.subheader('Prediction Probability')
st.write(f"Probability of rain tomorrow: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Probability of no rain tomorrow: {prediction_proba[0][0]*100:.2f}%")

# Show model accuracy
st.sidebar.subheader('Model Accuracy')
st.sidebar.write(f"Accuracy: {accuracy*100:.2f}%")

# Visualization section
st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("**Temperature Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df['MaxTemp'], kde=True, ax=ax)
    plt.xlabel('Maximum Temperature (°C)')
    st.pyplot(fig)

with col2:
    st.write("**Humidity at 9am vs 3pm**")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Humidity9am', y='Humidity3pm', hue='RainTomorrow', data=df, ax=ax)
    plt.xlabel('Humidity at 9am (%)')
    plt.ylabel('Humidity at 3pm (%)')
    st.pyplot(fig)

# Add some information
st.markdown("""
    ### About This App
    This app predicts the probability of rainfall tomorrow using weather data.
    The model was trained using Random Forest Classifier on historical weather data.
    
    **Note:** This is a demonstration app using synthetic data. For real-world applications,
    you should train the model with actual weather data from your region.
    """)

# Add download button for sample data
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df.head(100))
st.download_button(
    label="Download Sample Data (CSV)",
    data=csv,
    file_name='sample_weather_data.csv',
    mime='text/csv'
)
