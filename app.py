import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Correct URL to the raw image file
image_url = 'https://raw.github.com/rizkywibowk/Sigma-Cabs-Pricing-Analysis/tree/main/Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'

# Fetch the image from the URL
response = requests.get(image_url)

# Ensure the response is successful
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=True)
else:
    st.error("Failed to load the image. Please check the URL or try a different image.")

import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

# Enhanced CSS hijau daun cerah
st.markdown("""
<style>
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #2e7d32;
        --background-color: #e8f5e8;
        --text-color: #1b5e20;
        --card-background: rgba(255, 255, 255, 0.9);
        --border-color: #a5d6a7;
        --success-color: #2e7d32;
        --warning-color: #ff8f00;
        --danger-color: #d32f2f;
        --info-color: #1976d2;
        --accent-green: #4caf50;
        --light-green: #c8e6c9;
        --dark-green: #1b5e20;
    }
    .stApp {
        background: linear-gradient(135deg, var(--background-color) 0%, color-mix(in srgb, var(--background-color) 85%, white 15%) 25%, color-mix(in srgb, var(--background-color) 90%, var(--accent-green) 10%) 50%, color-mix(in srgb, var(--background-color) 80%, white 20%) 75%, color-mix(in srgb, var(--background-color) 95%, var(--dark-green) 5%) 100%);
        color: var(--text-color);
        min-height: 100vh;
        background-attachment: fixed;
    }
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        line-height: 1.2;
        font-weight: 700;
        color: var(--secondary-color);
        filter: none !important;
    }
    .prediction-box {
        background: linear-gradient(135deg, var(--secondary-color) 0%, color-mix(in srgb, var(--secondary-color) 70%, var(--dark-green) 30%) 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        border: 2px solid var(--accent-green);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .prediction-box h1 {
        font-size: clamp(3rem, 10vw, 5rem) !important;
        margin: 1rem 0 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    .metric-card {
        background: var(--card-background);
        backdrop-filter: blur(15px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        min-height: clamp(120px, 18vh, 180px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    .metric-card h4, .metric-card p {
        color: var(--text-color) !important;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-green), var(--secondary-color));
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(46, 125, 50, 0.2);
        border-color: var(--accent-green);
    }
    .info-box, .contact-info, .footer-container {
        background: var(--card-background);
        backdrop-filter: blur(15px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        margin: 0.8rem 0;
        color: var(--text-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# Load model, scaler, feature_names, dan encoder
@st.cache_resource
def load_artifacts():
    model = joblib.load('Model for Streamlit/lgbm_model.pkl')
    scaler = joblib.load('Model for Streamlit/scaler_lgbm.pkl')
    feature_names = joblib.load('Model for Streamlit/feature_names_lgbm.pkl')
    cab_encoder = joblib.load('Model for Streamlit/cab_encoder.pkl')
    dest_encoder = joblib.load('Model for Streamlit/dest_encoder.pkl')
    conf_encoder = joblib.load('Model for Streamlit/conf_encoder.pkl')
    return model, scaler, feature_names, cab_encoder, dest_encoder, conf_encoder

model, scaler, feature_names, cab_encoder, dest_encoder, conf_encoder = load_artifacts()

# Header
st.markdown('<h1 class="main-header">⚡ Fast & Accurate Taxi Fare Prediction with LightGBM</h1>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🌟 About Sigma Cabs</h3>
        <p><strong>Sigma Cabs</strong> provides exceptional cab service. Our pricing is powered by an
        <strong>Advanced LightGBM model</strong> for precise and transparent fares.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="contact-info">
        <h4>📞 Contact Info</h4>
        <p><strong>Toll-Free:</strong><br>📞 1800-420-9999</p>
        <p><strong>24/7:</strong><br>📱 040-63 63 63 63</p>
    </div>
    """, unsafe_allow_html=True)

# --- INPUT SECTION ---
st.markdown("## 🎯 Predict Your Taxi Fare")

# Trip Details
trip_container = st.container()
with trip_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns([1, 1])
    with trip_col1:
        distance_input = st.number_input("🛣️ Distance (km):", min_value=0.1, max_value=100.0, value=5.0, step=0.1, key="distance_input")
        cab_type_input = st.selectbox("🚙 Vehicle Type:", cab_encoder.classes_.tolist(), key="cab_type_input")
    with trip_col2:
        destination_input = st.selectbox("📍 Destination:", dest_encoder.classes_.tolist(), key="destination_input")
        rating_input = st.slider("⭐ Your Rating:", 1, 5, 4, key="rating_input")

# Customer Information
customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns([1, 1])
    with cust_col1:
        months_input = st.number_input("📅 Customer Since (Months):", min_value=0, max_value=120, value=12, key="months_input")
        lifestyle_input = st.slider("💎 Lifestyle Index:", 1.0, 3.0, 2.0, step=0.1, key="lifestyle_input")
    with cust_col2:
        cancellations_input = st.number_input("❌ Cancellations Last Month:", min_value=0, max_value=10, value=0, key="cancellations_input")
        confidence_input = st.selectbox("🎯 Service Confidence:", conf_encoder.classes_.tolist(), key="confidence_input")

# Advanced Pricing Factors
with st.expander("⚙️ Advanced Pricing Factors (Real-time Input)"):
    st.markdown("**Adjust these real-time factors for maximum precision:**")
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    with adv_col1:
        traffic_input = st.slider("🚦 Traffic Density:", 0.0, 100.0, 50.0, key="traffic_input")
    with adv_col2:
        demand_input = st.slider("📈 Demand Level:", 0.0, 100.0, 50.0, key="demand_input")
    with adv_col3:
        weather_input = st.slider("🌧 Weather Impact:", 0.0, 100.0, 30.0, key="weather_input")

# Mapping input ke feature_names dan ENCODING
input_data = {
    'Trip_Distance': float(distance_input),
    'Customer_Rating': float(rating_input),
    'Customer_Since_Months': int(months_input),
    'Life_Style_Index': float(lifestyle_input),
    'Type_of_Cab': cab_type_input,
    'Confidence_Life_Style_Index': confidence_input,
    'Destination_Type': destination_input,
    'Gender': 'Male',
    'Cancellation_Last_1Month': int(cancellations_input),
    'Var1': float(traffic_input),
    'Var2': float(demand_input),
    'Var3': float(weather_input)
}

# ENCODING KATEGORIKAL
try:
    input_data['Type_of_Cab'] = cab_encoder.transform([input_data['Type_of_Cab']])[0]
    input_data['Destination_Type'] = dest_encoder.transform([input_data['Destination_Type']])[0]
    input_data['Confidence_Life_Style_Index'] = conf_encoder.transform([input_data['Confidence_Life_Style_Index']])[0]
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Susun urutan sesuai feature_names
def preprocess_input_lgbm(input_dict, feature_names_lgbm):
    processed = [input_dict[f] for f in feature_names_lgbm]
    arr = np.array(processed).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled

# PREDICTION BUTTON
if st.button('🔮 Predict Fare', type="primary", use_container_width=True):
    try:
        X_input = preprocess_input_lgbm(input_data, feature_names)
        prediction = model.predict(X_input)
        fare = float(prediction[0])

        st.markdown("""
        <div class="prediction-box">
            <h2>💰 Predicted Fare</h2>
            <h1>${:.2f}</h1>
            <p>Powered by LightGBM + Scaler</p>
        </div>
        """.format(fare), unsafe_allow_html=True)

        st.markdown("### 📊 Detailed Analysis Results")
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])

        with result_col1:
            surge = fare / max(1.0, float(distance_input) * 2.5)
            category = "High" if surge > 2.5 else "Medium" if surge > 1.5 else "Low"
            surge_class = "surge-low" if surge <= 1.5 else "surge-medium" if surge <= 2.5 else "surge-high"
            surge_html = """
            <div class="metric-card {}">
                <h4>📊 Surge Analysis</h4>
                <p><strong>Category:</strong> {}</p>
                <p><strong>Multiplier:</strong> {:.2f}x</p>
                <p><strong>Distance:</strong> {} km</p>
            </div>
            """.format(surge_class, category, surge, distance_input)
            st.markdown(surge_html, unsafe_allow_html=True)

        with result_col2:
            loyalty = "VIP" if months_input > 24 else "Loyal" if months_input > 12 else "Regular" if months_input > 3 else "New"
            loyalty_html = """
            <div class="metric-card">
                <h4>👤 Customer Profile</h4>
                <p><strong>Loyalty Status:</strong> {}</p>
                <p><strong>Rating:</strong> {}/5.0 ⭐</p>
                <p><strong>Since:</strong> {} months</p>
            </div>
            """.format(loyalty, rating_input, months_input)
            st.markdown(loyalty_html, unsafe_allow_html=True)

        with result_col3:
            base_fare = 10.0
            distance_cost_val = float(distance_input) * 2.5
            surge_additional = (distance_cost_val * (surge - 1))
            total_fare = base_fare + distance_cost_val + surge_additional
            fare_html = """
            <div class="metric-card">
                <h4>💰 Precision Fare</h4>
                <p><strong>Base:</strong> ${:.2f}</p>
                <p><strong>Distance:</strong> ${:.2f}</p>
                <p><strong>Surge:</strong> +${:.2f}</p>
                <p><strong>Total:</strong> ${:.2f}</p>
            </div>
            """.format(base_fare, distance_cost_val, surge_additional, total_fare)
            st.markdown(fare_html, unsafe_allow_html=True)

    except Exception as e:
        error_msg = str(e)
        st.error("❌ Prediction error: {}".format(error_msg))
        st.markdown("""
        <div class="prediction-box">
            <h2>🎯 Fallback Pricing</h2>
            <h1>$15.00</h1>
            <p>Using simplified algorithm</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
footer_html = """
<div class="footer-container" style="text-align: center; padding: clamp(1.5rem, 4vw, 2rem); border-radius: 15px; margin-top: 1.5rem;">
    <h3 style="margin: 0; font-size: clamp(1.3rem, 5vw, 2rem);">🚕 Sigma Cabs - Powered by LightGBM</h3>
    <p style="margin: 1rem 0; font-size: clamp(1rem, 3vw, 1.2rem);">Safe • Reliable • Affordable • 24/7 Available</p>
    <p style="margin: 0; font-size: clamp(0.9rem, 2.5vw, 1rem);">
        <strong>Python {} | LightGBM Model | 🌱 All Device Optimized</strong>
    </p>
</div>
""".format(python_version)
st.markdown(footer_html, unsafe_allow_html=True)
