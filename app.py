import streamlit as st
from PIL import Image
import numpy as np
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🚕 Sigma Cabs - LightGBM Pricing Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Banner atas
image_path = 'Picture/Sigma-cabs-in-hyderabad-and-bangalore.jpg'
image = Image.open(image_path)
st.image(image, use_column_width=True)

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

# CSS netral agar tetap nyaman di light/dark mode
st.markdown("""
<style>
    .main-header {
        font-size: clamp(2rem, 6vw, 3rem);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.08);
        line-height: 1.2;
        font-weight: 700;
    }
    .prediction-box {
        background: linear-gradient(135deg, #388e3c 0%, #1b5e20 100%);
        padding: clamp(1.5rem, 4vw, 2rem);
        border-radius: 20px;
        color: #fff;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        border: 2px solid #4caf50;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .prediction-box h1 {
        font-size: clamp(3rem, 10vw, 5rem) !important;
        margin: 1rem 0 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
        color: #fff;
    }
    .metric-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(10px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
        min-height: clamp(120px, 18vh, 180px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: inherit;
        border: 1px solid #a5d6a7;
        position: relative;
        overflow: hidden;
    }
    .metric-card h4, .metric-card p {
        color: inherit !important;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4caf50, #2e7d32);
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(46, 125, 50, 0.13);
        border-color: #4caf50;
    }
    .info-box, .contact-info, .footer-container {
        background: rgba(255,255,255,0.87);
        backdrop-filter: blur(10px);
        padding: clamp(1rem, 3vw, 1.2rem);
        border-radius: 15px;
        margin: 0.8rem 0;
        color: inherit;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #a5d6a7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load('Model for Streamlit/lgbm_model.pkl')
    scaler = joblib.load('Model for Streamlit/scaler_lgbm.pkl')
    feature_names = joblib.load('Model for Streamlit/feature_names_lgbm.pkl')
    cab_encoder = joblib.load('Model for Streamlit/cab_encoder.pkl')
    dest_encoder = joblib.load('Model for Streamlit/dest_encoder.pkl')
    conf_encoder = joblib.load('Model for Streamlit/conf_encoder.pkl')
    gender_encoder = joblib.load('Model for Streamlit/gender_encoder.pkl')
    return model, scaler, feature_names, cab_encoder, dest_encoder, conf_encoder, gender_encoder

model, scaler, feature_names, cab_encoder, dest_encoder, conf_encoder, gender_encoder = load_artifacts()

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

st.markdown("## 🎯 Predict Your Taxi Fare")

GENDER_OPTIONS = ["Male", "Female", "Other"]
CAB_TYPE_OPTIONS = ["Economy", "Comfy", "Exclusive"]
CONFIDENCE_OPTIONS = ["High Confidence", "Medium Confidence", "Low Confidence"]
DEST_OPTIONS = ["Airport", "City Center", "Mall", "University"]

trip_container = st.container()
with trip_container:
    st.markdown("### 🚗 Trip Details")
    trip_col1, trip_col2 = st.columns([1, 1])
    with trip_col1:
        distance_input = st.number_input(
            "🛣️ Distance (km):",
            min_value=0.1, max_value=100.0, value=5.0, step=0.1,
            key="distance_input",
            help="Enter the trip distance in kilometers (e.g., 5.0 km)."
        )
        cab_type_input = st.selectbox(
            "🚙 Vehicle Type:",
            CAB_TYPE_OPTIONS,
            key="cab_type_input",
            help="Choose your preferred vehicle type: Economy (budget), Comfy (standard), or Exclusive (premium)."
        )
    with trip_col2:
        destination_input = st.selectbox(
            "📍 Destination:",
            DEST_OPTIONS,
            key="destination_input",
            help="Pick your destination type for the trip."
        )
        rating_input = st.slider(
            "⭐ Your Rating:",
            1, 5, 4,
            key="rating_input",
            help="How would you rate your overall experience as a customer? (1 = lowest, 5 = highest)"
        )

customer_container = st.container()
with customer_container:
    st.markdown("### 👤 Customer Information")
    cust_col1, cust_col2 = st.columns([1, 1])
    with cust_col1:
        months_input = st.number_input(
            "📅 Customer Since (Months):",
            min_value=0, max_value=120, value=12,
            key="months_input",
            help="How many months have you been a Sigma Cabs customer?"
        )
        lifestyle_input = st.slider(
            "💎 Lifestyle Index:",
            1.0, 3.0, 2.0, step=0.1,
            key="lifestyle_input",
            help="Lifestyle index (1.0 = basic, 3.0 = luxury)."
        )
    with cust_col2:
        cancellations_input = st.number_input(
            "❌ Cancellations Last Month:",
            min_value=0, max_value=10, value=0,
            key="cancellations_input",
            help="How many times did you cancel a ride last month?"
        )
        confidence_input = st.selectbox(
            "🎯 Service Confidence:",
            CONFIDENCE_OPTIONS,
            key="confidence_input",
            help="How confident are you in your lifestyle index? Higher confidence may give more accurate fare."
        )
        gender_input = st.selectbox(
            "🚻 Gender:",
            GENDER_OPTIONS,
            key="gender_input",
            help="Select your gender for a personalized fare experience."
        )

with st.expander("⚙️ Advanced Pricing Factors (Real-time Input)"):
    st.markdown("**Adjust these real-time factors for maximum precision:**")
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    with adv_col1:
        traffic_input = st.slider(
            "🚦 Traffic Density:", 0.0, 100.0, 50.0,
            key="traffic_input",
            help="Estimate the current traffic density (0 = empty road, 100 = jammed)."
        )
    with adv_col2:
        demand_input = st.slider(
            "📈 Demand Level:", 0.0, 100.0, 50.0,
            key="demand_input",
            help="Estimate the current demand for cabs (0 = low, 100 = very high)."
        )
    with adv_col3:
        weather_input = st.slider(
            "🌧 Weather Impact:", 0.0, 100.0, 30.0,
            key="weather_input",
            help="Estimate the weather's impact on your ride (0 = clear, 100 = severe weather)."
        )

input_data = {
    'Trip_Distance': float(distance_input),
    'Customer_Rating': float(rating_input),
    'Customer_Since_Months': int(months_input),
    'Life_Style_Index': float(lifestyle_input),
    'Type_of_Cab': cab_type_input,
    'Confidence_Life_Style_Index': confidence_input,
    'Destination_Type': destination_input,
    'Gender': gender_input,
    'Cancellation_Last_1Month': int(cancellations_input),
    'Var1': float(traffic_input),
    'Var2': float(demand_input),
    'Var3': float(weather_input)
}

try:
    input_data['Type_of_Cab'] = cab_encoder.transform([input_data['Type_of_Cab']])[0]
    input_data['Destination_Type'] = dest_encoder.transform([input_data['Destination_Type']])[0]
    input_data['Confidence_Life_Style_Index'] = conf_encoder.transform([input_data['Confidence_Life_Style_Index']])[0]
    input_data['Gender'] = gender_encoder.transform([input_data['Gender']])[0]
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

def preprocess_input_lgbm(input_dict, feature_names_lgbm):
    processed = [input_dict[f] for f in feature_names_lgbm]
    arr = np.array(processed).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled

if st.button('🔮 Predict Fare', type="primary", use_container_width=True):
    try:
        X_input = preprocess_input_lgbm(input_data, feature_names)
        prediction = model.predict(X_input)
        fare = float(prediction[0])

        st.markdown(f"""
        <div class="prediction-box">
            <h2>💰 Predicted Fare</h2>
            <h1>${fare:.2f}</h1>
            <p>Powered by LightGBM + Scaler</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Detailed Analysis Results")
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])

        with result_col1:
            surge = fare / max(1.0, float(distance_input) * 2.5)
            category = "High" if surge > 2.5 else "Medium" if surge > 1.5 else "Low"
            surge_class = "surge-low" if surge <= 1.5 else "surge-medium" if surge <= 2.5 else "surge-high"
            surge_html = f"""
            <div class="metric-card {surge_class}">
                <h4>📊 Surge Analysis</h4>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Multiplier:</strong> {surge:.2f}x</p>
                <p><strong>Distance:</strong> {distance_input} km</p>
            </div>
            """
            st.markdown(surge_html, unsafe_allow_html=True)

        with result_col2:
            loyalty = "VIP" if months_input > 24 else "Loyal" if months_input > 12 else "Regular" if months_input > 3 else "New"
            loyalty_html = f"""
            <div class="metric-card">
                <h4>👤 Customer Profile</h4>
                <p><strong>Loyalty Status:</strong> {loyalty}</p>
                <p><strong>Rating:</strong> {rating_input}/5.0 ⭐</p>
                <p><strong>Since:</strong> {months_input} months</p>
            </div>
            """
            st.markdown(loyalty_html, unsafe_allow_html=True)

        with result_col3:
            base_fare = 10.0
            distance_cost_val = float(distance_input) * 2.5
            surge_additional = (distance_cost_val * (surge - 1))
            total_fare = base_fare + distance_cost_val + surge_additional
            fare_html = f"""
            <div class="metric-card">
                <h4>💰 Precision Fare</h4>
                <p><strong>Base:</strong> ${base_fare:.2f}</p>
                <p><strong>Distance:</strong> ${distance_cost_val:.2f}</p>
                <p><strong>Surge:</strong> +${surge_additional:.2f}</p>
                <p><strong>Total:</strong> ${total_fare:.2f}</p>
            </div>
            """
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

st.markdown("""
<style>
.watermark-box {
    width: 100%;
    margin: 0 auto 1.1em auto;
    padding: 1.2em 0.5em 1.1em 0.5em;
    border-radius: 9px;
    background: linear-gradient(90deg, #263238 0%, #37474f 100%);
    text-align: center;
    border-bottom: 4px solid #4caf50;
    box-shadow: 0 2px 12px rgba(76,175,80,0.04);
}
.watermark-title {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: clamp(1.4rem, 3vw, 2.2rem);
    font-weight: 900;
    color: #e8f5e9;
    margin-bottom: 0.1em;
    letter-spacing: 1.3px;
    text-shadow: 1px 2px 8px rgba(76,175,80,0.10);
}
.watermark-sub {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: clamp(1.1rem, 2.3vw, 1.6rem);
    font-weight: 700;
    color: #66bb6a;
    margin-top: 0.2em;
    letter-spacing: 1.1px;
}
@media (prefers-color-scheme: light) {
    .watermark-box {
        background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
        border-bottom: 4px solid #388e3c;
    }
    .watermark-title { color: #263238; }
    .watermark-sub { color: #388e3c; }
}
</style>
<div class="watermark-box">
    <div class="watermark-title">
        AI Fare Predict - Powered by <span style="color:#4caf50;">Advanced LightGBM</span>
    </div>
    <div class="watermark-sub">
        RIZKY WIBOWO KUSUMO MODEL
    </div>
</div>
""", unsafe_allow_html=True)
    <b>Python 3.9</b> | <b>Advanced LightGBM Model</b> <span style="font-size:1.2em;">🌱</span> <span style="color:#43a047;">All Device Optimized</span>
</div>
""", unsafe_allow_html=True)
