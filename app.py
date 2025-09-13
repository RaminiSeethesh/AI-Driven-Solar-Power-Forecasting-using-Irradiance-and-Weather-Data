import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, time

# Modern UI Configuration
st.set_page_config(
    page_title="Solar Forecast",
    page_icon="☀️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('solar_power_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except:
        return None, None

def predict_power(data, model, scaler):
    df = pd.DataFrame([data])
    df['DateTime'] = pd.to_datetime(df['Date and Time'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    df['Season'] = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    features = ['Global Irradiance (GHI)', 'Direct Normal Irradiance (DNI)', 'Azimuth Angle',
               'Dry Bulb Temperature', 'Wet Bulb Temperature', 'Dew Point Temperature',
               'Relative Humidity', 'Cloud Coverage', 'Hour', 'Month', 'Season', 'DayOfYear',
               'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
    
    X = scaler.transform(df[features])
    return max(0, model.predict(X)[0])

def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar Power Forecast</h1>', unsafe_allow_html=True)
    
    model, scaler = load_models()
    if not model:
        st.error("🚫 Model not found. Run training first.")
        return
    
    # Input Section
    st.markdown("### 🌤️ Weather Conditions")
    
    col1, col2 = st.columns(2)
    with col1:
        date_val = st.date_input("📅 Date", datetime.now().date())
        ghi = st.slider("☀️ Global Irradiance", 0, 1000, 800, help="W/m²")
        azimuth = st.slider("🧭 Azimuth Angle", 0, 360, 180, help="degrees")
        dry_temp = st.slider("🌡️ Temperature", -10, 50, 25, help="°C")
        humidity = st.slider("💧 Humidity", 0, 100, 65, help="%")
    
    with col2:
        time_val = st.time_input("🕐 Time", time(12, 0))
        dni = st.slider("🔆 Direct Irradiance", 0, 800, 600, help="W/m²")
        wet_temp = st.slider("🌡️ Wet Bulb Temp", -10, 40, 20, help="°C")
        dew_temp = st.slider("💨 Dew Point", -20, 35, 15, help="°C")
        cloud_cover = st.slider("☁️ Cloud Cover", 0, 100, 20, help="%")
    
    # Prediction
    if st.button("🔮 Predict Power", type="primary", use_container_width=True):
        data = {
            'Date and Time': f"{date_val} {time_val}",
            'Global Irradiance (GHI)': ghi, 'Direct Normal Irradiance (DNI)': dni,
            'Azimuth Angle': azimuth, 'Dry Bulb Temperature': dry_temp,
            'Wet Bulb Temperature': wet_temp, 'Dew Point Temperature': dew_temp,
            'Relative Humidity': humidity, 'Cloud Coverage': cloud_cover
        }
        
        prediction = predict_power(data, model, scaler)
        
        # Result Display
        st.markdown(f"""
        <div class="prediction-result">
            <h2>⚡ {prediction:.1f} kWh</h2>
            <p>Predicted Solar Power Output</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Power Output"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 3], 'color': "#ffcccb"},
                    {'range': [3, 6], 'color': "#ffd700"},
                    {'range': [6, 10], 'color': "#90ee90"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8}}))
        
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Condition Status
        if ghi > 700 and cloud_cover < 30:
            st.success("🌞 Excellent solar conditions")
        elif ghi > 400 and cloud_cover < 60:
            st.info("⛅ Good solar conditions")
        else:
            st.warning("☁️ Limited solar potential")

if __name__ == "__main__":
    main()