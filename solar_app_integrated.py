import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, time
import os

# Modern UI Configuration
st.set_page_config(page_title="Solar Forecast", page_icon="☀️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
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

def train_model():
    """Train and save the solar power prediction model"""
    with st.spinner("Training model..."):
        # Generate sample data
        np.random.seed(42)
        n_samples = 2000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='h')
        
        df = pd.DataFrame({
            'Date and Time': dates,
            'Global Irradiance (GHI)': np.random.normal(400, 200, n_samples).clip(0, 1000),
            'Direct Normal Irradiance (DNI)': np.random.normal(300, 150, n_samples).clip(0, 800),
            'Azimuth Angle': np.random.uniform(0, 360, n_samples),
            'Dry Bulb Temperature': np.random.normal(25, 10, n_samples),
            'Wet Bulb Temperature': np.random.normal(20, 8, n_samples),
            'Dew Point Temperature': np.random.normal(15, 8, n_samples),
            'Relative Humidity': np.random.uniform(30, 90, n_samples),
            'Cloud Coverage': np.random.uniform(0, 100, n_samples)
        })
        
        # Create target variable
        df['Solar Power (kWh)'] = (
            0.005 * df['Global Irradiance (GHI)'] + 
            0.003 * df['Direct Normal Irradiance (DNI)'] +
            0.01 * df['Dry Bulb Temperature'] -
            0.002 * df['Cloud Coverage'] +
            np.random.normal(0, 0.5, n_samples)
        ).clip(0, None)
        
        # Feature engineering
        df['DateTime'] = pd.to_datetime(df['Date and Time'])
        df['Hour'] = df['DateTime'].dt.hour
        df['Month'] = df['DateTime'].dt.month
        df['DayOfYear'] = df['DateTime'].dt.dayofyear
        df['Season'] = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Prepare features
        feature_cols = [
            'Global Irradiance (GHI)', 'Direct Normal Irradiance (DNI)', 'Azimuth Angle',
            'Dry Bulb Temperature', 'Wet Bulb Temperature', 'Dew Point Temperature',
            'Relative Humidity', 'Cloud Coverage', 'Hour', 'Month', 'Season', 'DayOfYear',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
        ]
        
        X = df[feature_cols]
        y = df['Solar Power (kWh)']
        
        # Scale and split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Save models
        joblib.dump(model, 'solar_power_model.pkl')
        joblib.dump(scaler, 'feature_scaler.pkl')
        
        return r2, rmse

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
    st.markdown('<h1 class="main-header">☀️ Solar Power Forecast</h1>', unsafe_allow_html=True)
    
    # Sidebar for model management
    with st.sidebar:
        st.header("🔧 Model Management")
        
        model, scaler = load_models()
        
        if model is None:
            st.warning("No trained model found")
            if st.button("🚀 Train Model", type="primary"):
                r2, rmse = train_model()
                st.success(f"Model trained! R² = {r2:.3f}, RMSE = {rmse:.3f}")
                st.rerun()
        else:
            st.success("✅ Model loaded")
            if st.button("🔄 Retrain Model"):
                r2, rmse = train_model()
                st.success(f"Model retrained! R² = {r2:.3f}, RMSE = {rmse:.3f}")
                st.rerun()
    
    if model is None:
        st.info("👈 Please train the model first using the sidebar")
        return
    
    # Main prediction interface
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
    
    # Auto-predict on input change
    data = {
        'Date and Time': f"{date_val} {time_val}",
        'Global Irradiance (GHI)': ghi, 'Direct Normal Irradiance (DNI)': dni,
        'Azimuth Angle': azimuth, 'Dry Bulb Temperature': dry_temp,
        'Wet Bulb Temperature': wet_temp, 'Dew Point Temperature': dew_temp,
        'Relative Humidity': humidity, 'Cloud Coverage': cloud_cover
    }
    
    prediction = predict_power(data, model, scaler)
    
    # Result Display
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
    
    with col2:
        st.markdown("### 📊 Conditions")
        st.metric("Irradiance", f"{ghi} W/m²")
        st.metric("Temperature", f"{dry_temp}°C")
        st.metric("Humidity", f"{humidity}%")
        st.metric("Cloud Cover", f"{cloud_cover}%")
        
        if ghi > 700 and cloud_cover < 30:
            st.success("🌞 Excellent")
        elif ghi > 400 and cloud_cover < 60:
            st.info("⛅ Good")
        else:
            st.warning("☁️ Limited")

if __name__ == "__main__":
    main()