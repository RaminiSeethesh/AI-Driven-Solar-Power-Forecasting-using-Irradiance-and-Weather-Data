import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def predict_solar_power(input_data, model_path='solar_power_model.pkl', scaler_path='feature_scaler.pkl'):
    """
    Predict solar power output for new data
    
    Args:
        input_data (dict): Dictionary with weather and time data
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
    
    Returns:
        float: Predicted solar power output in kWh
    """
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Feature engineering (same as training)
    if 'Date and Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date and Time'])
        df['Hour'] = df['DateTime'].dt.hour
        df['Month'] = df['DateTime'].dt.month
        df['DayOfYear'] = df['DateTime'].dt.dayofyear
        df['Season'] = df['Month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                       6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
        
        # Cyclical encoding
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Select features
    feature_cols = [
        'Global Irradiance (GHI)', 'Direct Normal Irradiance (DNI)',
        'Azimuth Angle', 'Dry Bulb Temperature', 'Wet Bulb Temperature',
        'Dew Point Temperature', 'Relative Humidity', 'Cloud Coverage',
        'Hour', 'Month', 'Season', 'DayOfYear',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
    ]
    
    X = df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return max(0, prediction)  # Ensure non-negative output

# Example usage
if __name__ == "__main__":
    # Example input data
    sample_input = {
        'Date and Time': '2023-06-15 12:00:00',
        'Global Irradiance (GHI)': 800,
        'Direct Normal Irradiance (DNI)': 600,
        'Azimuth Angle': 180,
        'Dry Bulb Temperature': 28,
        'Wet Bulb Temperature': 22,
        'Dew Point Temperature': 18,
        'Relative Humidity': 65,
        'Cloud Coverage': 20
    }
    
    try:
        prediction = predict_solar_power(sample_input)
        print(f"Predicted Solar Power Output: {prediction:.2f} kWh")
    except FileNotFoundError:
        print("Model files not found. Please run the training script first.")
    except Exception as e:
        print(f"Error making prediction: {e}")