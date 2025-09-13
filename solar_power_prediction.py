import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SolarPowerPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def preprocess_data(self, df):
        """Preprocess the solar dataset"""
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Convert Date and Time to datetime
        df['DateTime'] = pd.to_datetime(df['Date and Time'])
        
        # Extract time features
        df['Hour'] = df['DateTime'].dt.hour
        df['Day'] = df['DateTime'].dt.day
        df['Month'] = df['DateTime'].dt.month
        df['DayOfYear'] = df['DateTime'].dt.dayofyear
        df['Season'] = df['Month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                       6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
        
        # Cyclical encoding for time features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def prepare_features(self, df):
        """Select and prepare features for modeling"""
        feature_cols = [
            'Global Irradiance (GHI)', 'Direct Normal Irradiance (DNI)',
            'Azimuth Angle', 'Dry Bulb Temperature', 'Wet Bulb Temperature',
            'Dew Point Temperature', 'Relative Humidity', 'Cloud Coverage',
            'Hour', 'Month', 'Season', 'DayOfYear',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
        ]
        
        self.feature_names = feature_cols
        return df[feature_cols]
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        # Random Forest
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        # Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred),
                'predictions': y_pred
            }
            
        return results
    
    def plot_results(self, y_test, results):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, (name, metrics) in enumerate(results.items()):
            axes[i].scatter(y_test, metrics['predictions'], alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Solar Power (kWh)')
            axes[i].set_ylabel('Predicted Solar Power (kWh)')
            axes[i].set_title(f'{name}\nR² = {metrics["R²"]:.3f}, RMSE = {metrics["RMSE"]:.3f}')
            
        plt.tight_layout()
        plt.show()

def main():
    # Load your dataset (replace with your actual file path)
    # df = pd.read_csv('your_solar_data.csv')
    
    # For demonstration, create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    sample_data = {
        'Date and Time': dates,
        'Global Irradiance (GHI)': np.random.normal(400, 200, n_samples).clip(0, 1000),
        'Direct Normal Irradiance (DNI)': np.random.normal(300, 150, n_samples).clip(0, 800),
        'Azimuth Angle': np.random.uniform(0, 360, n_samples),
        'Dry Bulb Temperature': np.random.normal(25, 10, n_samples),
        'Wet Bulb Temperature': np.random.normal(20, 8, n_samples),
        'Dew Point Temperature': np.random.normal(15, 8, n_samples),
        'Relative Humidity': np.random.uniform(30, 90, n_samples),
        'Cloud Coverage': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create target variable (solar power output)
    # Simplified relationship: power depends mainly on irradiance and temperature
    df['Solar Power (kWh)'] = (
        0.005 * df['Global Irradiance (GHI)'] + 
        0.003 * df['Direct Normal Irradiance (DNI)'] +
        0.01 * df['Dry Bulb Temperature'] -
        0.002 * df['Cloud Coverage'] +
        np.random.normal(0, 0.5, n_samples)
    ).clip(0, None)
    
    # Initialize predictor
    predictor = SolarPowerPredictor()
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = predictor.preprocess_data(df)
    
    # Prepare features and target
    X = predictor.prepare_features(df_processed)
    y = df_processed['Solar Power (kWh)']
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\nTraining models...")
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = predictor.evaluate_models(X_test, y_test)
    
    # Print results
    print("\nModel Performance:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  R²:   {metrics['R²']:.4f}")
        print()
    
    # Plot results
    predictor.plot_results(y_test, results)
    
    # Feature importance for Random Forest
    rf_model = predictor.models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': predictor.feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features (Random Forest):")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()