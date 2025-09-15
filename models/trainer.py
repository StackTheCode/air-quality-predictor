import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import get_engine

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_data(self):
        """Load and prepare data for training"""
        engine = get_engine()
        
        # Join air quality and weather data
        query = """
        SELECT 
            a.timestamp,
            a.city,
            a.aqi,
            a.pm25,
            a.pm10,
            w.temperature,
            w.humidity,
            w.pressure,
            w.wind_speed
        FROM air_quality_table a
        JOIN weather_data_table w ON a.city = w.city 
            AND ABS(EXTRACT(EPOCH FROM (a.timestamp - w.timestamp))) < 3600
        ORDER BY a.timestamp
        """
        
        df = pd.read_sql(query, engine)
        
        if len(df) < 10:
            raise ValueError("Not enough data to train model. Collect more data first!")
        
        # Feature engineering
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Features for prediction
        feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'hour', 'day_of_week']
        
        X = df[feature_cols]
        y = df['pm25']  # Predict PM2.5 levels
        
        self.feature_names = feature_cols
        
        return X, y, df
    
    def train_model(self):
        """Train the air quality prediction model"""
        try:
            X, y, df = self.prepare_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Training MAE: {train_mae:.2f}")
            print(f"📊 Test MAE: {test_mae:.2f}")
            print(f"📊 Training data points: {len(X_train)}")
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 Feature Importance:")
            print(importance)
            
            return {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': importance,
                'data_points': len(X_train)
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train first!")
        
        os.makedirs('models/saved_models', exist_ok=True)
        
        joblib.dump(self.model, 'models/saved_models/air_quality_model.pkl')
        joblib.dump(self.scaler, 'models/saved_models/feature_scaler.pkl')
        
        # Save feature names
        with open('models/saved_models/feature_names.txt', 'w') as f:
            f.write(','.join(self.feature_names))
        
        print("✅ Model saved successfully!")
    
    def predict(self, temperature, humidity, pressure, wind_speed, hour, day_of_week):
        """Make a prediction"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input
        X_new = np.array([[temperature, humidity, pressure, wind_speed, hour, day_of_week]])
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict
        prediction = self.model.predict(X_new_scaled)[0]
        
        return max(0, prediction)  # PM2.5 can't be negative

# Test training
# if __name__ == "__main__":
#     predictor = AirQualityPredictor()
#     results = predictor.train_model()
    
#     if results:
#         predictor.save_model()
        
#         # Test prediction
#         pred = predictor.predict(
#             temperature=20.0,
#             humidity=60,
#             pressure=1013,
#             wind_speed=5.0,
#             hour=14,
#             day_of_week=1
#         )
#         print(f"\n🔮 Sample prediction: {pred:.2f} μg/m³ PM2.5")