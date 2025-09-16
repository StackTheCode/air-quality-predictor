import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import get_engine

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_stats = {}
    
    def prepare_data(self):
        """Load and prepare data with enhanced feature engineering"""
        engine = get_engine()
        
        # Enhanced query with better data filtering
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
        WHERE a.pm25 IS NOT NULL 
            AND a.pm25 > 0 
            AND a.pm25 < 500  -- Remove unrealistic values
            AND w.temperature > -50 
            AND w.temperature < 60
            AND w.humidity BETWEEN 0 AND 100
            AND w.pressure > 900 
            AND w.pressure < 1100
            AND w.wind_speed >= 0 
            AND w.wind_speed < 50
        ORDER BY a.timestamp
        """
        
        df = pd.read_sql(query, engine)
        
        if len(df) < 50:
            raise ValueError(f"Not enough data to train model! Found {len(df)} records. Collect more data first!")
        
        print(f"📊 Loaded {len(df)} data points for training")
        
        # Enhanced feature engineering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].apply(self._get_season)
        
        # Weather-based features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['pressure_normalized'] = (df['pressure'] - 1013) / 50  # Normalize around sea level
        df['wind_calm'] = (df['wind_speed'] < 2).astype(int)  # Calm wind indicator
        
        # Time-based features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
        df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
        
        # Features for prediction (choose best performing ones)
        feature_cols = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 
            'hour', 'day_of_week',
            # Optional: uncomment if you want more features
            # 'temp_humidity_interaction', 'pressure_normalized', 
            # 'wind_calm', 'is_weekend', 'is_rush_hour'
        ]
        
        X = df[feature_cols].copy()
        y = df['pm25'].copy()
        
        # Remove outliers using IQR method
        X, y = self._remove_outliers(X, y)
        
        self.feature_names = feature_cols
        print(f"✅ Using features: {feature_cols}")
        print(f"📊 After cleaning: {len(X)} data points")
        
        return X, y, df
    
    def _get_season(self, month):
        """Convert month to season (0-3)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def _remove_outliers(self, X, y, method='iqr'):
        """Remove outliers to improve model quality"""
        initial_count = len(X)
        
        if method == 'iqr':
            # Remove outliers in target variable
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            
            # Keep values within 1.5 * IQR
            outlier_mask = (y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))
            
            X_clean = X[outlier_mask].copy()
            y_clean = y[outlier_mask].copy()
            
            removed_count = initial_count - len(X_clean)
            if removed_count > 0:
                print(f"🧹 Removed {removed_count} outliers ({removed_count/initial_count*100:.1f}%)")
            
            return X_clean, y_clean
        
        return X, y
    
    def train_model(self):
        """Train optimized model with better hyperparameters"""
        try:
            X, y, df = self.prepare_data()
            
            # Store training statistics for confidence calculation
            self.training_stats = {}
            for col in X.columns:
                self.training_stats[col] = {
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'median': float(X[col].median())
                }
            
            # Split data with stratification attempt
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            print(f"📊 Training set: {len(X_train)} samples")
            print(f"📊 Test set: {len(X_test)} samples")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Optimized Random Forest parameters based on dataset size
            n_samples = len(X_train)
            
            if n_samples < 100:
                n_estimators = 50
                max_depth = 8
                min_samples_split = 5
                min_samples_leaf = 3
            elif n_samples < 500:
                n_estimators = 100
                max_depth = 12
                min_samples_split = 4
                min_samples_leaf = 2
            else:
                n_estimators = 150
                max_depth = 15
                min_samples_split = 3
                min_samples_leaf = 1
            
            # Train Random Forest with optimized parameters
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features='sqrt',  # Good for regression
                random_state=42,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1  # Use all CPU cores
            )
            
            print("🤖 Training model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"\n✅ Model trained successfully!")
            print(f"📊 Training MAE: {train_mae:.2f} μg/m³")
            print(f"📊 Test MAE: {test_mae:.2f} μg/m³")
            print(f"📊 Training RMSE: {train_rmse:.2f} μg/m³")
            print(f"📊 Test RMSE: {test_rmse:.2f} μg/m³")
            print(f"📊 Training R²: {train_r2:.3f}")
            print(f"📊 Test R²: {test_r2:.3f}")
            print(f"📊 Cross-validation MAE: {cv_mae:.2f} ± {cv_std:.2f}")
            print(f"📊 OOB Score: {self.model.oob_score_:.3f}")
            
            # Feature importance analysis
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Feature Importance:")
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            # Model quality assessment
            quality_score = self._assess_model_quality(test_mae, test_r2, cv_mae, cv_std)
            print(f"\n🎯 Model Quality: {quality_score}")
            
            return {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'oob_score': self.model.oob_score_,
                'feature_importance': importance,
                'data_points': len(X_train),
                'quality_score': quality_score
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def _assess_model_quality(self, test_mae, test_r2, cv_mae, cv_std):
        """Assess overall model quality"""
        if test_mae < 5 and test_r2 > 0.8:
            return "🟢 Excellent"
        elif test_mae < 8 and test_r2 > 0.7:
            return "🟡 Good"
        elif test_mae < 12 and test_r2 > 0.5:
            return "🟠 Fair"
        else:
            return "🔴 Needs Improvement"
    
    def save_model(self):
        """Save model with all components needed for prediction"""
        if self.model is None:
            raise ValueError("No model to save. Train first!")
        
        # Create directory
        os.makedirs('models/saved_models', exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, 'models/saved_models/air_quality_model.pkl')
        joblib.dump(self.scaler, 'models/saved_models/feature_scaler.pkl')
        
        # Save feature names
        with open('models/saved_models/feature_names.txt', 'w') as f:
            f.write(','.join(self.feature_names))
        
        # Save training statistics for confidence calculation
        with open('models/saved_models/training_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Save model metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'oob_score': float(self.model.oob_score_) if hasattr(self.model, 'oob_score_') else None,
            'features': self.feature_names,
            'training_samples': len(self.training_stats)
        }
        
        with open('models/saved_models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model and all components saved successfully!")
        print(f"📁 Saved to: models/saved_models/")
    
    def predict_sample(self, temperature, humidity, pressure, wind_speed, hour, day_of_week):
        """Test prediction with sample data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input
        X_new = np.array([[temperature, humidity, pressure, wind_speed, hour, day_of_week]])
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict
        prediction = self.model.predict(X_new_scaled)[0]
        
        return max(0, prediction)

# Example usage and testing
if __name__ == "__main__":
    predictor = AirQualityPredictor()
    
    print("🚀 Starting model training...")
    results = predictor.train_model()
    
    if results:
        predictor.save_model()
        
        print(f"\n🔮 Testing prediction...")
        # Test with sample data
        sample_pred = predictor.predict_sample(
            temperature=20.0,
            humidity=60,
            pressure=1013,
            wind_speed=5.0,
            hour=14,
            day_of_week=1
        )
        print(f"Sample prediction: {sample_pred:.2f} μg/m³ PM2.5")
        
        # Quality recommendations
        print(f"\n💡 Recommendations:")
        if results['test_mae'] > 10:
            print("- Collect more diverse training data")
            print("- Check data quality and remove more outliers")
        if results['test_r2'] < 0.6:
            print("- Consider adding more weather features")
            print("- Try different model parameters")
        if results['data_points'] < 200:
            print("- Aim for at least 200+ training samples for better accuracy")
        
        print(f"\n🎯 Expected confidence range: 60-85%")
    else:
        print("❌ Training failed. Check your database and data quality.")