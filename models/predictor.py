import joblib
import numpy as np
import os
MODEL_DIR =  os.path.join("models", "saved_models")

class AirQualityModel:
    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "air_quality_model.pkl")
        scaler_path = os.path.join(MODEL_DIR,"feature_scaler.pkl")
        features_path = os.path.join(MODEL_DIR, "feature_names.txt")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path,"r") as f :
            self.feature_names = f.read().split(",")
    
    def predict(self, temperature,humidity,pressure,wind_speed,hour,day_of_week):
        X_new = np.array([[temperature,humidity,pressure,wind_speed,hour,day_of_week]])
        X_scaled = self.scaler.transform(X_new)
        prediction = self.model.predict(X_scaled)[0]
        return max(0,prediction)
    
    def predict_with_confidence(self, temperature, humidity, pressure, wind_speed, hour, day_of_week):
        X_new = np.array([[temperature, humidity, pressure, wind_speed, hour, day_of_week]])
        X_scaled = self.scaler.transform(X_new)

        # Predictions from all trees
        all_preds = np.array([est.predict(X_scaled)[0] for est in self.model.estimators_])

        mean_pred = all_preds.mean()
        std_dev = all_preds.std()

        # Confidence = 1 - relative std deviation
        confidence = max(0.0, 1.0 - (std_dev / mean_pred)) if mean_pred > 0 else 0.0

        return max(0, mean_pred), round(confidence, 3)