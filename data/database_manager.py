import sys
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now imports will work
from config.database import get_engine
from data.collector import get_weather, get_air_quality


def store_prediction(city, predicted_pm25, model_version="RandomForest_v1",confidence_score=None):
    """Insert prediction into the predictions table"""
    try:
      engine = get_engine()
      now = datetime.now()

      query = text("""
         INSERT INTO predictions (timestamp, city, predicted_aqi, confidence_score, model_version)
         VALUES (:timestamp, :city, :predicted_aqi, :confidence_score, :model_version)
      """)

      with engine.begin() as conn:
         conn.execute(query, {
             "timestamp": now,
             "city": city,
             "predicted_aqi": int(predicted_pm25),   # store rounded AQI
             "confidence_score": float(confidence_score) if confidence_score is not None else None,              # placeholder (can add later)
             "model_version": model_version
            })
    except Exception as e:
        # Log error but don't crash the app
        logger.error(f"Failed to store prediction for {city}: {str(e)}")



def fetch_history(city,country,limit: int =10):
   try: 
      engine = get_engine()
      query = text("""
          SELECT id, timestamp, city, country, pm25, pm10, aqi, co, no2, so2, o3, created_at
          FROM air_quality_table
          WHERE city = :city AND country = :country
          ORDER BY timestamp DESC
          LIMIT :limit
      """)
      with engine.connect() as conn:
          rows = conn.execute(query, {"city": city, "country": country, "limit": limit}).fetchall()
          return [dict(row._mapping) for row in rows] if rows else []
   except Exception as e:
       return []



def store_air_quality_data(city, country_code="GB"):
    """Store current air quality data in database"""
    try:
        air_data = get_air_quality(city, country_code)
        engine = get_engine()
        
        # Prepare data for database
        data = {
            'timestamp': datetime.now(),
            'city': air_data['city'],
            'country': country_code,
            'pm25': air_data['air_quality']['components']['pm2_5'],
            'pm10': air_data['air_quality']['components']['pm10'],
            'aqi': air_data['air_quality']['aqi'],
            'co': air_data['air_quality']['components']['co'],
            'no2': air_data['air_quality']['components']['no2'],
            'so2': air_data['air_quality']['components']['so2'],
            'o3': air_data['air_quality']['components']['o3']
        }
        
        # Insert into database
        with engine.connect() as conn:
            query = text("""
                INSERT INTO air_quality_table 
                (timestamp, city, country, pm25, pm10, aqi, co, no2, so2, o3)
                VALUES (:timestamp, :city, :country, :pm25, :pm10, :aqi, :co, :no2, :so2, :o3)
            """)
            conn.execute(query, data)
            conn.commit()
            
    except Exception as e:
        logger.error(f"Failed to store air quality data for {city}: {str(e)}")

def store_weather_data(city, country_code="GB"):
    """Store current weather data in database"""
    try:
        weather_data = get_weather(city, country_code)
        engine = get_engine()
        
        # Prepare data for database
        data = {
            'timestamp': datetime.now(),
            'city': weather_data['city'],
            'temperature': weather_data['weather']['temperature'],
            'humidity': weather_data['weather']['humidity'],
            'pressure': weather_data['weather']['pressure'],
            'wind_speed': weather_data['weather']['wind_speed'],
            'wind_direction': 0  # Default since not in all responses
        }
        
        # Insert into database
        with engine.connect() as conn:
            query = text("""
                INSERT INTO weather_data_table
                (timestamp, city, temperature, humidity, pressure, wind_speed, wind_direction)
                VALUES (:timestamp, :city, :temperature, :humidity, :pressure, :wind_speed, :wind_direction)
            """)
            conn.execute(query, data)
            conn.commit()
            
    except Exception as e:
        logger.error(f"Failed to store weather data for {city}: {str(e)}")
