import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def get_coordinates(city, country_code="",limit="1"):
    
    """Use OpenWeather Geocoding API to get lat/lon from city name."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{country_code}&limit={limit}&appid={OPENWEATHER_API_KEY}"
    resp = requests.get(geo_url).json()
    if not resp:
        raise ValueError(f"❌ Could not find city: {city}")
    return resp[0]["lat"], resp[0]["lon"]


def get_weather(city, country_code=""):
 
    if not OPENWEATHER_API_KEY:
        raise ValueError("❌ OPENWEATHER_API_KEY not found in .env")

    lat, lon = get_coordinates(city, country_code)

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    resp = requests.get(url).json()

    return {
       "city": resp.get("name", city),
        "coords": {"lat": lat, "lon": lon},
        "weather": {
            "temperature": resp["main"]["temp"],
            "feels_like": resp["main"]["feels_like"],
            "temp_min": resp["main"]["temp_min"],
            "temp_max": resp["main"]["temp_max"],
            "humidity": resp["main"]["humidity"],
            "pressure": resp["main"]["pressure"],
            "wind_speed": resp["wind"]["speed"],
            "wind_deg": resp["wind"].get("deg"),
            "description": resp["weather"][0]["description"],
        },
    }

def get_air_quality(city,country_code):
      if not OPENWEATHER_API_KEY:
        raise ValueError("❌ OPENWEATHER_API_KEY not found in .env")
      lat,lon= get_coordinates(city,country_code)
      url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
      response = requests.get(url)
      
      if response.status_code!=200:
        raise ValueError(f"Air quality API error: {response.status_code}")
      data = response.json()
      
      return{
          "city":city,
          "coords":{"lat":lat,"lon": lon},
          "air_quality":{
              "aqi":data["list"][0]["main"]["aqi"],
              "components": data["list"][0]["components"],
          }
      }
    
        
if __name__ == "__main__":
    # Example test
    try:
        print("Testing weather API...")
        weather = get_weather("London", "GB")
        print("Weather data:", weather)
        
        print("\nTesting air quality API...")
        air = get_air_quality("London", "GB")
        print("Air quality data:", air)
        
    except Exception as e:
        print(f"❌ Error: {e}")