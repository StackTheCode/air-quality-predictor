import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_coordinates(city, country_code="", limit=1):
    """Use OpenWeather Geocoding API to get lat/lon from city name."""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap API key not found. Please check your environment configuration.")
    
    if not city or not city.strip():
        raise ValueError("City name cannot be empty.")
    
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': f"{city.strip()},{country_code}",
        'limit': limit,
        'appid': OPENWEATHER_API_KEY
    }
    
    try:
        response = requests.get(geo_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError(f"Location '{city}' not found. Please check spelling and try again.")
        
        return data[0]["lat"], data[0]["lon"]
    
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out. Please check your internet connection.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Connection error. Please check your internet connection.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"API request failed: {str(e)}")
    except (KeyError, IndexError):
        raise ValueError(f"Invalid response from geocoding API for '{city}'.")

def get_weather(city, country_code=""):
    """Get weather data for a city"""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap API key not found.")
    
    try:
        lat, lon = get_coordinates(city, country_code)
        
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": data.get("name", city),
            "coords": {"lat": lat, "lon": lon},
            "weather": {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "temp_min": data["main"]["temp_min"],
                "temp_max": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg"),
                "description": data["weather"][0]["description"],
            },
        }
        
    except requests.exceptions.Timeout:
        raise ValueError("Weather API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Connection error while fetching weather data.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Weather API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid weather data received: {str(e)}")

def get_air_quality(city, country_code=""):
    """Get air quality data for a city"""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap API key not found.")
    
    try:
        lat, lon = get_coordinates(city, country_code)
        
        url = f"http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("list") or len(data["list"]) == 0:
            raise ValueError("No air quality data available for this location.")
        
        return {
            "city": city,
            "coords": {"lat": lat, "lon": lon},
            "air_quality": {
                "aqi": data["list"][0]["main"]["aqi"],
                "components": data["list"][0]["components"],
            }
        }
        
    except requests.exceptions.Timeout:
        raise ValueError("Air quality API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Connection error while fetching air quality data.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Air quality API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid air quality data received: {str(e)}")