import streamlit as st
import sys
import os
from datetime import datetime
import time
import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.database_manager import store_air_quality_data, store_weather_data, store_prediction, fetch_history
    from models.predictor import AirQualityModel
    from data.collector import get_weather, get_air_quality
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Air Quality Monitor", 
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.title("🌬️ Air Quality Monitor")
st.markdown("**Real-time air quality data and predictions for cities worldwide**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes")
    show_detailed = st.checkbox("Show detailed metrics", value=True)
    
    st.header("ℹ️ About")
    st.markdown("""
    This app provides real-time air quality data and ML-powered predictions.
    Data source: OpenWeatherMap API
    """)

# Last updated time
col_time = st.columns([1])[0]
with col_time:
    st.write(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")

st.divider()

@st.cache_data(ttl=300)
def fetch_live_data(city_name, country_code):
    """Fetches weather and air quality data from APIs."""
    try:
        weather_data = get_weather(city_name, country_code)
        air_data = get_air_quality(city_name, country_code)
        return weather_data, air_data, None
    except Exception as e:
        return None, None, str(e)

def get_aqi_category(aqi):
    """Get AQI category and color"""
    levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    colors = ["🟢", "🟡", "🟠", "🔴", "🟣"]
    return colors[aqi-1], levels[aqi]

def get_pm25_category(value):
    """Get PM2.5 health category"""
    if value <= 12: 
        return "🟢 Good"
    elif value <= 35: 
        return "🟡 Moderate"
    elif value <= 55: 
        return "🟠 Unhealthy (Sensitive)"
    elif value <= 150: 
        return "🔴 Unhealthy"
    else: 
        return "🟣 Very Unhealthy"

# Session State Initialization
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'air_data' not in st.session_state:
    st.session_state.air_data = None
if 'error' not in st.session_state:
    st.session_state.error = None

# Location Selection
st.subheader("🌍 Location Selection")
col_city, col_country, col_search = st.columns([2, 1, 1])

with col_city:
    city = st.text_input("City name", value="London", help="Enter the name of the city")

with col_country:
    country = st.text_input("Country code", value="GB", help="2-letter country code (e.g., US, GB, FR)", max_chars=2)

with col_search:
    st.write("")
    search_button = st.button("🔍 Get Air Quality", type="primary")

# Manual refresh and cache controls
col_refresh, col_clear = st.columns([1, 1])

with col_refresh:
    if st.button("🔄 Refresh Data"):
        st.session_state.last_refresh = None
        st.rerun()

with col_clear:
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Auto-refresh logic
if auto_refresh and st.session_state.last_refresh and (time.time() - st.session_state.last_refresh > 300):
    st.session_state.last_refresh = None
    st.rerun()

st.divider()

# Data Fetching and Display
if search_button:
    with st.spinner(f"🔍 Fetching data for {city}, {country}..."):
        st.session_state.weather_data, st.session_state.air_data, st.session_state.error = fetch_live_data(city, country)
        st.session_state.last_refresh = time.time()
        
        # Store data in database
        if st.session_state.weather_data and st.session_state.air_data:
            try:
                store_weather_data(city, country)
                store_air_quality_data(city, country)
            except Exception as e:
                st.warning(f"Failed to store data: {e}")

# Display results
weather_data = st.session_state.weather_data
air_data = st.session_state.air_data
error = st.session_state.error

if error:
    if "api key" in error.lower():
        st.error(" **API Key Issue**: Please check your API key configuration.")
    elif "not found" in error.lower():
        st.error(f"🗺️ **Location Not Found**: Could not find '{city}, {country}'. Please check the spelling and country code.")
    elif "network" in error.lower() or "timeout" in error.lower():
        st.error(" **Network Issue**: Please check your internet connection and try again.")
    else:
        st.error(f"❌ **Error**: {error}")

elif weather_data and air_data:
    # Location header
    coords = weather_data["coords"]
    st.markdown(f"""
    ### 📍 {weather_data['city']}, {country.upper()}
    **Coordinates:** {coords['lat']:.4f}, {coords['lon']:.4f}  
    **Weather:** {weather_data['weather']['description'].title()}
    """)
    
    # Main metrics
    st.subheader("📊 Current Air Quality & Weather")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        aqi = air_data["air_quality"]["aqi"]
        aqi_color, aqi_label = get_aqi_category(aqi)
        st.metric("Air Quality Index", f"{aqi_color} {aqi}", aqi_label, 
                 help="1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor")
    
    with col2:
        pm25 = air_data["air_quality"]["components"]["pm2_5"]
        pm25_delta = "Healthy" if pm25 <= 12 else "Monitor" if pm25 <= 35 else "Unhealthy"
        st.metric("PM2.5", f"{pm25:.1f} μg/m³", pm25_delta)
    
    with col3:
        temp = weather_data["weather"]["temperature"]
        humidity = weather_data["weather"]["humidity"]
        st.metric("Temperature", f"{temp:.1f}°C", f"{humidity}% humidity")
    
    with col4:
        try:
            predictor = AirQualityModel()
            now = datetime.now()
            pred_pm25, confidence = predictor.predict_with_confidence(
                temperature=temp, 
                humidity=humidity,
                pressure=weather_data["weather"]["pressure"], 
                wind_speed=weather_data["weather"]["wind_speed"],
                hour=now.hour, 
                day_of_week=now.weekday()
            )
            
            category = get_pm25_category(pred_pm25)
            st.metric("Predicted PM2.5", f"{pred_pm25:.1f} μg/m³", 
                     f"{category} ({confidence:.1%} conf.)", 
                     help="ML prediction based on current weather conditions")
            
            store_prediction(city, pred_pm25, confidence_score=confidence)
        except Exception as e:
            st.metric("Prediction", "Unavailable", f"Model error")
    
    # Detailed information
    if show_detailed:
        st.divider()
        st.subheader("📈 Detailed Measurements")
        col_weather, col_air, col_health = st.columns(3)
        
        with col_weather:
            st.markdown("**🌤️ Weather Data**")
            weather = weather_data["weather"]
            st.write(f"**Pressure:** {weather['pressure']} hPa")
            st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
            st.write(f"**Feels Like:** {weather['feels_like']:.1f}°C")
        
        with col_air:
            st.markdown("**💨 Air Pollutants**")
            components = air_data["air_quality"]["components"]
            st.write(f"**PM10:** {components['pm10']:.1f} μg/m³")
            st.write(f"**NO2:** {components['no2']:.1f} μg/m³")
            st.write(f"**CO:** {components['co']:.1f} μg/m³")
            st.write(f"**O3:** {components['o3']:.1f} μg/m³")
        
        with col_health:
            st.markdown("**📊 Health Impact**")
            if aqi <= 2:
                st.success("✅ Air quality is satisfactory")
                st.write("Safe for outdoor activities")
            elif aqi == 3:
                st.warning("⚠️ Moderate air quality")
                st.write("Sensitive individuals should limit prolonged outdoor activities")
            else:
                st.error("🚨 Poor air quality")
                st.write("Limit outdoor activities, especially exercise")
    
    # Quick Actions
    st.divider()
    col_action1, col_action2, col_action3 = st.columns(3)
    
    # Initialize session state for action buttons
    if "show_share" not in st.session_state:
        st.session_state.show_share = False
    if "show_alert" not in st.session_state:
        st.session_state.show_alert = False
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    
    with col_action1:
        if st.button("📱 Share Location"):
            st.session_state.show_share = not st.session_state.show_share
            st.session_state.show_alert = False
            st.session_state.show_history = False
        
        if st.session_state.show_share:
            aqi_color, aqi_label = get_aqi_category(aqi)
            share_text = f"{city}, {country} - AQI: {aqi} ({aqi_label}), PM2.5: {pm25:.1f} μg/m³"
            st.code(share_text, language="text")
            st.success(" Copy this text to share!")
    
    with col_action2:
        if st.button("⚠️ Set Alert"):
            st.session_state.show_alert = not st.session_state.show_alert
            st.session_state.show_share = False
            st.session_state.show_history = False
        
        if st.session_state.show_alert:
            threshold = st.number_input("Alert if AQI is above:", min_value=1, max_value=5, value=3)
            aqi_color, aqi_label = get_aqi_category(aqi)
            if aqi >= threshold:
                st.error(f"🚨 Alert! AQI is currently {aqi} ({aqi_label})")
            else:
                st.success(f"✅ AQI is {aqi} ({aqi_label}). Below your alert threshold.")

    with col_action3:
        if st.button("📊 View History"):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.show_share = False
            st.session_state.show_alert = False
        
        if st.session_state.show_history:
            try:
                history = fetch_history(city, country, limit=10)
                if history:
                    df = pd.DataFrame(history)
                    st.dataframe(df)
                else:
                    st.info("No historical data found for this location.")
            except Exception as e:
                st.error(f"Could not load history: {e}")

else:
    st.info("👆 Enter a city and click **Get Air Quality** to fetch data")