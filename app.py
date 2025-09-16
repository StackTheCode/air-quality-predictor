import streamlit as st
import sys
import os
from datetime import datetime
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data functions (assuming they are in these paths)
try:
    from data.database_manager import store_air_quality_data, store_weather_data, store_prediction
    from models.predictor import AirQualityModel
    from data.collector import get_weather, get_air_quality
    from config.database import test_connection
except ImportError as e:
    st.error(f"Failed to import a necessary module: {e}. Please check your file structure and Python path.")
    st.stop()

# --- Page Configuration and CSS ---
st.set_page_config(
    page_title="Air Quality Monitor", 
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; margin-bottom: 2rem; }
    .metric-container { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .city-info { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🌬️ Air Quality Monitor")
st.markdown("**Real-time air quality data and predictions for cities worldwide**")
st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes")
    show_detailed = st.checkbox("Show detailed metrics", value=True)
    
    st.header("ℹ️ About")
    st.write("""
    This app provides real-time air quality data and ML-powered predictions.
    Data sources: OpenWeatherMap API
    """)

# --- Database Status ---
with st.container():
    col_status, col_time = st.columns([3, 1])
    with col_status:
        try:
            if test_connection():
                st.success("✅ Database connected")
            else:
                st.error("❌ Database connection failed")
        except Exception as e:
            st.error(f"❌ Database error: {str(e)[:50]}...")
    
    with col_time:
        st.write(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# --- Cache Data Fetching ---
@st.cache_data(ttl=300)
def fetch_live_data(city_name, country_code):
    """Fetches weather and air quality data from APIs."""
    try:
        weather_data = get_weather(city_name, country_code)
        air_data = get_air_quality(city_name, country_code)
        return weather_data, air_data, None
    except Exception as e:
        return None, None, str(e)

# --- Session State Initialization ---
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'air_data' not in st.session_state:
    st.session_state.air_data = None
if 'error' not in st.session_state:
    st.session_state.error = None

# --- Location Selection ---
st.subheader("🌍 Location Selection")
col_city, col_country, col_search = st.columns([2, 1, 1])

with col_city:
    city = st.text_input("City name", value="London", help="Enter the name of the city you want to monitor")

with col_country:
    country = st.text_input("Country code", value="GB", help="2-letter country code (e.g., US, GB, FR)", max_chars=2)

with col_search:
    st.write("")
    search_button = st.button("🔍 Get Air Quality", type="primary")

# --- Manual & Auto-refresh logic ---
col_refresh, col_clear = st.columns([1, 1])

with col_refresh:
    if st.button("🔄 Refresh Data"):
        st.session_state.last_refresh = None
        st.rerun()

with col_clear:
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Check for auto-refresh
if auto_refresh and st.session_state.last_refresh and (time.time() - st.session_state.last_refresh > 300):
    st.session_state.last_refresh = None
    st.rerun()

# --- Data Fetching and Display ---
st.divider()

# Only fetch if a button is clicked or auto-refresh is triggered
if search_button or st.session_state.last_refresh is None:
    with st.spinner(f"🔍 Fetching data for {city}, {country}..."):
        st.session_state.weather_data, st.session_state.air_data, st.session_state.error = fetch_live_data(city, country)
        st.session_state.last_refresh = time.time()
        
        # Store data in the database (separate from the cached fetch)
        if st.session_state.weather_data and st.session_state.air_data:
            try:
                store_weather_data(city, country)
                store_air_quality_data(city, country)
            except Exception as e:
                st.warning(f"Failed to store data in the database: {e}")

# Display based on session state
weather_data = st.session_state.weather_data
air_data = st.session_state.air_data
error = st.session_state.error

if error:
    if "api key" in error.lower():
        st.error("🔑 **API Key Issue**: Please check your .env file and ensure your OpenWeatherMap API key is valid.")
    elif "not found" in error.lower():
        st.error(f"🗺️ **Location Not Found**: Could not find '{city}, {country}'. Please check the spelling and country code.")
    elif "network" in error.lower() or "timeout" in error.lower():
        st.error("🌐 **Network Issue**: Please check your internet connection and try again.")
    else:
        st.error(f"❌ **Error**: {error}")
    
    st.subheader("📊 Air Quality Metrics (No Data)")
    cols = st.columns(4)
    for col in cols:
        with col: st.metric("—", "—")

elif weather_data and air_data:
    # Location header
    coords = weather_data["coords"]
    st.markdown(f"""
    <div class="city-info">
        <h3>📍 {weather_data['city']}, {country.upper()}</h3>
        <p><strong>Coordinates:</strong> {coords['lat']:.4f}, {coords['lon']:.4f}</p>
        <p><strong>Weather:</strong> {weather_data['weather']['description'].title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main metrics
    st.subheader("📊 Current Air Quality & Weather")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        aqi_levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
        aqi_colors = ["🟢", "🟡", "🟠", "🔴", "🟣"]
        aqi = air_data["air_quality"]["aqi"]
        aqi_color = aqi_colors[aqi-1]
        st.metric("Air Quality Index", f"{aqi_color} {aqi}", aqi_levels[aqi], help="1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor")
    
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
                temperature=weather_data["weather"]["temperature"], humidity=weather_data["weather"]["humidity"],
                pressure=weather_data["weather"]["pressure"], wind_speed=weather_data["weather"]["wind_speed"],
                hour=now.hour, day_of_week=now.weekday()
            )
            def pm25_to_category(value):
                if value <= 12: return "🟢 Good"
                elif value <= 35: return "🟡 Moderate"
                elif value <= 55: return "🟠 Unhealthy (Sensitive)"
                elif value <= 150: return "🔴 Unhealthy"
                else: return "🟣 Very Unhealthy"
            
            st.metric("Predicted PM2.5", f"{pred_pm25:.1f} μg/m³", f"{pm25_to_category(pred_pm25)} ({confidence:.1%} conf.)", help="ML prediction based on current weather conditions")
            store_prediction(city, pred_pm25, confidence_score=confidence)
        except Exception as e:
            st.metric("Prediction", "Unavailable", f"Error: {str(e)[:20]}...")
    
    # Detailed information (conditional)
    if show_detailed:
        st.divider()
        st.subheader("📈 Detailed Measurements")
        col_weather, col_air, col_trend = st.columns(3)
        with col_weather:
            st.markdown("**🌤️ Weather Data**")
            weather = weather_data["weather"]
            st.write(f"**Pressure:** {weather['pressure']} hPa")
            st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
            st.write(f"**Visibility:** Good")
        with col_air:
            st.markdown("**💨 Air Pollutants**")
            components = air_data["air_quality"]["components"]
            st.write(f"**PM10:** {components['pm10']:.1f} μg/m³")
            st.write(f"**NO2:** {components['no2']:.1f} μg/m³")
            st.write(f"**CO:** {components['co']:.1f} μg/m³")
            st.write(f"**O3:** {components['o3']:.1f} μg/m³")
        with col_trend:
            st.markdown("**📊 Health Impact**")
            if aqi <= 2:
                st.success("✅ Air quality is satisfactory")
                st.write("Safe for outdoor activities")
            elif aqi == 3:
                st.warning("⚠️ Moderate air quality")
                st.write("Sensitive individuals should limit outdoor activities")
            else:
                st.error("🚨 Poor air quality")
                st.write("Avoid outdoor activities, especially exercise")
    
    # Quick actions
    st.divider()
    col_action1, col_action2, col_action3 = st.columns(3)
    with col_action1:
        if st.button("📱 Share Location"):
            st.info(f"Share this: {city}, {country} - AQI: {aqi} ({aqi_levels[aqi]})")
    with col_action2:
        if st.button("⚠️ Set Alert"):
            st.info("Alert feature coming soon!")
    with col_action3:
        if st.button("📊 View History"):
            st.info("Historical data feature coming soon!")

else:
    st.info("👆 Enter a city and click **Get Air Quality** to fetch data")

# Footer with development status
st.divider()
st.subheader("🚀 Development Progress")
progress_items = [
    ("✅", "Database connection", True),
    ("✅", "Live API integration", True),
    ("✅", "Data storage", True),
    ("✅", "ML predictions", True),
    ("🔄", "Interactive dashboard", False),
    ("🔄", "Historical trends", False),
    ("🔄", "Cloud deployment", False),
]
cols = st.columns(len(progress_items))
for i, (icon, item, completed) in enumerate(progress_items):
    with cols[i]:
        if completed:
            st.success(f"{icon} {item}")
        else:
            st.info(f"{icon} {item}")

# Performance note
if st.checkbox("Show Performance Info", help="Show technical details"):
    st.info(f"""
    **Cache Status:** Data cached for 5 minutes to reduce API calls
    **Last Cache Clear:** {st.session_state.get('last_refresh', 'Never')}
    **Auto-refresh:** {'Enabled' if auto_refresh else 'Disabled'}
    """)