import streamlit as st
import sys
import os

from data.database_manager import store_air_quality_data, store_weather_data

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Air Quality Predictor", 
    page_icon="🌬️",
    layout="wide"
)

st.title("🌬️ Air Quality Predictor")
st.write("Predicting air quality for better health decisions")

# Test database connection
try:
    from config.database import test_connection
    if test_connection():
        st.success("✅ Database connected successfully!")
    else:
        st.error("❌ Database connection failed!")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
except Exception as e:
    st.error(f"❌ Error: {e}")

# Import data collector
try:
    from data.collector import get_weather, get_air_quality
    
    # City selection
    col_city1, col_city2 = st.columns([2, 1])
    with col_city1:
        city = st.text_input("Enter city name:", value="London")
    with col_city2:
        country = st.text_input("Country code:", value="GB")
    
    # Cache data for 5 minutes to avoid excessive API calls
    @st.cache_data(ttl=300)
    def get_live_data(city_name, country_code):
        try:
            weather_data = get_weather(city_name, country_code)
            air_data = get_air_quality(city_name, country_code)
            store_weather_data(city_name, country_code)
            store_air_quality_data(city_name, country_code)

            return weather_data, air_data, None
        except Exception as e:
            return None, None, str(e)

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Get live data
    weather_data, air_data, error = get_live_data(city, country)
    
    if error:
        st.error(f"❌ Failed to fetch data: {error}")
        # Show placeholder data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current AQI", "—", "No data")
        with col2:
            st.metric("PM2.5", "— μg/m³", "No data")
        with col3:
            st.metric("Temperature", "— °C", "No data")
    
    elif weather_data and air_data:
        # Display real data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            aqi_levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
            aqi = air_data["air_quality"]["aqi"]
            aqi_color = ["🟢", "🟡", "🟠", "🔴", "🟣"][aqi-1]
            st.metric("Air Quality Index", f"{aqi_color} {aqi}", aqi_levels[aqi])
        
        with col2:
            pm25 = air_data["air_quality"]["components"]["pm2_5"]
            st.metric("PM2.5", f"{pm25:.1f} μg/m³")
        
        with col3:
            temp = weather_data["weather"]["temperature"]
            st.metric("Temperature", f"{temp:.1f}°C")
        
        # Additional data in expandable section
        with st.expander("📊 Detailed Information"):
            col_weather, col_air = st.columns(2)
            
            with col_weather:
                st.subheader("🌤️ Weather Details")
                weather = weather_data["weather"]
                st.write(f"**Condition:** {weather['description'].title()}")
                st.write(f"**Humidity:** {weather['humidity']}%")
                st.write(f"**Pressure:** {weather['pressure']} hPa")
                st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
            
            with col_air:
                st.subheader("💨 Air Quality Components")
                components = air_data["air_quality"]["components"]
                st.write(f"**PM10:** {components['pm10']:.1f} μg/m³")
                st.write(f"**NO2:** {components['no2']:.1f} μg/m³")
                st.write(f"**CO:** {components['co']:.1f} μg/m³")
                st.write(f"**O3:** {components['o3']:.1f} μg/m³")
        
        # Location info
        coords = weather_data["coords"]
        st.info(f"📍 **{weather_data['city']}** - Lat: {coords['lat']:.4f}, Lon: {coords['lon']:.4f}")
    
    else:
        st.warning("⏳ Loading data...")

except ImportError as e:
    st.error(f"❌ Could not import data collector: {e}")
    st.info("Make sure your data/collector.py file exists and your API key is set in .env")
    
    # Fallback to original hardcoded data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current AQI", "52", "Good")
    with col2:
        st.metric("PM2.5", "15.2 μg/m³", "-2.1")
    with col3:
        st.metric("Prediction", "68", "Moderate")

st.markdown("---")
st.subheader("🚀 Next Steps")
st.write("""
1. ✅ Database connection established
2. ✅ Live API data integration
3. 🔄 Store data in PostgreSQL
4. 🔄 Build ML prediction model
5. 🔄 Create interactive dashboard
6. 🔄 Deploy to cloud
""")