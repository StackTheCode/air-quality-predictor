🌬️ Air Quality Predictor

This is a  web app that helps you check and predict the air quality (AQI) in different cities.
It combines live weather and pollution data with a machine learning model I trained, so you can see not just the current AQI, but also a predicted value with a confidence score.

The app is built with Streamlit for the UI, uses PostgreSQL to store historical data, and a RandomForest model (from scikit-learn) for predictions.
It’s simple to use: type a city, hit a button, and you’ll get the current and predicted air quality, along with health recommendations.

🚀 Features

📡 Live Data Fetching – Collects air quality & weather data from external APIs.

🤖 ML Prediction – RandomForest model trained on environmental features.

📊 Confidence Scoring – Multi-factor confidence calculation (tree agreement, feature ranges, prediction spread, time-based).

🗄️ Database Integration – Stores AQI, weather, and prediction history in PostgreSQL.

📱 Interactive Web UI – Streamlit interface with quick actions:

Share Location

Set AQI Alerts

View Historical Trends

🔍 Explainability – Each prediction comes with category, health impact, and confidence level.

🛠️ Tech Stack

Frontend/UI: Streamlit

Backend/DB: PostgreSQL + SQLAlchemy

Machine Learning: scikit-learn (RandomForest)

Data Handling: Pandas, NumPy



