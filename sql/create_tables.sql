CREATE TABLE IF NOT EXISTS air_quality_table(
    id SERIAL PRIMARY KEY,
   timestamp TIMESTAMP NOT NULL,
   city VARCHAR(100) NOT NULL,
   country VARCHAR(50) NOT NULL,
   pm25 FLOAT,
   pm10 FLOAT,
   aqi INTEGER,
   co FLOAT,
   no2 FLOAT,
   so2 FLOAT,
   o3 FLOAT,
   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weather_data_table (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    city VARCHAR(100) NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


