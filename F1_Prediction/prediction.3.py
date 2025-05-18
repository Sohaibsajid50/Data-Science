import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Enable the FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

#Load 2024 Japan GP race data
session_2024 = fastf1.get_session(2024, "Japan", 'R')
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col}"] = laps_2024[f"{col}"].dt.total_seconds()

# Group by driver to get average sector time per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]].mean().reset_index()


# 2025 Qualifying Data
session_2025 = fastf1.get_session(2025, "China", 'Q')
session_2025.load()
q_times_2025 = session_2025.laps[["Driver", "LapTime"]].copy()
q_times_2025.dropna(subset=["LapTime"], inplace=True)
# Convert lap times to seconds
q_times_2025["LapTime"] = q_times_2025["LapTime"].dt.total_seconds()
# Filter for best Q times for each driver
qualifying_2025 = q_times_2025.groupby("Driver")["LapTime"].min().sort_values().reset_index()
# Rename columns for clarity
qualifying_2025.columns = ["Driver", "Qualifying Time"]

# Merge 2024 sector times with 2025 qualifying times
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")


merged_data.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time"], inplace=True)


# Driver-Specific Wet Performance Data from Wet Performance based on 2022/2023 Canadian GP
driver_wet_performance = {
    "VER": 0.975196,     # 2.5% slower in wet conditions (slower lap times under rain conditions)
    "HAM": 0.976464,     # 2.4% slower in wet conditions (slower lap times in wet weather)
    "LEC": 0.975862,    # 2.5% slower in wet conditions (slightly slower lap times under rain)
    "NOR": 0.978179,       # 1% slower in wet conditions (minor impact of wet weather on performance)
    "ALO": 0.972655,    # 2.8% slower in wet conditions (significantly slower in wet conditions)
    "RUS": 0.966878,     # 3.3% slower in wet conditions (more affected by rain, slower lap times)
    "SAI": 0.978754,   # 1% slower in wet conditions (minor performance reduction in wet)
    "TSU": 0.996338,       # 0.8% slower in wet conditions (performs better in the wet than dry)
    "OCO": 0.981010,       # 1.9% slower in wet conditions (slightly worse lap times in wet conditions)
    "ALB": 0.978120,    # 2.3% slower in wet conditions (slower lap times in rain conditions)
    "GAS": 0.978832,       # 1.9% slower in wet conditions (performance impacted slightly by rain)
    "STR": 0.979857        # 1.4% slower in wet conditions (slightly slower lap times in the wet)
}

# Add Wet Performance Feature
merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(driver_wet_performance)

# Fill missing values with 1 (no performance loss in dry conditions)
merged_data["WetPerformanceFactor"].fillna(1, inplace=True)

# Forecasted Weather Data for Suzuka using OpenWeatherMap API
API_KEY = "7187df4dcfc990107404c4ffcd97295f"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Suzuka&appid={API_KEY}&units=metric"

# fetching weather data
response = requests.get(weather_url)
weather_data = response.json()

# Ectracting relevant weather data for the race on Sunday 2pm local time
forecast_time = "2024-10-06 14:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

# Extract weather conditions (rain probability, temperature)
if forecast_data:
    rain_probability = forecast_data["pop"]  # Probability of precipitation
    temperature = forecast_data["main"]["temp"]  # Temperature in Celsius
else:
    rain_probability = 0
    temperature = 20

# Add weather features to the merged data
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define the features and target variable
X = merged_data[["Qualifying Time", "Sector1Time", "Sector2Time", "Sector3Time", "WetPerformanceFactor", "RainProbability", "Temperature"]]
X.dropna(inplace=True)
# use the average lap time as the target variable
y = merged_data.merge(laps_2024.groupby("Driver")["LapTime"].mean().reset_index(), on="Driver", how="left")["LapTime"]
y.dropna(inplace=True)


# train gradient boosting model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Copy only the drivers that are in merged_data
qualifying_2025_filtered = qualifying_2025[qualifying_2025["Driver"].isin(merged_data["Driver"])]
qualifying_2025_filtered = qualifying_2025_filtered.reset_index(drop=True)

# Predict using 2025 qualifying times and best sector times
predicted_race_times = model.predict(X)
qualifying_2025_filtered["PredictedRaceTime"] = predicted_race_times

# Rank drivers by predicted race times
qualifying_2025 = qualifying_2025_filtered.sort_values(by="PredictedRaceTime")

# Print final predictions
print(" \n Predicted 2025 Japanese GP Winner \n")
print(qualifying_2025[["Driver", "PredictedRaceTime"]])

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Error: {mse:.2f} seconds")
