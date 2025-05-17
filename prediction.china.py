import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Enable the FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

#Load FastF1 2024 Australian GP race data
session_2024 = fastf1.get_session(2024, "China", 'R')
session_2024.load()
print("Cache enabled and session data loaded.")


# Extracting lap times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col}"] = laps_2024[f"{col}"].dt.total_seconds()

print(laps_2024.head())

# Get Avg lap times for each driver
best_laps_2024 = laps_2024.groupby("Driver")["LapTime"].mean().reset_index()
best_laps_2024.columns = ["Driver", "AvgLapTime"]


# Group by driver to get average sector time per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]].mean().reset_index()
print(sector_times_2024.head())
# Filter Q3 times from the China qualifiers for 2025
session_2025 = fastf1.get_session(2025, "China", 'Q')
session_2025.load()
q_times_2025 = session_2025.laps[["Driver", "LapTime"]].copy()
q_times_2025.dropna(subset=["LapTime"], inplace=True)
# Convert lap times to seconds
q_times_2025["LapTime"] = q_times_2025["LapTime"].dt.total_seconds()
# Filter for best Q times for each driver
qualifying_time = q_times_2025.groupby("Driver")["LapTime"].min().sort_values().reset_index()
# Rename columns for clarity
qualifying_time.columns = ["Driver", "Qualifying Time"]
print(qualifying_time)



# Merge 2024 sector times with 2025 qualifying times
merged_data = qualifying_time.merge(sector_times_2024, on="Driver", how="left")
print(merged_data)

# Merge with Avg lap times from 2024
merged_data = merged_data.merge(best_laps_2024, on="Driver", how="left")

merged_data.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time", "AvgLapTime"], inplace=True)


X = merged_data[["Qualifying Time", "Sector1Time", "Sector2Time", "Sector3Time"]]
y = merged_data["AvgLapTime"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# Initialize the model
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, random_state=39)
# Train the model
model.fit(X_train, y_train)

# Predict using 2025 qualifying times and best sector times
predicted_race_times = model.predict(X)
# Add predictions to the DataFrame
merged_data["PredictedBestRaceLap2025"] = predicted_race_times
# Rank drivers based on predicted lap times
merged_data = merged_data.sort_values(by="PredictedBestRaceLap2025")

# Print final predictions
print("Predicted Lap Times for 2025 China GP Qualifying:")
print(merged_data[["Driver", "PredictedBestRaceLap2025"]])


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Error: {mse:.2f} seconds")
