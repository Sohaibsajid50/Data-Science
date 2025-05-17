import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Enable the FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

#Load FastF1 2024 Australian GP race data
session_2024 = fastf1.get_session(2024, "Australia", 'R')
session_2024.load()
print("Cache enabled and session data loaded.")


# Extracting lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime"] = laps_2024["LapTime"].dt.total_seconds()
print(laps_2024.head())


# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russel", "Yuki Tsunoda", "Alexander Albon",
                "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz", "Isack Hadjar", "Fernando Alonso"],
    "Qualifying Time": [75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5]
})
driver_mapping = {
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Max Verstappen": "VER",
    "George Russel": "RUS",
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI",
    "Isack Hadjar": "HAD",
    "Fernando Alonso": "ALO"
}
# Map driver names to codes 
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
print(qualifying_2025)

# Merge 2025 qualifying data with 2024 lap times
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")
print(merged_data)


X = merged_data[["Qualifying Time"]]
y = merged_data["LapTime"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# Initialize the model
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, random_state=39)
# Train the model
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["Qualifying Time"]])
# Add predictions to the DataFrame
qualifying_2025["Predicted Lap Time"] = predicted_lap_times
# Rank drivers based on predicted lap times
qualifying_2025 = qualifying_2025.sort_values(by="Predicted Lap Time")

# Print final predictions
print("Predicted Lap Times for 2025 Australia GP Qualifying:")
print(qualifying_2025[["Driver", "Predicted Lap Time"]])

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Error: {mse:.2f} seconds")
