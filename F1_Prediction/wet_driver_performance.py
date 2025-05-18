import fastf1
import pandas as pd

# Enable the FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

# load 2023 Canadian GP (wet race)
session_2023 = fastf1.get_session(2023, "Canada", 'R')
session_2023.load()

# Load 2022 Canadian GP (dry race)
session_2022 = fastf1.get_session(2022, "Canada", 'R')
session_2022.load()

# Extracting lap times both wet and dry races
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()

# Drop rows with NaN values
laps_2023.dropna(subset=["LapTime"], inplace=True)
laps_2022.dropna(subset=["LapTime"], inplace=True)

# Convert lap times to seconds
laps_2023["LapTime"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime"] = laps_2022["LapTime"].dt.total_seconds()

# Calculate average lap times for each driver
avg_lap_times_2023 = laps_2023.groupby("Driver")["LapTime"].mean().reset_index()
avg_lap_times_2022 = laps_2022.groupby("Driver")["LapTime"].mean().reset_index()

# merge the two datasets on driver
merged_data = avg_lap_times_2023.merge(avg_lap_times_2022, on="Driver", suffixes=('_Dry', '_Wet'))

# Calculate the performance difference in lap times between wet and dry conditions
merged_data["Performance_Difference"] = merged_data["LapTime_Wet"] - merged_data["LapTime_Dry"]

# Calculate the percentage change in lap times between wet and dry conditions
merged_data["Percentage_Change"] = (merged_data["Performance_Difference"] / merged_data["LapTime_Dry"]) * 100

# Now we create wet performance scores based on the percentage change
merged_data["Wet_Performance_Score"] = 1 - (merged_data["Percentage_Change"] / 100)


# Print values to verify the formula change
print(merged_data[["Driver", "LapTime_Dry", "LapTime_Wet", "Wet_Performance_Score", "Percentage_Change"]])
# Prrint out the wet performance scores for each driver
print("\nDriver Wet Performance Scores (2023 vs 2022):")
print(merged_data[["Driver", "Wet_Performance_Score"]])

