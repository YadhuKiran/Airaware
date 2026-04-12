import pandas as pd
import numpy as np
import os

# Copy these functions directly from your main code
CSV_PATH = "intelligent_indoor_environment_dataset.csv"
ROOM_OCCUPANT_MAP = {
    "Room 101": [1, 2, 3],
    "Room 102": [4, 5, 6],
    "Room 103": [7, 8, 9, 10],
}
ROOMS = list(ROOM_OCCUPANT_MAP.keys())
CSV_COL = {
    "aqi": "room_air_quality",
    "temperature": "room_temperature",
    "humidity": "room_humidity",
    "co2": "room_CO2",
}

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
room_labels = {}
for room, occ_ids in ROOM_OCCUPANT_MAP.items():
    for oid in occ_ids:
        room_labels[oid] = room
df["room"] = df["occupant_id"].map(room_labels)

features = ["room_air_quality", "room_temperature", "room_humidity", "room_CO2"]
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Build sequential pairs
X_list, y_list = [], []
for room in ROOMS:
    room_df = df[df["room"] == room].sort_values("timestamp").reset_index(drop=True)
    grouped = room_df.groupby("timestamp")[features].mean().sort_index().reset_index()
    for i in range(len(grouped) - 1):
        X_list.append(grouped.iloc[i][features].values.astype(np.float64))
        y_list.append(grouped.iloc[i+1][features].values.astype(np.float64))

X = np.array(X_list)
y = np.array(y_list)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Add bias column
def design_matrix(X):
    return np.hstack([np.ones((len(X), 1)), X])

Xd_train = design_matrix(X_train)
Xd_test  = design_matrix(X_test)

# Train ridge regression
ridge = 0.15
W = np.linalg.solve(Xd_train.T @ Xd_train + ridge * np.eye(5), Xd_train.T @ y_train)

# Evaluate
pred = Xd_test @ W
err  = y_test - pred
mae  = np.mean(np.abs(err), axis=0)
rmse = np.sqrt(np.mean(err**2, axis=0))

labels = ["AQI", "Temperature (°C)", "Humidity (%)", "CO2 (ppm)"]
print(f"{'Target':<20} {'MAE':>10} {'RMSE':>10}")
print("-" * 42)
for i, label in enumerate(labels):
    print(f"{label:<20} {mae[i]:>10.4f} {rmse[i]:>10.4f}")

print(f"\nTraining samples : {split}")
print(f"Test samples     : {len(X) - split}")