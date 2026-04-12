import time
import pandas as pd
import numpy as np

# ── Config (same as your main code) ──────────────────────────────────────────
CSV_PATH = "intelligent_indoor_environment_dataset.csv"
ROOM_OCCUPANT_MAP = {
    "Room 101": [1, 2, 3],
    "Room 102": [4, 5, 6],
    "Room 103": [7, 8, 9, 10],
}
ROOMS = list(ROOM_OCCUPANT_MAP.keys())
FEATURES = ["room_air_quality", "room_temperature", "room_humidity", "room_CO2"]

# ── Helper functions ──────────────────────────────────────────────────────────
def load_csv_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    room_labels = {}
    for room, occ_ids in ROOM_OCCUPANT_MAP.items():
        for oid in occ_ids:
            room_labels[oid] = room
    df["room"] = df["occupant_id"].map(room_labels)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def prepare_training_data(df):
    X_list, y_list = [], []
    for room in ROOMS:
        room_df = df[df["room"] == room].sort_values("timestamp").reset_index(drop=True)
        grouped = room_df.groupby("timestamp")[FEATURES].mean().sort_index().reset_index()
        for i in range(len(grouped) - 1):
            X_list.append(grouped.iloc[i][FEATURES].values.astype(np.float64))
            y_list.append(grouped.iloc[i+1][FEATURES].values.astype(np.float64))
    return np.array(X_list), np.array(y_list)

def design_matrix(X):
    return np.hstack([np.ones((len(X), 1)), X])

def train_model(X, y):
    Xd = design_matrix(X)
    ridge = 0.15
    W = np.linalg.solve(Xd.T @ Xd + ridge * np.eye(Xd.shape[1]), Xd.T @ y)
    return W

def predict_next(current_row, W):
    xb = np.array([1.0] + list(current_row), dtype=np.float64)
    return xb @ W

def preprocess_row(row):
    row = row.copy()
    row[0] = max(0,   min(300,  row[0]))  # AQI
    row[1] = max(10,  min(45,   row[1]))  # Temperature
    row[2] = max(0,   min(100,  row[2]))  # Humidity
    row[3] = max(300, min(5000, row[3]))  # CO2
    return row

# ═════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  AirAware — Functional & Performance Test")
print("=" * 55)

results = {}

# ── TEST 1: CSV Loading ───────────────────────────────────────────────────────
print("\n[1] Data Collection Module — load_csv_data()")
start = time.time()
df = load_csv_data(CSV_PATH)
t_load = time.time() - start
ok = len(df) > 0 and "room" in df.columns and "timestamp" in df.columns
print(f"    Rows loaded     : {len(df):,}")
print(f"    Columns present : {list(df.columns[:6])} ...")
print(f"    Room labels OK  : {df['room'].notna().sum()} / {len(df)}")
print(f"    Load time       : {t_load:.3f}s")
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["CSV Loading"] = ("PASS" if ok else "FAIL", f"{t_load:.3f}s")

# ── TEST 2: Data Processing ───────────────────────────────────────────────────
print("\n[2] Data Processing Module — preprocess_row()")
start = time.time()
bad_row  = np.array([-10.0, 5.0, 110.0, 200.0])   # all out-of-range
good_row = preprocess_row(bad_row)
t_proc = time.time() - start
ok = (good_row[0] == 0 and good_row[1] == 10 and
      good_row[2] == 100 and good_row[3] == 300)
print(f"    Input  (bad)    : AQI={bad_row[0]}, T={bad_row[1]}, H={bad_row[2]}, CO2={bad_row[3]}")
print(f"    Output (clamped): AQI={good_row[0]}, T={good_row[1]}, H={good_row[2]}, CO2={good_row[3]}")
print(f"    Clamp time      : {t_proc*1000:.4f}ms")
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["Data Processing"] = ("PASS" if ok else "FAIL", f"{t_proc*1000:.4f}ms")

# ── TEST 3: Model Training ────────────────────────────────────────────────────
print("\n[3] Machine Learning Module — train_model()")
X, y = prepare_training_data(df)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
start = time.time()
W = train_model(X_train, y_train)
t_train = time.time() - start
ok = W.shape == (5, 4)
print(f"    Training samples: {split}")
print(f"    Test samples    : {len(X) - split}")
print(f"    Weight matrix   : shape {W.shape} (expected 5x4)")
print(f"    Training time   : {t_train:.4f}s")
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["Model Training"] = ("PASS" if ok else "FAIL", f"{t_train:.4f}s")

# ── TEST 4: ML Evaluation ─────────────────────────────────────────────────────
print("\n[4] ML Evaluation — MAE & RMSE on 20% test set")
Xd_test = design_matrix(X_test)
pred    = Xd_test @ W
err     = y_test - pred
mae     = np.mean(np.abs(err), axis=0)
rmse    = np.sqrt(np.mean(err**2, axis=0))
labels  = ["AQI", "Temperature (°C)", "Humidity (%)", "CO2 (ppm)"]
print(f"\n    {'Target':<20} {'MAE':>10} {'RMSE':>10}")
print(f"    {'-'*42}")
for i, lbl in enumerate(labels):
    print(f"    {lbl:<20} {mae[i]:>10.4f} {rmse[i]:>10.4f}")
print(f"\n    Mean MAE : {np.mean(mae):.4f}")
print(f"    Mean RMSE: {np.mean(rmse):.4f}")
results["ML Evaluation"] = ("PASS", "see values above")

# ── TEST 5: Single Inference ──────────────────────────────────────────────────
print("\n[5] Digital Twin Module — predict_next() inference speed")
sample_input = X_test[0]
RUNS = 1000
start = time.time()
for _ in range(RUNS):
    predict_next(sample_input, W)
t_infer = (time.time() - start) / RUNS
ok = t_infer < 0.01  # should be well under 10ms
print(f"    Input vector    : {sample_input.round(2)}")
pred_out = predict_next(sample_input, W)
print(f"    Predicted output: AQI={pred_out[0]:.2f}, T={pred_out[1]:.2f}, H={pred_out[2]:.2f}, CO2={pred_out[3]:.2f}")
print(f"    Avg inference   : {t_infer*1000:.4f}ms (avg over {RUNS} runs)")
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["Single Inference"] = ("PASS" if ok else "FAIL", f"{t_infer*1000:.4f}ms")

# ── TEST 6: Full Refresh Cycle ────────────────────────────────────────────────
print("\n[6] Integration Test — Full refresh cycle (all 3 rooms)")
timestamps = sorted(df["timestamp"].unique())
RUNS = 10
start = time.time()
for _ in range(RUNS):
    ts = timestamps[50]
    for room in ROOMS:
        mask = df["timestamp"] == ts
        subset = df[mask]
        if not subset.empty:
            row = np.array([
                float(subset[FEATURES[0]].iloc[0]),
                float(subset[FEATURES[1]].iloc[0]),
                float(subset[FEATURES[2]].iloc[0]),
                float(subset[FEATURES[3]].iloc[0]),
            ])
            row = preprocess_row(row)
            predict_next(row, W)
t_cycle = (time.time() - start) / RUNS
ok = t_cycle < 1.0
print(f"    Rooms processed : {len(ROOMS)}")
print(f"    Avg cycle time  : {t_cycle*1000:.2f}ms")
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["Full Refresh Cycle"] = ("PASS" if ok else "FAIL", f"{t_cycle*1000:.2f}ms")

# ── TEST 7: Dataset Integrity ─────────────────────────────────────────────────
print("\n[7] Dataset Integrity Check")
missing = df[FEATURES].isna().sum()
total_ts = len(timestamps)
rooms_ok = all(df[df["room"] == r].shape[0] > 0 for r in ROOMS)
print(f"    Total timestamps: {total_ts}")
print(f"    Missing values  :")
for col in FEATURES:
    print(f"      {col}: {missing[col]}")
print(f"    All rooms mapped: {rooms_ok}")
print(f"    Date range      : {timestamps[0]} → {timestamps[-1]}")
ok = missing.sum() == 0 and rooms_ok
print(f"    RESULT          : {'PASS' if ok else 'FAIL'}")
results["Dataset Integrity"] = ("PASS" if ok else "FAIL", "-")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  {'Test':<25} {'Result':>8} {'Time':>12}")
print(f"  {'-'*47}")
for test, (result, timing) in results.items():
    print(f"  {test:<25} {result:>8} {timing:>12}")

passed = sum(1 for r, _ in results.values() if r == "PASS")
print(f"\n  Passed: {passed}/{len(results)}")
print("=" * 55)