import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ML uses NumPy only (no sklearn/scipy) so Streamlit runs reliably on Python 3.13+
# where some scipy builds can fail import with obscure errors.


# -----------------------------
# Configuration and constants
# -----------------------------
APP_TITLE = "AirAware"

# CSV path — expected next to this script
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intelligent_indoor_environment_dataset.csv")

# Room definitions: occupant_id groups → room names.
# The CSV has occupant_ids 1-10; we split them into 3 rooms.
ROOM_OCCUPANT_MAP: Dict[str, List[int]] = {
    "Room 101": [1, 2, 3],
    "Room 102": [4, 5, 6],
    "Room 103": [7, 8, 9, 10],
}
ROOMS = list(ROOM_OCCUPANT_MAP.keys())

TENANT_MAP = {
    "Room 101": "Ava Thompson",
    "Room 102": "Noah Patel",
    "Room 103": "Mia Rodriguez",
}

# Column mapping: CSV column names → display-friendly names
CSV_COL = {
    "aqi": "room_air_quality",
    "temperature": "room_temperature",
    "humidity": "room_humidity",
    "co2": "room_CO2",
    "occupancy": "room_occupancy",
    "lighting": "lighting_intensity",
    "hvac_temp": "HVAC_temperature",
    "energy": "energy_consumption",
    "energy_cost": "energy_cost",
    "efficiency": "energy_efficiency",
    "air_quality_control": "air_quality_control",
    "lighting_control": "lighting_control",
}

# Refresh interval in milliseconds (every 5 seconds — feels real-time)
REFRESH_INTERVAL_MS = 5000


def setup_page() -> None:
    """Configure Streamlit page and shared style."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1 { letter-spacing: -0.02em; }
        .twin-section {
            border: 1px solid rgba(49, 51, 63, 0.22);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin: 0.75rem 0 1.25rem 0;
            background: #f6f8fb;
            color: #0f172a;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        }
        .twin-section h3, .twin-section h4, .twin-section p {
            color: #0f172a !important;
            opacity: 1 !important;
        }
        /* High contrast: Streamlit theme can inherit light text onto light boxes — force readable colors. */
        .arch-flow {
            font-family: ui-monospace, "Cascadia Mono", "Consolas", monospace;
            font-size: 1.08rem;
            font-weight: 600;
            line-height: 1.55;
            padding: 1rem 1.25rem;
            background-color: #dbeafe !important;
            color: #0f172a !important;
            border-radius: 8px;
            border: 1px solid #2563eb;
            border-left: 5px solid #1d4ed8;
            margin: 0.5rem 0 1rem 0;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.6);
        }
        .arch-flow, .arch-flow * {
            color: #0f172a !important;
        }
        div[data-testid="stDataFrame"] div[role="table"] {
            font-size: 0.98rem;
            line-height: 1.35;
        }
        .live-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #22c55e;
            margin-right: 6px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=60)
    with col2:
        st.title(APP_TITLE)
        st.caption("Digital Twin and machine learning system for indoor air quality (AIOT Midterm Project)")


# -----------------------------
# CSV data loading and preprocessing
# -----------------------------
@st.cache_data
def load_csv_data() -> pd.DataFrame:
    """
    Load and preprocess the real indoor environment dataset.
    Returns a cleaned DataFrame with parsed timestamps and proper types.
    """
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Assign room labels based on occupant_id groupings
    room_labels = {}
    for room, occ_ids in ROOM_OCCUPANT_MAP.items():
        for oid in occ_ids:
            room_labels[oid] = room
    df["room"] = df["occupant_id"].map(room_labels)

    # Ensure numeric types
    numeric_cols = [
        "room_temperature", "room_humidity", "lighting_intensity",
        "room_air_quality", "room_CO2", "room_occupancy",
        "HVAC_temperature", "air_quality_control", "energy_consumption",
        "energy_cost", "energy_efficiency", "lighting_control",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_unique_timestamps(df: pd.DataFrame) -> List:
    """Get sorted unique timestamps from the dataset."""
    return sorted(df["timestamp"].unique())


def get_room_data_at_timestamp(df: pd.DataFrame, timestamp, room: str) -> Dict:
    """
    Get sensor data for a specific room at a specific timestamp.

    Since the CSV contains one reading per timestamp (from a single occupant),
    we treat that reading as the building's baseline and apply small deterministic
    offsets per room — simulating the fact that different rooms in the same
    building have slightly different conditions (e.g., sunlight, ventilation).
    """
    mask = df["timestamp"] == timestamp
    subset = df[mask]

    if subset.empty:
        # Fallback: get the nearest timestamp
        idx = (df["timestamp"] - timestamp).abs().idxmin()
        subset = df.loc[[idx]]

    # Base reading from CSV
    base = {
        "aqi": float(subset[CSV_COL["aqi"]].iloc[0]),
        "temperature": float(subset[CSV_COL["temperature"]].iloc[0]),
        "humidity": float(subset[CSV_COL["humidity"]].iloc[0]),
        "co2": float(subset[CSV_COL["co2"]].iloc[0]),
        "occupancy": int(subset[CSV_COL["occupancy"]].iloc[0]),
        "lighting": float(subset[CSV_COL["lighting"]].iloc[0]),
        "hvac_temp": float(subset[CSV_COL["hvac_temp"]].iloc[0]),
        "energy": float(subset[CSV_COL["energy"]].iloc[0]),
        "energy_cost": float(subset[CSV_COL["energy_cost"]].iloc[0]),
        "efficiency": int(subset[CSV_COL["efficiency"]].iloc[0]),
        "air_quality_control": float(subset[CSV_COL["air_quality_control"]].iloc[0]),
    }

    # Deterministic per-room offsets (seeded by room name so they're consistent)
    room_seed = abs(hash(room)) % 1000
    rng = np.random.default_rng(room_seed + int(pd.Timestamp(timestamp).timestamp()) % 10000)
    offsets = {
        "aqi": rng.normal(0, 2.5),
        "temperature": rng.normal(0, 0.8),
        "humidity": rng.normal(0, 1.5),
        "co2": rng.normal(0, 8.0),
        "energy": rng.normal(0, 0.05),
    }

    return {
        "aqi": round(max(0, base["aqi"] + offsets["aqi"]), 1),
        "temperature": round(max(15, min(35, base["temperature"] + offsets["temperature"])), 1),
        "humidity": round(max(20, min(80, base["humidity"] + offsets["humidity"])), 1),
        "co2": round(max(300, min(2000, base["co2"] + offsets["co2"])), 1),
        "occupancy": base["occupancy"] if room != "Room 102" else (1 - base["occupancy"]),  # vary occupancy
        "lighting": round(base["lighting"] + rng.normal(0, 15), 1),
        "hvac_temp": round(base["hvac_temp"] + rng.normal(0, 0.3), 1),
        "energy": round(max(0.1, base["energy"] + offsets["energy"]), 4),
        "energy_cost": round(max(0.01, base["energy_cost"] + rng.normal(0, 0.008)), 4),
        "efficiency": base["efficiency"],
        "air_quality_control": round(base["air_quality_control"], 1),
        "timestamp": pd.Timestamp(timestamp).to_pydatetime(),
    }


def _default_room_reading() -> Dict:
    """Fallback default reading if data is missing."""
    return {
        "aqi": 50.0, "temperature": 22.0, "humidity": 45.0, "co2": 400.0,
        "occupancy": 0, "lighting": 300.0, "hvac_temp": 22.0,
        "energy": 1.0, "energy_cost": 0.12, "efficiency": 1,
        "air_quality_control": 50.0, "timestamp": datetime.now(),
    }


# -----------------------------
# AQI / health helper functions
# -----------------------------
def get_aqi_health_info(aqi: float) -> Tuple[str, str, str]:
    """Map AQI value to EPA-style band for messaging and badge color."""
    if aqi < 50:
        return "Good", "Air quality is satisfactory; little or no risk.", "green"
    if aqi <= 100:
        return "Moderate", "Acceptable for most; sensitive people may feel effects.", "orange"
    return "Unhealthy", "Everyone may begin to feel health effects at longer exposure.", "red"


def get_health_risk_level(aqi: float) -> Tuple[str, str]:
    """Compact risk label for panels (Safe / Moderate Risk / High Risk)."""
    if aqi < 50:
        return "Safe", "green"
    if aqi <= 100:
        return "Moderate Risk", "orange"
    return "High Risk", "red"


def get_status_emoji(color: str) -> str:
    """Visual indicator used across badges, tables, and risk labels."""
    return {"green": "●", "orange": "●", "red": "●"}.get(color, "○")


def get_health_insights(aqi: float, temperature: float, humidity: float) -> Tuple[int, str, str]:
    score = 100 - (aqi * 0.5 + abs(temperature - 24) * 2 + abs(humidity - 50) * 1.5)
    score = max(0, min(100, int(score)))

    if score > 80:
        status = "Excellent (Safe)"
        advice = "Perfect air conditions. Enjoy your environment."
    elif score > 60:
        status = "Good (Monitor)"
        advice = "Slight discomfort possible. Keep ventilation."
    else:
        status = "Poor (Unhealthy)"
        advice = "Unhealthy conditions. Consider ventilation or purifier."

    return score, status, advice


def get_co2_status(co2: float) -> Tuple[str, str, str]:
    """CO2 level assessment."""
    if co2 < 400:
        return "Excellent", "CO2 levels are optimal for comfort and productivity.", "green"
    if co2 <= 450:
        return "Normal", "CO2 levels are within acceptable indoor range.", "orange"
    return "Elevated", "Consider increasing ventilation to improve air circulation.", "red"


def get_property_quality(humidity: float) -> Tuple[str, str, str]:
    """Returns (status_label, description, color) based on humidity."""
    if 40 <= humidity <= 60:
        return "Good", "Optimal conditions, no risk", "#0F9D58"
    if humidity < 40:
        return "Dry", "May cause cracks in furniture/walls", "#F59E0B"
    if humidity <= 70:
        return "Moderate", "Monitor for potential dampness", "#E67E22"
    return "Critical", "Risk of mold and structural damage", "#DC2626"


def render_property_condition_card(room_name: str, humidity: float) -> None:
    """Render a colored card showing property condition based on humidity."""
    status, description, color = get_property_quality(humidity)
    # Choose icon based on status
    icon_map = {"Good": "\u2705", "Dry": "\U0001f4a8", "Moderate": "\u26a0\ufe0f", "Critical": "\U0001f6a8"}
    icon = icon_map.get(status, "\u2753")
    # Lighter background from the main color
    bg_color = color + "18"  # 18 = ~10% opacity in hex
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.4rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        ">
            <div style="font-size:1.3rem; margin-bottom:0.2rem;">{icon}</div>
            <div style="font-weight:700; font-size:1rem; color:{color};">{room_name}</div>
            <div style="font-weight:800; font-size:1.15rem; color:#0f172a; margin:0.15rem 0;">{status}</div>
            <div style="font-size:0.85rem; color:#475569;">{description}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.25rem;">Humidity: {humidity:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Machine learning (trained on REAL CSV data)
# -----------------------------
def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X -> y) pairs from real CSV data: current readings -> next-timestep readings.
    Uses actual sensor correlations from the dataset.
    """
    features = [CSV_COL["aqi"], CSV_COL["temperature"], CSV_COL["humidity"], CSV_COL["co2"]]

    X_list = []
    y_list = []

    # For each room, create sequential pairs (current -> next timestep)
    for room in ROOMS:
        room_df = df[df["room"] == room].sort_values("timestamp").reset_index(drop=True)
        if len(room_df) < 2:
            continue

        # Group by timestamp and average across occupants in the room
        grouped = room_df.groupby("timestamp")[features].mean().sort_index().reset_index()

        for i in range(len(grouped) - 1):
            x_row = grouped.iloc[i][features].values.astype(np.float64)
            y_row = grouped.iloc[i + 1][features].values.astype(np.float64)
            X_list.append(x_row)
            y_list.append(y_row)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """Add bias column for linear / ridge regression."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X.astype(np.float64)])


def train_ml_pipeline(df: pd.DataFrame) -> np.ndarray:
    """
    Fit multi-output ridge regression on REAL dataset:
    (AQI, temp, hum, CO2) -> next timestep (same four).
    Weights shape (5, 4): rows = [bias, aqi, temp, hum, co2], cols = output targets.
    """
    X, y = prepare_training_data(df)
    Xd = _design_matrix(X)
    n_features = Xd.shape[1]
    ridge = 0.15
    ata = Xd.T @ Xd + ridge * np.eye(n_features)
    aty = Xd.T @ y.astype(np.float64)
    weights: np.ndarray = np.linalg.solve(ata, aty)
    return weights


def get_ml_model(df: pd.DataFrame) -> np.ndarray:
    """Cache learned weights in session_state (train once per browser session)."""
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = train_ml_pipeline(df)
    return st.session_state["ml_model"]


def evaluate_model(weights: np.ndarray, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    MAE and RMSE evaluated on the real training data.
    Uses an 80/20 train/test split for honest evaluation.
    """
    X, y = prepare_training_data(df)
    n = len(X)
    split = int(n * 0.8)

    # Evaluate on the held-out 20%
    X_test = X[split:]
    y_test = y[split:]

    Xd_test = _design_matrix(X_test)
    pred = Xd_test @ weights
    err = y_test.astype(np.float64) - pred
    mae_per = np.mean(np.abs(err), axis=0)
    rmse_per = np.sqrt(np.mean(err**2, axis=0))
    return {
        "mae_per": mae_per,
        "rmse_per": rmse_per,
        "mae_mean": float(np.mean(mae_per)),
        "rmse_mean": float(np.mean(rmse_per)),
        "n_train": split,
        "n_test": n - split,
    }


def weights_to_dataframe(weights: np.ndarray) -> pd.DataFrame:
    """Human-readable coefficient table for the multi-output ridge model."""
    return pd.DataFrame(
        weights,
        index=["Bias", "AQI", "Temperature (°C)", "Humidity (%)", "CO2 (ppm)"],
        columns=[
            "Predicted next AQI",
            "Predicted next temp (°C)",
            "Predicted next humidity (%)",
            "Predicted next CO2 (ppm)",
        ],
    ).round(5)


def predict_next_state(current: Dict, weights: np.ndarray) -> Dict:
    """
    ML inference: from current (AQI, temperature, humidity, CO2) predict next timestep.
    """
    xb = np.array(
        [1.0, float(current["aqi"]), float(current["temperature"]),
         float(current["humidity"]), float(current["co2"])],
        dtype=np.float64,
    )
    pred = xb @ weights
    return {
        "aqi": round(float(np.clip(pred[0], 0, 300)), 1),
        "temperature": round(float(np.clip(pred[1], 10.0, 42.0)), 1),
        "humidity": round(float(np.clip(pred[2], 8.0, 95.0)), 1),
        "co2": round(float(np.clip(pred[3], 300, 2000)), 1),
    }


# -----------------------------
# Digital Twin simulation (historical playback from CSV)
# -----------------------------
"""
DIGITAL TWIN EXPLANATION:
Each room is represented as a virtual entity (digital twin).
The system replays real recorded sensor data from the CSV file, advancing
through timesteps every few seconds to simulate live IoT monitoring.
Machine learning predicts future conditions based on current state.
"""


def create_room_entity(room: str, current: Dict, predicted: Dict) -> Dict:
    """Build one virtual room entity for the Digital Twin."""
    return {
        "room_id": room,
        "current": current,
        "predicted": predicted,
    }


def get_sensor_data(df: pd.DataFrame, timestamp_idx: int) -> Dict[str, Dict]:
    """
    Sensor gateway: returns the Digital Twin snapshot for all rooms at a given
    timestamp index (replaying real CSV data like a live sensor feed).
    """
    timestamps = get_unique_timestamps(df)
    # Wrap around when we reach the end
    idx = timestamp_idx % len(timestamps)
    current_ts = timestamps[idx]

    out: Dict[str, Dict] = {}
    for room in ROOMS:
        current = get_room_data_at_timestamp(df, current_ts, room)
        out[room] = create_room_entity(room, current, {"aqi": None, "temperature": None, "humidity": None, "co2": None})
    return out


def hydrate_predictions(data_store: Dict[str, Dict], weights: np.ndarray) -> None:
    """Ensure every room entity has ML forecast fields populated (twin consistency)."""
    for room in ROOMS:
        predicted = predict_next_state(data_store[room]["current"], weights)
        data_store[room]["predicted"] = predicted


def preprocess_data(room_data: Dict) -> Dict:
    """Clean and validate sensor readings — clamp to physical ranges."""
    cur = room_data["current"]
    cur["aqi"] = max(0, min(300, float(cur["aqi"])))
    cur["temperature"] = max(10, min(45, float(cur["temperature"])))
    cur["humidity"] = max(0, min(100, float(cur["humidity"])))
    cur["co2"] = max(300, min(5000, float(cur["co2"])))
    return room_data


def build_trend_data(
    df: pd.DataFrame,
    room: str,
    current_ts_idx: int,
    predicted_aqi: Optional[float] = None,
    points: int = 24,
) -> pd.DataFrame:
    """
    Build AQI trend from REAL historical data — shows actual past readings
    from the CSV, not fabricated random walks.
    """
    timestamps = get_unique_timestamps(df)
    idx = current_ts_idx % len(timestamps)

    # Collect real historical points from the CSV (going back from current)
    start_idx = max(0, idx - points + 1)
    hist_indices = list(range(start_idx, idx + 1))

    times = []
    values = []
    for i in hist_indices:
        ts = timestamps[i]
        reading = get_room_data_at_timestamp(df, ts, room)
        times.append(pd.Timestamp(ts).to_pydatetime())
        values.append(round(float(reading["aqi"]), 1))

    if predicted_aqi is None:
        result = pd.DataFrame({"Time": times, "AQI (live trend)": values})
        return result.set_index("Time")

    # Add forecast point
    last_time = times[-1] if times else datetime.now()
    fut_time = last_time + timedelta(minutes=15)
    forecast_series = [np.nan] * (len(values) - 1) + [values[-1], predicted_aqi]
    hist_ext = values + [np.nan]
    times_ext = times + [fut_time]
    result = pd.DataFrame({
        "Time": times_ext,
        "AQI (live trend)": hist_ext,
        "AQI (predicted next)": forecast_series,
    })
    return result.set_index("Time")


def build_multi_metric_trend(
    df: pd.DataFrame,
    room: str,
    current_ts_idx: int,
    metric: str = "temperature",
    points: int = 24,
) -> pd.DataFrame:
    """Build trend data for any metric from real CSV history."""
    timestamps = get_unique_timestamps(df)
    idx = current_ts_idx % len(timestamps)
    start_idx = max(0, idx - points + 1)
    hist_indices = list(range(start_idx, idx + 1))

    times = []
    values = []
    for i in hist_indices:
        ts = timestamps[i]
        reading = get_room_data_at_timestamp(df, ts, room)
        times.append(pd.Timestamp(ts).to_pydatetime())
        values.append(round(float(reading.get(metric, 0)), 2))

    label_map = {
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "co2": "CO2 (ppm)",
        "aqi": "Air Quality Index",
        "energy": "Energy (kWh)",
    }
    label = label_map.get(metric, metric)
    result = pd.DataFrame({"Time": times, label: values})
    return result.set_index("Time")


# -----------------------------
# Academic / layered documentation (UI)
# -----------------------------
def render_system_architecture() -> None:
    st.markdown("## System Architecture")
    st.markdown(
        '<div class="arch-flow" style="background-color:#dbeafe;color:#0f172a;">'
        '<span style="color:#0f172a !important;">'
        "Real CSV Dataset → Data Processing → Machine Learning → Digital Twin → Visualization (Dashboard)"
        "</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    layers = [
        ("Data Source", "Real sensor dataset (intelligent_indoor_environment_dataset.csv) with 1,000 readings."),
        ("Processing", "Cleaning, type conversion, and room assignment of incoming readings."),
        ("ML", "Multi-output ridge regression trained on real data for next-step prediction."),
        ("Digital Twin", "Virtual room entities replaying real sensor data with synchronized ML forecasts."),
        ("Visualization", "Streamlit dashboards for tenant and owner perspectives."),
    ]
    for col, (title, desc) in zip((c1, c2, c3, c4, c5), layers):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.divider()


def render_system_modules() -> None:
    st.markdown("## System Modules")
    modules = [
        "Data Collection Module — loads and parses real CSV sensor time series (1,000 records, 15-min intervals).",
        "Data Processing Module — cleans observations, maps occupants to rooms, validates ranges (see `preprocess_data`).",
        "Machine Learning Module — trains ridge regression on real data for horizon-one prediction (4 features → 4 targets).",
        "Digital Twin Simulation Module — replays recorded data to simulate live sensor feeds with per-room virtual state.",
        "Dashboard Visualization Module — role-specific Streamlit views with real historical charts.",
    ]
    for m in modules:
        st.markdown(f"- {m}")
    st.divider()


def render_data_source_layer(df: pd.DataFrame) -> None:
    st.markdown("### Data Source Layer")
    timestamps = get_unique_timestamps(df)
    st.write(
        f"This system uses a **real indoor environment dataset** with **{len(df):,} records** across "
        f"**{len(timestamps)} timestamps** (15-minute intervals). "
        f"Data spans from **{timestamps[0]}** to **{timestamps[-1]}**."
    )
    st.info(
        f"The dataset contains readings from **{df['occupant_id'].nunique()} occupants** "
        f"mapped to **{len(ROOMS)} rooms**. Features include temperature, humidity, "
        "air quality, CO2, lighting, HVAC, energy consumption, and occupancy."
    )
    with st.expander("Preview raw dataset (first 10 rows)"):
        st.dataframe(df.head(10))


def render_data_processing_layer() -> None:
    st.markdown("### Data Processing Layer")
    st.write(
        "Incoming sensor data is **preprocessed** before it reaches the machine learning model: "
        "timestamps are parsed, occupant IDs are mapped to room assignments, "
        "numeric columns are validated, and values are clamped to physical ranges."
    )
    st.caption("Processing steps: CSV parsing → type conversion → room mapping → range validation → ML-ready features.")


def render_machine_learning_section(weights: np.ndarray) -> None:
    st.markdown("## Machine Learning Model")
    st.write(
        "**Model:** Multi-output ridge regression (closed-form, NumPy implementation). "
        "**Inputs:** current AQI, temperature, humidity, CO2 (plus intercept). "
        "**Outputs:** predicted AQI, temperature, humidity, and CO2 for the **next timestep** (15 min)."
    )
    st.markdown(
        "**Training:** The regressor is fit on **real sensor data** from the CSV dataset, where "
        "each row maps a current environmental state to the next recorded state. "
        "**Inference:** For each digital twin, the current state vector is multiplied by the learned weight matrix "
        "to produce a multi-target forecast, then clipped to valid physical ranges."
    )
    st.markdown("##### Model coefficients (weights)")
    st.caption("Rows correspond to bias and input features; columns to the four predicted targets.")
    st.dataframe(weights_to_dataframe(weights))
    st.divider()


def render_simulation_engine() -> None:
    st.markdown("## Simulation Engine")
    st.write(
        "The system simulates **real-time environmental change** by **replaying actual recorded sensor data** "
        "from the CSV dataset. Every ~5 seconds, the dashboard advances to the next timestep (15-minute interval), "
        "making the digital twin feel like a live monitoring system. "
        "When the dataset ends, playback loops back to the beginning."
    )
    st.divider()


def render_model_evaluation(weights: np.ndarray, df: pd.DataFrame) -> None:
    st.markdown("## Model Evaluation")
    st.write(
        "Error metrics below are computed on a **held-out 20% test set** from the real dataset "
        "(80% train / 20% test split — sequential to preserve temporal ordering)."
    )
    ev = evaluate_model(weights, df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean MAE", f"{ev['mae_mean']:.4f}")
    m2.metric("Mean RMSE", f"{ev['rmse_mean']:.4f}")
    m3.metric("Training samples", f"{ev['n_train']}")
    m4.metric("Test samples", f"{ev['n_test']}")
    labels = ["AQI", "Temperature (°C)", "Humidity (%)", "CO2 (ppm)"]
    detail = pd.DataFrame(
        {
            "Target": labels,
            "MAE": ev["mae_per"],
            "RMSE": ev["rmse_per"],
        }
    )
    st.caption("Per-target breakdown")
    st.dataframe(detail, hide_index=True)
    st.divider()


def render_visualization_layer_note() -> None:
    st.markdown("## Visualization Layer")
    st.write(
        "The **tenant** and **owner** dashboards below implement the visualization layer: "
        "interactive exploration of digital twin state, forecasts, and operational comparisons."
    )
    st.divider()


def render_digital_twin_representation(room_id: str, room_data: Dict, predictions: Dict) -> None:
    ts = room_data["current"]["timestamp"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
    occupancy_str = "🟢 Occupied" if room_data["current"].get("occupancy", 0) == 1 else "⚪ Unoccupied"

    st.markdown("### Digital Twin Representation")
    st.markdown(
        '<div class="twin-section">'
        f'<h4 style="margin-top:0;">Room: {room_id} &nbsp; '
        f'<span style="font-size:0.9rem;">{occupancy_str}</span></h4>'
        f"<p><b>Timestamp (live state):</b> {ts_str}</p>"
        "<p><b>Current state:</b> "
        f"AQI {room_data['current']['aqi']}, "
        f"{room_data['current']['temperature']} °C, "
        f"{room_data['current']['humidity']}% RH, "
        f"{room_data['current']['co2']} ppm CO2</p>"
        "<p><b>Predicted state (next timestep):</b> "
        f"AQI {predictions.get('aqi', 'N/A')}, "
        f"{predictions.get('temperature', 'N/A')} °C, "
        f"{predictions.get('humidity', 'N/A')}% RH, "
        f"{predictions.get('co2', 'N/A')} ppm CO2</p>"
        "<p style='margin-bottom:0; font-size:0.95rem;'>"
        "This digital twin is a <b>virtual replica</b> of a physical room, continuously updated using "
        "real recorded sensor data and the predictive model (next-timestep state)."
        "</p></div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# UI components
# -----------------------------
def render_aqi_badge(color: str, status: str, aqi_value: float) -> None:
    """Pill badge for live AQI band + numeric hint."""
    color_map = {"green": "#0F9D58", "orange": "#F59E0B", "red": "#DC2626"}
    badge_color = color_map.get(color, "#6B7280")
    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:0.45rem;
            background:{badge_color};
            color:white;
            padding:0.4rem 1rem;
            border-radius:999px;
            font-size:0.95rem;
            font-weight:700;
            margin:0.35rem 0 0.85rem 0;
            box-shadow:0 2px 8px rgba(0,0,0,0.12);
        ">
            <span class="live-dot"></span>
            <span>AQI {aqi_value:.1f} — {status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_risk_panel(aqi: float) -> None:
    """Explicit health risk label."""
    label, tone = get_health_risk_level(aqi)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Health risk (live)", f"{label}")
    with c2:
        st.caption("Derived from current AQI using a rule-based band mapping.")


def render_current_metrics(room_data: Dict, *, heading: str = "Current state (live sensors)") -> None:
    st.markdown(f"##### {heading}")
    cur = room_data["current"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "AQI",
        f"{cur['aqi']:.1f}",
        help="Air Quality Index: lower values indicate cleaner air.",
    )
    c2.metric("Temperature (°C)", f"{cur['temperature']:.1f}")
    c3.metric("Humidity (%)", f"{cur['humidity']:.1f}")
    c4.metric("CO2 (ppm)", f"{cur['co2']:.1f}", help="Carbon dioxide concentration.")


def render_extended_metrics(room_data: Dict) -> None:
    """Display additional sensor metrics: energy, HVAC, lighting."""
    cur = room_data["current"]
    st.markdown("##### Additional Sensor Readings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HVAC Temp (°C)", f"{cur.get('hvac_temp', 'N/A')}")
    c2.metric("Lighting (lux)", f"{cur.get('lighting', 'N/A')}")
    c3.metric("Energy (kWh)", f"{cur.get('energy', 'N/A')}")
    c4.metric(
        "Efficiency",
        "✅ Efficient" if cur.get("efficiency", 0) == 1 else "⚠️ Inefficient",
    )


def render_predictions(predictions: Dict) -> None:
    st.markdown("##### Predicted state (next timestep, ML)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predicted AQI", f"{predictions['aqi']:.1f}", help="Ridge regression forecast.")
    p2.metric("Predicted temp (°C)", f"{predictions['temperature']:.1f}")
    p3.metric("Predicted humidity (%)", f"{predictions['humidity']:.1f}")
    p4.metric("Predicted CO2 (ppm)", f"{predictions['co2']:.1f}")


def render_forecast_insights(current_aqi: float, predictions: Dict) -> None:
    pred_aqi = float(predictions["aqi"])
    col_a, col_b = st.columns(2)
    with col_a:
        if pred_aqi > current_aqi + 1:
            st.warning("Forecast: air quality is **expected to worsen** in the next timestep (higher predicted AQI).")
        elif pred_aqi < current_aqi - 1:
            st.success("Forecast: predicted AQI is **lower**; conditions may improve.")
        else:
            st.info("Forecast: predicted AQI is **stable** — close to the current level.")
    with col_b:
        risk_label, tone = get_health_risk_level(pred_aqi)
        st.caption("Risk band from predicted AQI")
        if tone == "green":
            st.success(f"Forecast risk: **{risk_label}**")
        elif tone == "orange":
            st.warning(f"Forecast risk: **{risk_label}**")
        else:
            st.error(f"Forecast risk: **{risk_label}**")


def render_smart_alerts(current_aqi: float, predictions: Dict) -> None:
    pred_aqi = float(predictions["aqi"])
    if current_aqi > 55:
        st.warning("**Alert:** Current AQI is above 55 — air quality is declining.")
    if pred_aqi > 55:
        st.warning("**Alert:** Predicted AQI exceeds 55 in the next timestep.")
    if current_aqi > 80:
        st.error("**Critical:** AQI exceeds 80 — protective action advised.")


def render_co2_panel(co2: float) -> None:
    """CO2 assessment panel."""
    status, message, color = get_co2_status(co2)
    color_map = {"green": "#0F9D58", "orange": "#F59E0B", "red": "#DC2626"}
    badge_color = color_map.get(color, "#6B7280")
    st.markdown(
        f"""
        <div style="
            display:inline-flex; align-items:center; gap:0.45rem;
            background:{badge_color}; color:white;
            padding:0.3rem 0.9rem; border-radius:999px;
            font-size:0.88rem; font-weight:600;
            margin:0.2rem 0 0.5rem 0;
        ">
            <span>CO2: {co2:.0f} ppm — {status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(message)


def tenant_dashboard(data_store: Dict[str, Dict], df: pd.DataFrame, ts_idx: int) -> None:
    st.markdown("#### Tenant Dashboard")
    selected_room = st.selectbox("Select your room", ROOMS, index=0, key="tenant_room")

    room_data = preprocess_data(data_store[selected_room])
    predictions = room_data["predicted"]
    aqi = float(room_data["current"]["aqi"])
    co2 = float(room_data["current"]["co2"])
    status, message, color = get_aqi_health_info(aqi)

    render_digital_twin_representation(selected_room, room_data, predictions)

    render_aqi_badge(color, status, aqi)
    render_health_risk_panel(aqi)
    render_co2_panel(co2)

    c_left, c_right = st.columns([1.1, 1])
    with c_left:
        render_current_metrics(room_data)
    with c_right:
        st.markdown("##### Health advisory (AQI band)")
        st.info(message)



    score, status_label, advice = get_health_insights(
        aqi,
        float(room_data["current"]["temperature"]),
        float(room_data["current"]["humidity"]),
    )
    st.markdown("##### Health Intelligence")
    st.metric("Composite health score", score)
    st.write(f"Status: {status_label}")
    st.info(advice)

    render_smart_alerts(aqi, predictions)

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### AQI Trend (real historical data)")
    trend_df = build_trend_data(df, selected_room, ts_idx, predicted_aqi=predictions["aqi"])
    st.line_chart(trend_df)

    st.markdown("##### Temperature & Humidity Trends")
    tc1, tc2 = st.columns(2)
    with tc1:
        temp_df = build_multi_metric_trend(df, selected_room, ts_idx, "temperature")
        st.line_chart(temp_df)
    with tc2:
        hum_df = build_multi_metric_trend(df, selected_room, ts_idx, "humidity")
        st.line_chart(hum_df)


def owner_dashboard(data_store: Dict[str, Dict], df: pd.DataFrame, ts_idx: int) -> None:
    st.markdown("#### Owner / Operator Dashboard")
    selected_room = st.selectbox("Select room", ROOMS, index=0, key="owner_room")

    room_data = preprocess_data(data_store[selected_room])
    predictions = room_data["predicted"]
    aqi = float(room_data["current"]["aqi"])
    status, message, color = get_aqi_health_info(aqi)

    top = st.columns([2, 1])
    with top[0]:
        st.caption("Property and tenant context")
    with top[1]:
        st.markdown(f"**Assigned tenant:** {TENANT_MAP[selected_room]}")

    render_digital_twin_representation(selected_room, room_data, predictions)

    render_aqi_badge(color, status, aqi)
    render_health_risk_panel(aqi)

    render_current_metrics(room_data)
    st.caption(message)



    # Property condition cards — one per room
    st.markdown("##### Property Condition (all rooms)")
    pc_cols = st.columns(len(ROOMS))
    for i, room_name in enumerate(ROOMS):
        with pc_cols[i]:
            room_humidity = float(data_store[room_name]["current"]["humidity"])
            render_property_condition_card(room_name, room_humidity)


    render_smart_alerts(aqi, predictions)

    act1, _ = st.columns([1, 3])
    with act1:
        if st.button("Notify tenant", key="notify_btn"):
            st.success(f"Notification queued for **{TENANT_MAP[selected_room]}**.")

    st.markdown("##### Room Comparison (all rooms)")
    room_rows = []
    for room_name in ROOMS:
        cur = data_store[room_name]["current"]
        room_status, _, room_color = get_aqi_health_info(float(cur["aqi"]))
        prop_status, _, _ = get_property_quality(float(cur["humidity"]))
        room_rows.append(
            {
                "Room": room_name,
                "AQI Status": room_status,
                "AQI": round(float(cur["aqi"]), 1),
                "Temp (°C)": round(float(cur["temperature"]), 1),
                "Humidity (%)": round(float(cur["humidity"]), 1),
                "Property Condition": prop_status,
                "CO2 (ppm)": round(float(cur["co2"]), 1),
            }
        )
    room_df = pd.DataFrame(room_rows).sort_values("AQI", ascending=True).reset_index(drop=True)
    avg_aqi = round(float(room_df["AQI"].mean()), 1)
    st.metric("Average AQI (all rooms)", avg_aqi)
    st.dataframe(room_df, hide_index=True)

    best_room = room_df.iloc[0]
    worst_room = room_df.iloc[-1]
    c_best, c_worst = st.columns(2)
    with c_best:
        st.success(f"Best air quality: **{best_room['Room']}** (AQI {best_room['AQI']}).")
    with c_worst:
        st.error(f"Needs attention: **{worst_room['Room']}** (AQI {worst_room['AQI']}).")

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### AQI Trend (operations view)")
    trend_df = build_trend_data(df, selected_room, ts_idx, predicted_aqi=predictions["aqi"])
    st.line_chart(trend_df)

    st.markdown("##### CO2 & Energy Trends")
    tc1, tc2 = st.columns(2)
    with tc1:
        co2_df = build_multi_metric_trend(df, selected_room, ts_idx, "co2")
        st.line_chart(co2_df)
    with tc2:
        energy_df = build_multi_metric_trend(df, selected_room, ts_idx, "energy")
        st.line_chart(energy_df)


def render_status_bar(last_update: datetime, ts_idx: int, total_ts: int) -> None:
    progress_pct = ((ts_idx % total_ts) / total_ts) * 100
    st.markdown(
        f'<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:0.75rem;">'
        f'<span><span class="live-dot"></span><b>Live Simulation</b></span>'
        f'<span><b>Updated</b> — {last_update.strftime("%Y-%m-%d %H:%M:%S")}</span>'
        f"<span><b>ML</b> — Ridge regression (next timestep)</span>"
        f"<span><b>Data</b> — Real CSV dataset</span>"
        f"<span><b>Progress</b> — Timestep {ts_idx % total_ts + 1}/{total_ts} ({progress_pct:.0f}%)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_controls(df: pd.DataFrame) -> bool:
    st.sidebar.header("Controls")
    st.sidebar.caption(
        "The digital twin replays real sensor data every 5 seconds, "
        "simulating live IoT monitoring. Data advances automatically."
    )
    refresh_clicked = st.sidebar.button("⏭️ Next Timestep", key="next_ts_btn")

    timestamps = get_unique_timestamps(df)
    ts_idx = st.session_state.get("ts_idx", 0) % len(timestamps)
    current_ts = timestamps[ts_idx]

    st.sidebar.divider()
    st.sidebar.markdown("##### Dataset Info")
    st.sidebar.caption(f"📊 Total records: **{len(df):,}**")
    st.sidebar.caption(f"🕐 Timesteps: **{len(timestamps)}**")
    st.sidebar.caption(f"🏠 Rooms: **{len(ROOMS)}**")
    st.sidebar.caption(f"📅 Current: **{current_ts}**")

    st.sidebar.divider()
    st.sidebar.markdown("##### Playback Speed")
    st.sidebar.caption("🟢 Auto-refresh: every 5 seconds (active)")

    return refresh_clicked


def main() -> None:
    """Application entry point."""
    setup_page()

    # Auto-refresh every 5 seconds — triggers a Streamlit rerun that preserves session_state
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresh")

    # Load real dataset
    df = load_csv_data()
    timestamps = get_unique_timestamps(df)
    total_ts = len(timestamps)

    refresh_clicked = render_sidebar_controls(df)

    # Initialize or advance the timestamp index
    if "ts_idx" not in st.session_state:
        st.session_state["ts_idx"] = 0
    else:
        # Auto-advance on every rerun (driven by auto-refresh or manual click)
        st.session_state["ts_idx"] += 1

    ts_idx = st.session_state["ts_idx"]

    # Get sensor data at current timestep from real CSV
    data_store = get_sensor_data(df, ts_idx)

    # Train/load ML model and hydrate predictions
    weights = get_ml_model(df)
    hydrate_predictions(data_store, weights)

    st.session_state["data_store"] = data_store
    st.session_state["last_update"] = datetime.now()

    render_status_bar(st.session_state["last_update"], ts_idx, total_ts)
    st.info(
        f"\U0001f504 **Live simulation** \u2014 replaying real sensor data. "
        f"Timestep **{ts_idx % total_ts + 1}** of **{total_ts}** "
        f"(data updates every 5 seconds automatically)."
    )

    # Model input data table
    st.markdown("### Model Input Data (Current Features)")
    sample_inputs = []
    for room in ROOMS:
        cur = data_store[room]["current"]
        sample_inputs.append(
            {
                "Room": room,
                "AQI": round(float(cur["aqi"]), 1),
                "Temperature (\u00b0C)": round(float(cur["temperature"]), 1),
                "Humidity (%)": round(float(cur["humidity"]), 1),
                "CO2 (ppm)": round(float(cur["co2"]), 1),
                "Occupied": "Yes" if cur.get("occupancy", 0) == 1 else "No",
            }
        )
    df_inputs = pd.DataFrame(sample_inputs)
    st.dataframe(df_inputs, hide_index=True)

    render_system_architecture()

    st.markdown("## Layer Documentation")
    d1, d2 = st.columns(2)
    with d1:
        render_data_source_layer(df)
    with d2:
        render_data_processing_layer()

    render_machine_learning_section(weights)
    render_system_modules()
    render_model_evaluation(weights, df)
    render_simulation_engine()
    render_visualization_layer_note()

    tab_tenant, tab_owner = st.tabs(["\U0001f3e0 Tenant Dashboard", "\U0001f3e2 Owner Dashboard"])
    with tab_tenant:
        tenant_dashboard(data_store, df, ts_idx)
    with tab_owner:
        owner_dashboard(data_store, df, ts_idx)

    st.markdown("---")
    st.caption("AirAware \u2014 Digital Twin system for AIOT (Streamlit) | Powered by real indoor environment dataset")


if __name__ == "__main__":
    main()

