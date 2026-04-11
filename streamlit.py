import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh

    _HAS_AUTOREFRESH = True
except ImportError:
    st_autorefresh = None  # type: ignore[assignment, misc]
    _HAS_AUTOREFRESH = False

# ML uses NumPy only (no sklearn/scipy) so Streamlit runs reliably on Python 3.13+
# where some scipy builds can fail import with obscure errors.


# -----------------------------
# Configuration and constants
# -----------------------------
APP_TITLE = "AirAware"
ROOMS = ["Room 101", "Room 102", "Room 103"]
TENANT_MAP = {
    "Room 101": "Ava Thompson",
    "Room 102": "Noah Patel",
    "Room 103": "Mia Rodriguez",
}

# Drift scales mimic slow sensor / indoor air dynamics (tweak for demo feel).
DRIFT_SIGMA = {"aqi": 4.0, "temperature": 0.28, "humidity": 0.85}


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
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=60)
    with col2:
        st.title(APP_TITLE)
        st.caption("Digital Twin and machine learning system for indoor air quality (academic demonstration)")


def get_aqi_health_info(aqi: int) -> Tuple[str, str, str]:
    """Map AQI value to EPA-style band for messaging and badge color."""
    if aqi < 50:
        return "Good", "Air quality is satisfactory; little or no risk.", "green"
    if aqi <= 100:
        return "Moderate", "Acceptable for most; sensitive people may feel effects.", "orange"
    return "Unhealthy", "Everyone may begin to feel health effects at longer exposure.", "red"


def get_health_risk_level(aqi: int) -> Tuple[str, str]:
    """Compact risk label for panels (Safe / Moderate Risk / High Risk)."""
    if aqi < 50:
        return "Safe", "green"
    if aqi <= 100:
        return "Moderate Risk", "orange"
    return "High Risk", "red"


def get_status_emoji(color: str) -> str:
    """Visual indicator used across badges, tables, and risk labels."""
    return {"green": "●", "orange": "●", "red": "●"}.get(color, "○")


def get_health_insights(aqi: int, temperature: float, humidity: float) -> Tuple[int, str, str]:
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


def get_property_quality(humidity: float) -> Tuple[str, str]:
    if 40 <= humidity <= 60:
        return "Good (Safe)", "Optimal conditions, no risk"
    if humidity < 40:
        return "Dry (Attention)", "May cause cracks in furniture/walls"
    if humidity <= 70:
        return "Moderate (Monitor)", "Monitor for potential dampness"
    return "High Risk (Critical)", "Risk of mold and structural damage"


# -----------------------------
# Machine learning (synthetic training)
# -----------------------------
def generate_synthetic_training_data(n_samples: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic (X -> y) pairs: current readings -> next-hour readings.
    Uses simple physics-inspired correlations so the regressor learns smooth rules.
    """
    rng = np.random.default_rng(42)
    aqi = rng.uniform(18, 195, n_samples)
    temp = rng.uniform(17.5, 32.0, n_samples)
    hum = rng.uniform(28.0, 78.0, n_samples)
    X = np.column_stack([aqi, temp, hum])

    aqi_next = np.clip(
        aqi + 0.12 * (temp - 22.0) - 0.07 * (hum - 50.0) + rng.normal(0.0, 7.5, n_samples),
        0,
        300,
    )
    temp_next = np.clip(temp + rng.normal(0.0, 0.38, n_samples), 10.0, 40.0)
    hum_next = np.clip(hum + rng.normal(0.0, 1.15, n_samples), 10.0, 95.0)
    y = np.column_stack([aqi_next, temp_next, hum_next])
    return X, y


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """Add bias column for linear / ridge regression."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X.astype(np.float64)])


def train_ml_pipeline() -> np.ndarray:
    """
    Fit multi-output ridge regression: (AQI, temp, hum) -> next hour (same three).
    Weights shape (4, 3): rows = [bias, aqi, temp, hum], cols = output targets.
    """
    X, y = generate_synthetic_training_data()
    Xd = _design_matrix(X)
    n_features = Xd.shape[1]
    ridge = 0.15
    ata = Xd.T @ Xd + ridge * np.eye(n_features)
    aty = Xd.T @ y.astype(np.float64)
    weights: np.ndarray = np.linalg.solve(ata, aty)
    return weights


def get_ml_model() -> np.ndarray:
    """Cache learned weights in session_state (train once per browser session)."""
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = train_ml_pipeline()
    return st.session_state["ml_model"]


def evaluate_model_on_training_data(weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    MAE and RMSE on the same synthetic dataset used for training (in-sample metrics).
    """
    X, y = generate_synthetic_training_data()
    Xd = _design_matrix(X)
    pred = Xd @ weights
    err = y.astype(np.float64) - pred
    mae_per = np.mean(np.abs(err), axis=0)
    rmse_per = np.sqrt(np.mean(err**2, axis=0))
    return {
        "mae_per": mae_per,
        "rmse_per": rmse_per,
        "mae_mean": float(np.mean(mae_per)),
        "rmse_mean": float(np.mean(rmse_per)),
    }


def weights_to_dataframe(weights: np.ndarray) -> pd.DataFrame:
    """Human-readable coefficient table for the multi-output ridge model."""
    return pd.DataFrame(
        weights,
        index=["Bias", "AQI", "Temperature (°C)", "Humidity (%)"],
        columns=["Predicted next-hour AQI", "Predicted next-hour temp (°C)", "Predicted next-hour humidity (%)"],
    ).round(5)


# -----------------------------
# Digital Twin simulation (stateful drift)
# -----------------------------
"""
DIGITAL TWIN EXPLANATION:
Each room is represented as a virtual entity.
The system maintains state and simulates real-time sensor updates using drift.
Machine learning predicts future conditions based on current state.
"""


def _metrics_only(snapshot: Dict) -> Dict:
    """Strip to the three live metrics used for drift + ML."""
    return {
        "aqi": int(snapshot["aqi"]),
        "temperature": float(snapshot["temperature"]),
        "humidity": float(snapshot["humidity"]),
    }


def initialize_twin_baseline(room_name: str) -> Dict:
    """First reading per room: stable starting point (deterministic per room name)."""
    seed = abs(hash(room_name)) % (2**31)
    rng = random.Random(seed)
    return {
        "aqi": rng.randint(38, 115),
        "temperature": round(rng.uniform(20.0, 26.5), 1),
        "humidity": round(rng.uniform(42.0, 64.0), 1),
        "timestamp": datetime.now(),
    }


def drift_from_previous(prev: Dict) -> Dict:
    """Small Gaussian steps from last state = believable sensor stream."""
    rng = random.Random()
    p = _metrics_only(prev)
    new_aqi = int(np.clip(p["aqi"] + rng.gauss(0, DRIFT_SIGMA["aqi"]), 15, 240))
    new_temp = round(p["temperature"] + rng.gauss(0, DRIFT_SIGMA["temperature"]), 1)
    new_hum = round(p["humidity"] + rng.gauss(0, DRIFT_SIGMA["humidity"]), 1)
    new_temp = float(np.clip(new_temp, 16.0, 34.0))
    new_hum = float(np.clip(new_hum, 28.0, 76.0))
    return {
        "aqi": new_aqi,
        "temperature": new_temp,
        "humidity": new_hum,
        "timestamp": datetime.now(),
    }


def create_room_entity(room_name: str, previous_current: Optional[Dict]) -> Dict:
    """
    Build one virtual room entity for the Digital Twin.
    - previous_current: last live snapshot; if None, use baseline initializer.
    """
    if previous_current is None:
        current = initialize_twin_baseline(room_name)
    else:
        current = drift_from_previous(previous_current)

    return {
        "room_id": room_name,
        "current": current,
        "predicted": {"aqi": None, "temperature": None, "humidity": None},
    }


def get_sensor_data(previous_store: Optional[Dict[str, Dict]]) -> Dict[str, Dict]:
    """
    Sensor gateway: returns the latest Digital Twin snapshot for all rooms.
    When previous_store is provided, new readings drift from prior live values.
    """
    out: Dict[str, Dict] = {}
    for room in ROOMS:
        prev_room = previous_store.get(room) if previous_store else None
        prev_cur = prev_room["current"] if prev_room else None
        out[room] = create_room_entity(room, prev_cur)
    return out


def preprocess_data(room_data: Dict) -> Dict:
    """Hook for future cleaning / normalization; passes through for now."""
    return room_data


def predict_aqi(room_data: Dict) -> Dict:
    """
    ML inference: from current (AQI, temperature, humidity) predict next hour.
    Writes into room_data['predicted'] and returns the same triple for UI helpers.
    """
    weights = get_ml_model()
    cur = room_data["current"]
    xb = np.array(
        [1.0, float(cur["aqi"]), float(cur["temperature"]), float(cur["humidity"])],
        dtype=np.float64,
    )
    pred = xb @ weights
    aqi_p = int(np.clip(round(pred[0]), 0, 300))
    temp_p = round(float(np.clip(pred[1], 10.0, 42.0)), 1)
    hum_p = round(float(np.clip(pred[2], 8.0, 95.0)), 1)

    room_data["predicted"]["aqi"] = aqi_p
    room_data["predicted"]["temperature"] = temp_p
    room_data["predicted"]["humidity"] = hum_p

    return {"aqi": aqi_p, "temperature": temp_p, "humidity": hum_p}


def hydrate_predictions(data_store: Dict[str, Dict]) -> None:
    """Ensure every room entity has ML forecast fields populated (twin consistency)."""
    for room in ROOMS:
        predict_aqi(preprocess_data(data_store[room]))


def build_trend_data(
    current_aqi: int,
    predicted_aqi: Optional[int] = None,
    points: int = 24,
    seed_salt: int = 0,
) -> pd.DataFrame:
    """
    Smoothed historic AQI curve (random walk with small steps) ending at the live value.
    Optional second series: +1h forecast point for chart readability.
    """
    now = datetime.now()
    interval_min = 5
    past_times = [now - timedelta(minutes=interval_min * (points - 1 - i)) for i in range(points)]

    rng = random.Random((hash(current_aqi) ^ seed_salt) % (2**31))
    values: List[float] = []
    v = float(max(18, current_aqi - rng.uniform(10, 22)))
    for _ in range(points - 1):
        v += rng.gauss(0, 2.2)
        v = float(np.clip(v, 12, 260))
        values.append(v)
    values.append(float(current_aqi))

    hist = [int(round(x)) for x in values]

    if predicted_aqi is None:
        df = pd.DataFrame({"Time": past_times, "AQI (live trend)": hist})
        return df.set_index("Time")

    fut_time = now + timedelta(hours=1)
    forecast_series = [np.nan] * (points - 1) + [current_aqi, predicted_aqi]
    hist_ext = hist + [np.nan]
    times_ext = past_times + [fut_time]
    df = pd.DataFrame(
        {
            "Time": times_ext,
            "AQI (live trend)": hist_ext,
            "AQI (+1 hour forecast)": forecast_series,
        }
    )
    return df.set_index("Time")


# -----------------------------
# Academic / layered documentation (UI)
# -----------------------------
def render_system_architecture() -> None:
    st.markdown("## System Architecture")
    st.markdown(
        '<div class="arch-flow" style="background-color:#dbeafe;color:#0f172a;">'
        '<span style="color:#0f172a !important;">'
        "IoT / Data Source → Data Processing → Machine Learning → Digital Twin → Visualization (Dashboard)"
        "</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    layers = [
        ("Data Source", "Simulated IoT sensors producing AQI, temperature, and humidity streams."),
        ("Processing", "Cleaning and preprocessing of incoming readings before model input."),
        ("ML", "Multi-output ridge regression for next-hour state prediction."),
        ("Digital Twin", "Virtual room entities with stateful drift and synchronized forecasts."),
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
        "Data Collection Module — generates and ingests simulated sensor time series.",
        "Data Processing Module — prepares observations for the learning layer (see `preprocess_data`).",
        "Machine Learning Module — trains and applies ridge regression for horizon-one prediction.",
        "Digital Twin Simulation Module — maintains per-room virtual state with stochastic drift.",
        "Dashboard Visualization Module — role-specific Streamlit views and charts.",
    ]
    for m in modules:
        st.markdown(f"- {m}")
    st.divider()


def render_data_source_layer() -> None:
    st.markdown("### Data Source Layer")
    st.write(
        "This demonstration uses **simulated IoT sensor data**: Air Quality Index (AQI), "
        "temperature (°C), and relative humidity (%). "
        "Readings are not from physical hardware; they are generated to mimic realistic variability."
    )
    st.info(
        "Successive samples evolve through **stochastic drift** (Gaussian perturbations), "
        "approximating slow indoor environmental dynamics and sensor noise."
    )


def render_data_processing_layer() -> None:
    st.markdown("### Data Processing Layer")
    st.write(
        "Incoming sensor data is **preprocessed** before it is passed to the machine learning model. "
        "The pipeline reserves hooks for normalization, validation, and cleaning; "
        "the current implementation applies a pass-through while preserving the architectural boundary."
    )
    st.caption("Normalization and explicit cleaning rules can be extended inside `preprocess_data()` without changing model code.")


def render_machine_learning_section(weights: np.ndarray) -> None:
    st.markdown("## Machine Learning Model")
    st.write(
        "**Model:** Multi-output ridge regression (closed-form, NumPy implementation). "
        "**Inputs:** current AQI, temperature, humidity (plus intercept). "
        "**Outputs:** predicted AQI, temperature, and humidity for the **next hour**."
    )
    st.markdown(
        "**Training:** The regressor is fit on a **synthetic dataset** (deterministic seed) where "
        "each row maps current environmental state to a plausible next-hour state with coupled dynamics and noise. "
        "**Inference:** For each digital twin, the current state vector is multiplied by the learned weight matrix "
        "to produce a multi-target forecast, then clipped to valid physical ranges."
    )
    st.markdown("##### Model coefficients (weights)")
    st.caption("Rows correspond to bias and input features; columns to the three predicted targets.")
    st.dataframe(weights_to_dataframe(weights), use_container_width=True)
    st.divider()


def render_simulation_engine() -> None:
    st.markdown("## Simulation Engine")
    st.write(
        "The system simulates **real-time environmental change** using **stochastic drift**: "
        "each update applies small random adjustments to the previous live state per room. "
        "This mimics continuous IoT polling and gradual indoor air evolution without external APIs."
    )
    st.divider()


def render_model_evaluation(weights: np.ndarray) -> None:
    st.markdown("## Model Evaluation")
    st.write(
        "Error metrics below are computed on the **same synthetic training set** used to fit the ridge model "
        "(in-sample evaluation suitable for this academic demo)."
    )
    ev = evaluate_model_on_training_data(weights)
    m1, m2 = st.columns(2)
    m1.metric("Mean MAE (average over outputs)", f"{ev['mae_mean']:.4f}")
    m2.metric("Mean RMSE (average over outputs)", f"{ev['rmse_mean']:.4f}")
    labels = ["AQI", "Temperature (°C)", "Humidity (%)"]
    detail = pd.DataFrame(
        {
            "Target": labels,
            "MAE": ev["mae_per"],
            "RMSE": ev["rmse_per"],
        }
    )
    st.caption("Per-target breakdown")
    st.dataframe(detail, use_container_width=True, hide_index=True)
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
    st.markdown("### Digital Twin Representation")
    st.markdown(
        '<div class="twin-section">'
        f'<h4 style="margin-top:0;">Room: {room_id}</h4>'
        f"<p><b>Timestamp (live state):</b> {ts_str}</p>"
        "<p><b>Current state:</b> "
        f"AQI {room_data['current']['aqi']}, "
        f"{room_data['current']['temperature']} °C, "
        f"{room_data['current']['humidity']}% RH</p>"
        "<p><b>Predicted state (+1 hour):</b> "
        f"AQI {predictions['aqi']}, "
        f"{predictions['temperature']} °C, "
        f"{predictions['humidity']}% RH</p>"
        "<p style='margin-bottom:0; font-size:0.95rem;'>"
        "This digital twin is a <b>virtual replica</b> of a physical room, continuously updated using "
        "simulated sensor data and the predictive model (next-hour state)."
        "</p></div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# UI components
# -----------------------------
def render_aqi_badge(color: str, status: str, aqi_value: int) -> None:
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
            <span style="font-size:1.1rem;line-height:1;">{get_status_emoji(color)}</span>
            <span>AQI {aqi_value} — {status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_risk_panel(aqi: int) -> None:
    """Explicit health risk label."""
    label, tone = get_health_risk_level(aqi)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Health risk (live)", f"{label}")
    with c2:
        st.caption("Derived from current AQI using a rule-based band mapping (demonstration).")


def render_current_metrics(room_data: Dict, *, heading: str = "Current state (live sensors)") -> None:
    st.markdown(f"##### {heading}")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "AQI",
        room_data["current"]["aqi"],
        help="Air Quality Index: lower values indicate cleaner air.",
    )
    c2.metric("Temperature (°C)", room_data["current"]["temperature"])
    c3.metric("Humidity (%)", room_data["current"]["humidity"])


def render_predictions(predictions: Dict) -> None:
    st.markdown("##### Predicted state (+1 hour, ML)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted AQI", predictions["aqi"], help="Ridge regression forecast.")
    p2.metric("Predicted temp (°C)", predictions["temperature"])
    p3.metric("Predicted humidity (%)", predictions["humidity"])


def render_forecast_insights(current_aqi: int, predictions: Dict) -> None:
    pred_aqi = int(predictions["aqi"])
    col_a, col_b = st.columns(2)
    with col_a:
        if pred_aqi > current_aqi:
            st.warning("Forecast: air quality is **expected to worsen** in the next hour (higher predicted AQI).")
        elif pred_aqi < current_aqi:
            st.success("Forecast: predicted AQI is **lower**; conditions may improve.")
        else:
            st.info("Forecast: predicted AQI is **close** to the current level.")
    with col_b:
        risk_label, tone = get_health_risk_level(pred_aqi)
        st.caption("Risk band from predicted AQI")
        if tone == "green":
            st.success(f"Forecast risk: **{risk_label}**")
        elif tone == "orange":
            st.warning(f"Forecast risk: **{risk_label}**")
        else:
            st.error(f"Forecast risk: **{risk_label}**")


def render_smart_alerts(current_aqi: int, predictions: Dict) -> None:
    pred_aqi = int(predictions["aqi"])
    if current_aqi > 150:
        st.error("**Alert:** Current AQI exceeds 150 — protective action advised.")
    if pred_aqi > 150:
        st.error("**Alert:** Predicted AQI exceeds 150 within the next hour.")


def render_danger_alert(aqi: int) -> None:
    if aqi > 150:
        st.error("Dangerous air quality on live reading (AQI > 150).")


def tenant_dashboard(data_store: Dict[str, Dict]) -> None:
    st.markdown("#### Tenant dashboard")
    selected_room = st.selectbox("Select your room", ROOMS, index=0, key="tenant_room")

    room_data = preprocess_data(data_store[selected_room])
    predictions = {
        "aqi": room_data["predicted"]["aqi"],
        "temperature": room_data["predicted"]["temperature"],
        "humidity": room_data["predicted"]["humidity"],
    }
    aqi = int(room_data["current"]["aqi"])
    status, message, color = get_aqi_health_info(aqi)

    render_digital_twin_representation(selected_room, room_data, predictions)

    render_aqi_badge(color, status, aqi)
    render_health_risk_panel(aqi)

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
    st.markdown("##### Health intelligence")
    st.metric("Composite health score", score)
    st.write(f"Status: {status_label}")
    st.info(advice)

    render_smart_alerts(aqi, predictions)
    render_danger_alert(aqi)

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### AQI trend (smoothed history and +1 hour point)")
    trend_df = build_trend_data(aqi, predicted_aqi=predictions["aqi"], seed_salt=hash(selected_room) % 997)
    st.line_chart(trend_df, use_container_width=True)


def owner_dashboard(data_store: Dict[str, Dict]) -> None:
    st.markdown("#### Owner / operator dashboard")
    selected_room = st.selectbox("Select room", ROOMS, index=0, key="owner_room")

    room_data = preprocess_data(data_store[selected_room])
    predictions = {
        "aqi": room_data["predicted"]["aqi"],
        "temperature": room_data["predicted"]["temperature"],
        "humidity": room_data["predicted"]["humidity"],
    }
    aqi = int(room_data["current"]["aqi"])
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

    humidity = float(room_data["current"]["humidity"])
    quality, quality_msg = get_property_quality(humidity)
    st.markdown("##### Property condition (humidity-driven)")
    st.write(f"Status: {quality}")
    if quality.startswith("Good"):
        st.info(quality_msg)
    elif quality.startswith("Moderate"):
        st.warning(quality_msg)
    else:
        st.error(quality_msg)

    render_smart_alerts(aqi, predictions)
    render_danger_alert(aqi)

    act1, _ = st.columns([1, 3])
    with act1:
        if st.button("Notify tenant", key="notify_btn"):
            st.success(f"Notification queued for **{TENANT_MAP[selected_room]}**.")

    st.markdown("##### Room comparison (all rooms)")
    room_rows = []
    for room_name in ROOMS:
        cur = data_store[room_name]["current"]
        room_status, _, room_color = get_aqi_health_info(int(cur["aqi"]))
        room_rows.append(
            {
                "Room": room_name,
                "Status": f"{room_status}",
                "AQI": int(cur["aqi"]),
                "Temperature (°C)": float(cur["temperature"]),
                "Humidity (%)": float(cur["humidity"]),
            }
        )
    room_df = pd.DataFrame(room_rows).sort_values("AQI", ascending=True).reset_index(drop=True)
    avg_aqi = round(float(room_df["AQI"].mean()), 1)
    st.metric("Average AQI (all rooms)", avg_aqi)
    st.dataframe(room_df, use_container_width=True, hide_index=True)

    best_room = room_df.iloc[0]
    worst_room = room_df.iloc[-1]
    c_best, c_worst = st.columns(2)
    with c_best:
        st.success(f"Best air quality: **{best_room['Room']}** (AQI {best_room['AQI']}).")
    with c_worst:
        st.error(f"Needs attention: **{worst_room['Room']}** (AQI {worst_room['AQI']}).")

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### AQI trend (operations view)")
    trend_df = build_trend_data(aqi, predicted_aqi=predictions["aqi"], seed_salt=hash(selected_room) % 997)
    st.line_chart(trend_df, use_container_width=True)


def render_status_bar(last_update: datetime) -> None:
    st.markdown(
        f'<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:0.75rem;">'
        f'<span><b>Last updated</b> — {last_update.strftime("%Y-%m-%d %H:%M:%S")}</span>'
        f"<span><b>ML</b> — Multi-output ridge regression (+1 h)</span>"
        f"<span><b>Twin</b> — Stateful drift simulation</span>"
        f"<span><b>Mode</b> — Academic demo (no authentication)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_controls(data_store_ready: bool) -> bool:
    st.sidebar.header("Controls")
    st.sidebar.caption(
        "Live readings advance automatically (simulated polling). "
        "Manual refresh applies another drift step from the last state per room."
    )
    refresh_clicked = st.sidebar.button("Refresh live data", use_container_width=True)
    if data_store_ready:
        st.sidebar.caption(f"Rooms in memory: {len(ROOMS)}")
    return refresh_clicked


def main() -> None:
    """Application entry point."""
    setup_page()

    if _HAS_AUTOREFRESH and st_autorefresh is not None:
        st_autorefresh(interval=5000, key="auto_refresh")
    elif not st.session_state.get("_autorefresh_hint_shown"):
        st.session_state["_autorefresh_hint_shown"] = True
        st.sidebar.warning(
            "Auto-refresh is off: add **streamlit-autorefresh** to `requirements.txt` at your repo root, "
            "then redeploy. Use **Refresh live data** for manual updates."
        )

    refresh_clicked = render_sidebar_controls("data_store" in st.session_state)

    if "data_store" not in st.session_state:
        st.session_state["data_store"] = get_sensor_data(None)
        st.session_state["last_update"] = datetime.now()
    else:
        st.session_state["data_store"] = get_sensor_data(st.session_state["data_store"])
        st.session_state["last_update"] = datetime.now()

    if refresh_clicked:
        st.session_state["data_store"] = get_sensor_data(st.session_state["data_store"])
        st.session_state["last_update"] = datetime.now()

    hydrate_predictions(st.session_state["data_store"])
    weights = get_ml_model()

    render_status_bar(st.session_state["last_update"])
    st.info("Live simulation updates approximately every 5 seconds; use **Refresh live data** for an immediate step.")

    st.markdown("### Model Input Data (Features)")

    sample_inputs = []
    for room in ROOMS:
        cur = st.session_state["data_store"][room]["current"]
        sample_inputs.append(
            {
                "Room": room,
                "AQI": cur["aqi"],
                "Temperature": cur["temperature"],
                "Humidity": cur["humidity"],
            }
        )

    df_inputs = pd.DataFrame(sample_inputs)
    st.dataframe(df_inputs)

    render_system_architecture()

    st.markdown("## Layer documentation")
    d1, d2 = st.columns(2)
    with d1:
        render_data_source_layer()
    with d2:
        render_data_processing_layer()

    render_machine_learning_section(weights)
    render_system_modules()
    render_model_evaluation(weights)
    render_simulation_engine()
    render_visualization_layer_note()

    tab_tenant, tab_owner = st.tabs(["Tenant dashboard", "Owner dashboard"])
    with tab_tenant:
        tenant_dashboard(st.session_state["data_store"])
    with tab_owner:
        owner_dashboard(st.session_state["data_store"])

    st.markdown("---")
    st.caption("AirAware — Digital Twin system development demonstration (Streamlit)")


if __name__ == "__main__":
    main()
