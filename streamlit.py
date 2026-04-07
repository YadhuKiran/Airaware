import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Light professional dashboard styling (Streamlit-safe).
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
        .twin-section h3, .twin-section p {
            color: #0f172a !important;
            opacity: 1 !important;
        }
        /* Improve readability in room-comparison table. */
        div[data-testid="stDataFrame"] div[role="table"] {
            font-size: 0.98rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(f"🌬️ {APP_TITLE}")
    st.caption("Digital Twin + ML — smart air quality monitoring prototype")


def get_aqi_health_info(aqi: int) -> Tuple[str, str, str]:
    """
    Map AQI value to EPA-style band for messaging and badge color.
    """
    if aqi < 50:
        return "Good", "Air quality is satisfactory; little or no risk.", "green"
    if aqi <= 100:
        return "Moderate", "Acceptable for most; sensitive people may feel effects.", "orange"
    return "Unhealthy", "Everyone may begin to feel health effects at longer exposure.", "red"


def get_health_risk_level(aqi: int) -> Tuple[str, str]:
    """
    Compact risk label for panels (Safe / Moderate Risk / High Risk).
    """
    if aqi < 50:
        return "Safe", "green"
    if aqi <= 100:
        return "Moderate Risk", "orange"
    return "High Risk", "red"


def get_status_emoji(color: str) -> str:
    """Visual indicator used across badges, tables, and risk labels."""
    return {"green": "🟢", "orange": "🟠", "red": "🔴"}.get(color, "⚪")


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

    # Next hour: mild coupling + noise (demo-friendly, explains well in viva).
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
    ridge = 0.15  # mild L2 — stabilizes fit on synthetic data
    # Normal equations: (X'X + λI) W = X'Y
    ata = Xd.T @ Xd + ridge * np.eye(n_features)
    aty = Xd.T @ y.astype(np.float64)
    weights: np.ndarray = np.linalg.solve(ata, aty)
    return weights


def get_ml_model() -> np.ndarray:
    """Cache learned weights in session_state (train once per browser session)."""
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = train_ml_pipeline()
    return st.session_state["ml_model"]


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
        # Filled by predict_aqi() once ML runs (keeps structure backward-compatible).
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
    # Random walk with tight steps — visually smoother than sparse big jumps.
    values: List[float] = []
    v = float(max(18, current_aqi - rng.uniform(10, 22)))
    for _ in range(points - 1):
        v += rng.gauss(0, 2.2)
        v = float(np.clip(v, 12, 260))
        values.append(v)
    values.append(float(current_aqi))

    hist = [int(round(x)) for x in values]

    if predicted_aqi is None:
        df = pd.DataFrame({"Time": past_times, "🟢 AQI (live trend)": hist})
        return df.set_index("Time")

    fut_time = now + timedelta(hours=1)
    forecast_series = [np.nan] * (points - 1) + [current_aqi, predicted_aqi]
    hist_ext = hist + [np.nan]
    times_ext = past_times + [fut_time]
    df = pd.DataFrame(
        {
            "Time": times_ext,
            "🟢 AQI (live trend)": hist_ext,
            "🔮 AQI (+1h forecast)": forecast_series,
        }
    )
    return df.set_index("Time")


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
    """A / B / C feature: explicit health risk label."""
    label, tone = get_health_risk_level(aqi)
    icon = "✅" if tone == "green" else ("⚠️" if tone == "orange" else "🚨")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(f"{icon} Health risk", f"{get_status_emoji(tone)} {label}")
    with c2:
        st.caption("Based on current AQI snapshot (demo rule-based mapping).")


def render_current_metrics(room_data: Dict, *, heading: str = "📡 Current state (live)") -> None:
    """Live Digital Twin metrics — clearly separated from forecast."""
    st.markdown(f"##### {heading}")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "AQI",
        room_data["current"]["aqi"],
        help="AQI (Air Quality Index) indicates how clean or polluted the air is. Lower AQI means better air quality.",
    )
    c2.metric("Temperature (°C)", room_data["current"]["temperature"])
    c3.metric("Humidity (%)", room_data["current"]["humidity"])


def render_predictions(predictions: Dict) -> None:
    """Forecast section: ML output for +1 hour horizon."""
    st.markdown("##### 🔮 Predicted state (+1 hour, ML)")
    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Predicted AQI",
        predictions["aqi"],
        help="Predicted using machine learning model.",
    )
    p2.metric("Predicted temp (°C)", predictions["temperature"])
    p3.metric("Predicted humidity (%)", predictions["humidity"])


def render_forecast_insights(current_aqi: int, predictions: Dict) -> None:
    """Future warning + risk on forecast path."""
    pred_aqi = int(predictions["aqi"])
    col_a, col_b = st.columns(2)
    with col_a:
        if pred_aqi > current_aqi:
            st.warning("📈 Air quality **expected to worsen** in the next hour (predicted AQI is higher).")
        elif pred_aqi < current_aqi:
            st.success("📉 Predicted AQI is **lower** — conditions may improve.")
        else:
            st.info("➡️ Predicted AQI is **near** the current level.")
    with col_b:
        risk_label, tone = get_health_risk_level(pred_aqi)
        st.caption("Forecast risk band (from predicted AQI)")
        if tone == "green":
            st.success(f"Forecast: {get_status_emoji(tone)} **{risk_label}**")
        elif tone == "orange":
            st.warning(f"Forecast: {get_status_emoji(tone)} **{risk_label}**")
        else:
            st.error(f"Forecast: {get_status_emoji(tone)} **{risk_label}**")


def render_smart_alerts(current_aqi: int, predictions: Dict) -> None:
    """Current danger + future danger (assignment alerts)."""
    pred_aqi = int(predictions["aqi"])
    if current_aqi > 150:
        st.error("🛑 **Danger:** Current AQI exceeds 150 — take protective action.")
    if pred_aqi > 150:
        st.error("⚠️ **Future danger:** Predicted AQI crosses 150 within the next hour.")


def render_danger_alert(aqi: int) -> None:
    """Back-compat wrapper; smart alerts cover the full rule set."""
    if aqi > 150:
        st.error("🛑 Dangerous air quality detected (live reading).")


def tenant_dashboard(data_store: Dict[str, Dict]) -> None:
    """Tenant-facing dashboard."""
    st.header("🏠 Tenant dashboard")
    selected_room = st.selectbox("Select your room", ROOMS, index=0, key="tenant_room")

    room_data = preprocess_data(data_store[selected_room])
    predictions = {
        "aqi": room_data["predicted"]["aqi"],
        "temperature": room_data["predicted"]["temperature"],
        "humidity": room_data["predicted"]["humidity"],
    }
    aqi = int(room_data["current"]["aqi"])
    status, message, color = get_aqi_health_info(aqi)

    st.markdown(
        f'<div class="twin-section"><h3 style="margin:0 0 0.25rem 0;">🛰️ Live twin — {selected_room}</h3>'
        f"<p style='margin:0;opacity:0.8'>Current vs forecast from the same entity.</p></div>",
        unsafe_allow_html=True,
    )

    render_aqi_badge(color, status, aqi)
    render_health_risk_panel(aqi)

    c_left, c_right = st.columns([1.1, 1])
    with c_left:
        render_current_metrics(room_data)
    with c_right:
        st.markdown("##### 🩺 Health advisory")
        st.info(message)

    render_smart_alerts(aqi, predictions)
    render_danger_alert(aqi)

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### 📈 AQI trend (smoothed history + forecast point)")
    trend_df = build_trend_data(aqi, predicted_aqi=predictions["aqi"], seed_salt=hash(selected_room) % 997)
    st.line_chart(trend_df, use_container_width=True)


def owner_dashboard(data_store: Dict[str, Dict]) -> None:
    """Owner-facing dashboard with tenant context."""
    st.header("🏢 Owner / operator dashboard")
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
        st.markdown(
            f'<div class="twin-section"><h3 style="margin:0 0 0.25rem 0;">📍 Property — {selected_room}</h3>'
            f"<p style='margin:0;'>Current vs forecast from the same entity.</p></div>",
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown(f"**👤 Tenant:** {TENANT_MAP[selected_room]}")

    render_aqi_badge(color, status, aqi)
    render_health_risk_panel(aqi)

    render_current_metrics(room_data)
    st.caption(message)

    render_smart_alerts(aqi, predictions)
    render_danger_alert(aqi)

    act1, _ = st.columns([1, 3])
    with act1:
        if st.button("📧 Notify tenant", key="notify_btn"):
            st.success(f"Notification queued for **{TENANT_MAP[selected_room]}**.")

    st.markdown("##### 🏘️ Room comparison (all rooms)")
    room_rows = []
    for room_name in ROOMS:
        cur = data_store[room_name]["current"]
        room_status, _, room_color = get_aqi_health_info(int(cur["aqi"]))
        room_rows.append(
            {
                "Room": room_name,
                "Status": f"{get_status_emoji(room_color)} {room_status}",
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
        st.success(f"✅ Best room: **{best_room['Room']}** (AQI {best_room['AQI']})")
    with c_worst:
        st.error(f"🚨 Worst room: **{worst_room['Room']}** (AQI {worst_room['AQI']})")

    render_predictions(predictions)
    render_forecast_insights(aqi, predictions)

    st.markdown("##### 📈 AQI trend (operations view)")
    trend_df = build_trend_data(aqi, predicted_aqi=predictions["aqi"], seed_salt=hash(selected_room) % 997)
    st.line_chart(trend_df, use_container_width=True)


def render_sidebar_controls() -> Tuple[str, bool]:
    """Role + manual refresh (triggers new drifted reading)."""
    st.sidebar.header("🎛️ Controls")
    role = st.sidebar.radio("Choose role", ["Tenant", "Owner"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Refresh live data** requests a new simulated poll; readings drift from the last state per room."
    )
    refresh_clicked = st.sidebar.button("🔄 Refresh live data", use_container_width=True)
    return role, refresh_clicked


def render_status_bar(last_update: datetime) -> None:
    """Top status strip for presentation polish."""
    st.markdown(
        f'<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:0.5rem;">'
        f'<span><b>🕐 Last updated</b> — {last_update.strftime("%Y-%m-%d %H:%M:%S")}</span>'
        f"<span><b>🧠 Model</b> — Multi-output ridge regression (+1h, NumPy)</span>"
        f"<span><b>🪢 Twin</b> — Stateful drift simulation</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_system_overview() -> None:
    """Top-level project explanation for presentation and viva clarity."""
    st.markdown("### 🧭 System Overview")
    st.write(
        "AirAware is a Digital Twin system where each room is represented as a live virtual entity. "
        "It simulates real-time air quality sensor behavior, applies machine learning to forecast AQI "
        "for the next hour, and supports both tenants and owners with actionable insights."
    )
    st.markdown("### 🔁 System Flow")
    st.markdown(
        "**IoT/Data → Data Processing → Machine Learning → Digital Twin → Dashboard**"
    )
    st.divider()


def main() -> None:
    """Application entry point."""
    setup_page()
    role, refresh_clicked = render_sidebar_controls()

    # Session holds Digital Twin snapshots; evolve on refresh from previous live metrics.
    if "data_store" not in st.session_state:
        st.session_state["data_store"] = get_sensor_data(None)
        st.session_state["last_update"] = datetime.now()
    elif refresh_clicked:
        st.session_state["data_store"] = get_sensor_data(st.session_state["data_store"])
        st.session_state["last_update"] = datetime.now()

    # One inference pass for all rooms so owner/tenant views share the same twin snapshot.
    hydrate_predictions(st.session_state["data_store"])

    render_status_bar(st.session_state["last_update"])
    render_system_overview()

    if role == "Tenant":
        tenant_dashboard(st.session_state["data_store"])
    else:
        owner_dashboard(st.session_state["data_store"])

    st.markdown("---")
    st.caption("Developed as a Digital Twin + AI system for smart air quality monitoring")


if __name__ == "__main__":
    main()
