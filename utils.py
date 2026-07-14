"""
utils.py
========
Support functions for the dashboard that don't belong in app.py (UI),
graph.py (graph data), or astar.py (search algorithm).

Kept separate on purpose: app.py should only orchestrate UI + call these
helpers, not contain data logic. Makes the codebase easier to navigate
and unit-test independently of Streamlit.
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

DATA_PATH = "data.csv"

# Theme colors (kept in sync with the CSS in app.py)
COLOR_LOW = "#22D3A5"
COLOR_MODERATE = "#F5A623"
COLOR_HEAVY = "#F5556B"
COLOR_ACCENT = "#5B6EF5"


def _traffic_bucket(hour: int) -> float:
    """Coarse time-of-day traffic bucket used as a model feature."""
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 0.10   # peak hours
    elif 0 <= hour <= 5:
        return 0.005  # late night
    else:
        return 0.055  # normal hours


def predict_for_hour(model, hour: int, day: int, month: int) -> float:
    """Predict vehicle count for a single hour using the trained model."""
    avg_hourly = _traffic_bucket(hour)
    sample = [[hour, day, month, avg_hourly]]
    return float(model.predict(sample)[0])


def predict_current_traffic(model) -> float:
    now = datetime.now()
    return predict_for_hour(model, now.hour, now.day, now.month)


def get_24h_forecast(model, day: int = None, month: int = None):
    """Predict vehicle count for every hour of today (0-23)."""
    now = datetime.now()
    day = day or now.day
    month = month or now.month
    hours = list(range(24))
    values = [predict_for_hour(model, h, day, month) for h in hours]
    return hours, values


def get_traffic_thresholds(csv_path: str = DATA_PATH):
    """
    Data-driven Low/Moderate/Heavy thresholds using the 33rd/66th
    percentile of historical vehicle counts — avoids hard-coding
    arbitrary magic numbers that wouldn't generalize to new data.
    """
    df = pd.read_csv(csv_path)
    low = df["vehicle_count"].quantile(0.33)
    high = df["vehicle_count"].quantile(0.66)
    return float(low), float(high)


def classify_traffic(value: float, low: float, high: float):
    """Returns (label, color_hex) for a given predicted traffic value."""
    if value <= low:
        return "Low", COLOR_LOW
    elif value <= high:
        return "Moderate", COLOR_MODERATE
    else:
        return "Heavy", COLOR_HEAVY


def get_ai_recommendation(level: str, value: float) -> str:
    value = round(value)
    if level == "Heavy":
        return (
            f"🚨 Heavy traffic predicted (~{value} veh/hr) on this route right now. "
            f"Consider leaving 15–20 minutes earlier, or check back closer to a lighter window."
        )
    elif level == "Moderate":
        return (
            f"⚠️ Moderate traffic expected (~{value} veh/hr). "
            f"Minor delays are possible, but the current route is still efficient."
        )
    else:
        return (
            f"✅ Light traffic expected (~{value} veh/hr). "
            f"Good time to travel — minimal delays anticipated on this route."
        )


def build_forecast_chart(hours, values, low_thresh, high_thresh):
    """24-hour traffic forecast as a dark-themed Plotly area/line chart,
    with points colored by predicted congestion level."""
    now_hour = datetime.now().hour

    point_colors = [
        classify_traffic(v, low_thresh, high_thresh)[1] for v in values
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode="lines+markers",
        line=dict(color="rgba(139,154,184,0.35)", width=2, shape="spline"),
        marker=dict(size=7, color=point_colors, line=dict(width=0)),
        fill="tozeroy",
        fillcolor="rgba(91,110,245,0.08)",
        hovertemplate="%{x}:00 → %{y:.0f} veh/hr<extra></extra>",
    ))

    fig.add_vline(
        x=now_hour, line_width=1.5, line_dash="dot",
        line_color=COLOR_ACCENT, opacity=0.7,
    )

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=170,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8C9AB8", size=11, family="Inter, sans-serif"),
        xaxis=dict(
            showgrid=False, tickmode="array",
            tickvals=[0, 6, 12, 18, 23],
            ticktext=["00", "06", "12", "18", "23"],
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig
