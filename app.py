import streamlit as st
import folium
import joblib
import requests
from datetime import datetime
from streamlit_folium import st_folium

from graph import LANDMARKS, ROADS
from astar import build_adjacency_list, astar_search
from utils import (
    predict_current_traffic,
    get_24h_forecast,
    get_traffic_thresholds,
    classify_traffic,
    get_ai_recommendation,
    build_forecast_chart,
)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi Traffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #0B0F1A;
    --bg-surface: #131A2A;
    --bg-surface-2: #1B2436;
    --border-hair: #232D42;
    --accent: #5B6EF5;
    --accent-2: #22D3A5;
    --warn: #F5A623;
    --danger: #F5556B;
    --text-primary: #EAF0FB;
    --text-muted: #8C9AB8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .stat-value, .brand-title { font-family: 'Space Grotesk', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

#MainMenu, footer, header {visibility: hidden;}
.stApp { background: var(--bg-void); color: var(--text-primary); }

section[data-testid="stSidebar"] {
    background: #0D1220;
    border-right: 1px solid var(--border-hair);
}
section[data-testid="stSidebar"] * { color: var(--text-primary); }

.brand-row { display:flex; align-items:center; gap:10px; padding: 4px 0 18px 0; }
.brand-title { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.3px; }
.brand-sub { color: var(--text-muted); font-size: 0.72rem; }

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display:flex; align-items:center; gap:10px;
    padding:10px 14px; border-radius:10px; margin-bottom:4px;
    border:1px solid transparent; cursor:pointer;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.04);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(91,110,245,0.35), rgba(91,110,245,0.06));
    border-color: rgba(91,110,245,0.45);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display:none;
}

.profile-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-hair);
    border-radius: 12px;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
}
.profile-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.85rem; color:white; flex-shrink:0;
}
.profile-name { font-size: 0.82rem; font-weight: 600; }
.profile-role { font-size: 0.7rem; color: var(--text-muted); }

.hero {
    padding: 1.7rem 2rem;
    border-radius: 18px;
    background: radial-gradient(120% 180% at 0% 0%, #16203a 0%, #0d1424 60%);
    border: 1px solid var(--border-hair);
    margin-bottom: 1.1rem;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 4px 0; }
.hero p { color: var(--text-muted); margin: 0 0 14px 0; font-size: 0.92rem; }

.pipeline { display:flex; align-items:center; flex-wrap:wrap; gap:6px; }
.pipe-chip {
    background: var(--bg-surface-2);
    border: 1px solid var(--border-hair);
    color: var(--text-primary);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace;
}
.pipe-arrow { color: var(--accent); font-size: 0.85rem; }

.glass-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-hair);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.glass-card h4 {
    margin: 0 0 10px 0; font-size: 0.85rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}

.stat-chip {
    background: var(--bg-surface);
    border: 1px solid var(--border-hair);
    border-radius: 14px;
    padding: 14px 16px;
    height: 100%;
}
.stat-icon { font-size: 1.1rem; margin-bottom: 6px; }
.stat-label { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 4px; }
.stat-value { font-size: 1.4rem; font-weight: 700; }
.stat-unit { font-size: 0.78rem; color: var(--text-muted); font-weight: 500; }
.stat-sub { font-size: 0.76rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

.summary-row {
    display:flex; justify-content:space-between; align-items:center;
    padding: 7px 0; border-bottom: 1px dashed var(--border-hair);
    font-size: 0.85rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-label { color: var(--text-muted); }
.summary-value { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

.route-chip {
    display: inline-block;
    background: var(--bg-surface-2);
    color: var(--text-primary);
    border: 1px solid var(--border-hair);
    padding: 5px 13px;
    border-radius: 999px;
    margin: 3px 4px 3px 0;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}
.arrow { color: var(--accent); margin: 0 3px; }

.ai-rec {
    background: linear-gradient(120deg, rgba(91,110,245,0.12), rgba(34,211,165,0.06));
    border: 1px solid rgba(91,110,245,0.35);
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 0.86rem;
    line-height: 1.45;
}

.badge-pill {
    display:inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}

/* Streamlit widget restyle */
div[data-baseweb="select"] > div {
    background: var(--bg-surface) !important;
    border-color: var(--border-hair) !important;
    border-radius: 10px !important;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    height: 2.9em;
    background: linear-gradient(120deg, var(--accent), #7C8CFF);
    color: white;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(120deg, #4a5cf0, var(--accent));
    color: white;
}
div[data-testid="stMetric"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-hair);
    border-radius: 12px;
    padding: 12px 14px;
}
div[data-testid="stMetricLabel"] { color: var(--text-muted); }
hr { border-color: var(--border-hair); }
.footer-note { text-align:center; color: var(--text-muted); font-size: 0.78rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load ML model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

model = load_model()

# ── Session defaults ─────────────────────────────────────────────────────
locations = list(LANDMARKS.keys())
if "origin" not in st.session_state:
    st.session_state.origin = locations[0]
if "destination" not in st.session_state:
    st.session_state.destination = locations[1]

def swap_locations():
    st.session_state.origin, st.session_state.destination = (
        st.session_state.destination, st.session_state.origin,
    )

# ── Sidebar (brand + nav + profile) ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-row">
        <div style="font-size:1.5rem;">🚦</div>
        <div>
            <div class="brand-title">Karachi Traffic AI</div>
            <div class="brand-sub">A* Route Optimizer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🧭  Route Finder", "📊  Algorithm Insights", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="profile-card">
        <div class="profile-avatar">RH</div>
        <div>
            <div class="profile-name">M. Rohail Hussain</div>
            <div class="profile-role">BSCS Student · DAA Project</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── OSRM road geometry ───────────────────────────────────────────────────
def get_road_coords(start_name, end_name):
    lat1, lon1 = LANDMARKS[start_name]
    lat2, lon2 = LANDMARKS[end_name]
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    )
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        coords = data["routes"][0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in coords]
    except Exception:
        return [[lat1, lon1], [lat2, lon2]]

# ── Map builder (dark theme) ─────────────────────────────────────────────
def build_map(path):
    m = folium.Map(
        location=[24.8607, 67.0104],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    for a, b in ROADS:
        lat1, lon1 = LANDMARKS[a]
        lat2, lon2 = LANDMARKS[b]
        folium.PolyLine(
            [[lat1, lon1], [lat2, lon2]],
            color="#3A4356", weight=2, opacity=0.6,
        ).add_to(m)

    for i in range(len(path) - 1):
        road_coords = get_road_coords(path[i], path[i + 1])
        # glow halo behind the main line
        folium.PolyLine(road_coords, color="#5B6EF5", weight=14, opacity=0.15).add_to(m)
        folium.PolyLine(
            road_coords, color="#22D3A5", weight=5, opacity=0.95,
            tooltip=f"{path[i]} → {path[i + 1]}",
        ).add_to(m)

    for name, (lat, lon) in LANDMARKS.items():
        if name == path[0]:
            icon = folium.Icon(color="green", icon="play", prefix="fa")
        elif name == path[-1]:
            icon = folium.Icon(color="red", icon="flag", prefix="fa")
        elif name in path:
            icon = folium.Icon(color="blue", icon="circle", prefix="fa")
        else:
            icon = folium.Icon(color="lightgray", icon="circle", prefix="fa")

        folium.Marker([lat, lon], tooltip=name, popup=name, icon=icon).add_to(m)

    return m

def stat_chip(icon, label, value, unit="", sub="", sub_color="var(--text-muted)"):
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-icon">{icon}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value} <span class="stat-unit">{unit}</span></div>
        <div class="stat-sub" style="color:{sub_color};">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: ROUTE FINDER
# ══════════════════════════════════════════════════════════════════════
if page == "🧭  Route Finder":

    st.markdown("""
    <div class="hero">
        <h1>Smart Route. Optimal Path.</h1>
        <p>AI-predicted traffic scored through a hand-implemented A* search — not a library call.</p>
        <div class="pipeline">
            <span class="pipe-chip">🌦 Predict Traffic</span>
            <span class="pipe-arrow">➜</span>
            <span class="pipe-chip">🕸 Build Graph</span>
            <span class="pipe-arrow">➜</span>
            <span class="pipe-chip">🔍 Run A* Search</span>
            <span class="pipe-arrow">➜</span>
            <span class="pipe-chip">🗺 Render Route</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_o, col_swap, col_d, col_btn = st.columns([4, 0.6, 4, 2.5])
    with col_o:
        st.selectbox("Origin", locations, key="origin")
    with col_swap:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("⇅", on_click=swap_locations, use_container_width=True)
    with col_d:
        st.selectbox("Destination", locations, key="destination")
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        find_route = st.button("✨ Find Best Route", use_container_width=True)

    origin, destination = st.session_state.origin, st.session_state.destination

    if find_route:
        if origin == destination:
            st.warning("⚠️ Origin and destination cannot be the same.")
        else:
            with st.spinner("Predicting traffic and running A* search..."):
                traffic_now = predict_current_traffic(model)
                adjacency_list = build_adjacency_list(traffic_now)
                result = astar_search(adjacency_list, origin, destination)

                low_t, high_t = get_traffic_thresholds()
                level, level_color = classify_traffic(traffic_now, low_t, high_t)

            if not result["found"]:
                st.error(f"❌ No route found between {origin} and {destination}.")
            else:
                st.session_state["last_result"] = result
                st.session_state["last_route"] = (origin, destination)
                st.session_state["last_traffic"] = (traffic_now, level, level_color)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        o, d = st.session_state["last_route"]
        traffic_now, level, level_color = st.session_state["last_traffic"]
        path = result["path"]
        avg_speed_kmh = 30
        travel_time_min = (result["cost"] / avg_speed_kmh) * 60

        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_chip("🚗", "Predicted Traffic", f"{round(traffic_now)}", "veh/hr",
                       f"● {level} traffic", level_color)
        with c2:
            stat_chip("📏", "Distance", f"{result['cost']:.1f}", "km", f"{len(path)} stops")
        with c3:
            stat_chip("⏱️", "Est. Time", f"{round(travel_time_min)}", "min", "@ 30 km/h avg")
        with c4:
            stat_chip("🧮", "A* Nodes Expanded", f"{result['nodes_expanded']}", "",
                       f"{result['exec_time_ms']:.2f} ms")

        st.write("")
        map_col, side_col = st.columns([2.2, 1])

        with map_col:
            st.markdown("<div class='glass-card'><h4>Route</h4>", unsafe_allow_html=True)
            chips = ""
            for i, node in enumerate(path):
                chips += f'<span class="route-chip">{node}</span>'
                if i < len(path) - 1:
                    chips += '<span class="arrow">➜</span>'
            st.markdown(chips + "</div>", unsafe_allow_html=True)

            route_map = build_map(path)
            st_folium(route_map, use_container_width=True, height=460, returned_objects=[])

        with side_col:
            st.markdown(f"""
            <div class="glass-card">
                <h4>Route Summary</h4>
                <div class="summary-row"><span class="summary-label">Distance</span><span class="summary-value">{result['cost']:.2f} km</span></div>
                <div class="summary-row"><span class="summary-label">Est. Time</span><span class="summary-value">{round(travel_time_min)} min</span></div>
                <div class="summary-row"><span class="summary-label">Stops</span><span class="summary-value">{len(path)}</span></div>
                <div class="summary-row"><span class="summary-label">Traffic Level</span><span class="summary-value" style="color:{level_color};">{level}</span></div>
                <div class="summary-row"><span class="summary-label">Algorithm</span><span class="summary-value">A* Search</span></div>
                <div class="summary-row"><span class="summary-label">Nodes Expanded</span><span class="summary-value">{result['nodes_expanded']}</span></div>
                <div class="summary-row"><span class="summary-label">Exec. Time</span><span class="summary-value">{result['exec_time_ms']:.2f} ms</span></div>
            </div>
            """, unsafe_allow_html=True)

            hours, values = get_24h_forecast(model)
            low_t, high_t = get_traffic_thresholds()
            st.markdown("<div class='glass-card'><h4>24h Traffic Forecast</h4>", unsafe_allow_html=True)
            fig = build_forecast_chart(hours, values, low_t, high_t)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            rec_text = get_ai_recommendation(level, traffic_now)
            st.markdown(f"""
            <div class="glass-card">
                <h4>🤖 AI Recommendation</h4>
                <div class="ai-rec">{rec_text}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Set your origin and destination above, then click **Find Best Route**.")

# ══════════════════════════════════════════════════════════════════════
# PAGE: ALGORITHM INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊  Algorithm Insights":
    st.markdown("""
    <div class="hero">
        <h1>Algorithm Insights</h1>
        <p>A transparent look at how the custom A* search (astar.py) explores the graph.</p>
    </div>
    """, unsafe_allow_html=True)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        o, d = st.session_state["last_route"]

        st.markdown(f"<p class='mono' style='color:var(--text-muted);'>Last run: {o} → {d}</p>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            stat_chip("🧮", "Nodes Expanded", result["nodes_expanded"])
        with c2:
            stat_chip("⚡", "Execution Time", f"{result['exec_time_ms']:.3f}", "ms")
        with c3:
            stat_chip("🎯", "Path Cost", f"{result['cost']:.2f}", "km")

        st.write("")
        st.markdown("<div class='glass-card'><h4>Node Exploration Order</h4>", unsafe_allow_html=True)
        chips = "".join(
            f'<span class="route-chip">{i+1}. {n}</span>'
            for i, n in enumerate(result["explored_order"])
        )
        st.markdown(chips + "</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4>Why A* here?</h4>
            <ul style="font-size:0.9rem; line-height:1.7; color:var(--text-primary); margin:0; padding-left:1.2rem;">
                <li><b>Heuristic h(n):</b> straight-line distance between two landmarks — admissible, since real road distance is never shorter.</li>
                <li><b>Cost g(n):</b> actual weighted distance travelled so far, scaled by AI-predicted traffic.</li>
                <li><b>f(n) = g(n) + h(n):</b> the priority queue always expands the most promising node first.</li>
                <li>Because h(n) is admissible and consistent, A* is guaranteed to return the <b>optimal</b> path — same guarantee as Dijkstra, but expanding far fewer nodes.</li>
                <li><b>Complexity:</b> O(E log V) time with a binary heap, O(V) space.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run a search on the **Route Finder** page first to see performance metrics here.")

# ══════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="hero">
        <h1>About This Project</h1>
        <p>Design & Analysis of Algorithms coursework, built as a real, deployable tool.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4>Pipeline</h4>
        <ol style="font-size:0.9rem; line-height:1.8; color:var(--text-primary); margin:0; padding-left:1.2rem;">
            <li>A <code>RandomForestRegressor</code> predicts expected vehicle volume for the current hour/day/month.</li>
            <li>Predicted traffic scales the weight of every road edge (heavier traffic → higher effective cost).</li>
            <li>A custom <b>A*</b> implementation (<code>astar.py</code>) searches the weighted graph for the lowest-cost path — built from scratch with a binary heap, no NetworkX pathfinding.</li>
            <li>Route geometry is fetched from <b>OSRM</b> and rendered on an interactive <b>Folium</b> map.</li>
        </ol>
    </div>
    <div class="glass-card">
        <h4>Tech Stack</h4>
        <p style="font-size:0.9rem;">Python · Streamlit · scikit-learn · Folium · OSRM · Plotly · custom A* search</p>
    </div>
    <div class="glass-card">
        <h4>Author</h4>
        <p style="font-size:0.9rem;">M. Rohail Hussain — BSCS Student<br>Course: Design & Analysis of Algorithms (DAA)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">Built with Streamlit · Custom A* Search · Random Forest</div>', unsafe_allow_html=True)
