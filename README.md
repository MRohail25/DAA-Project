# 🚦 Karachi Traffic AI

**Optimal route planning across Karachi using a from-scratch A\* Search algorithm, driven by AI-predicted traffic.**

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Most student pathfinding projects call `nx.astar_path()` and call it a day. This project implements **A\* Search from scratch** — binary heap, admissible heuristic, the whole algorithm — and wraps it in a real, deployable traffic-routing dashboard.

Built for a Design & Analysis of Algorithms (DAA) course, designed like a product.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [The Algorithm: A\* Search](#the-algorithm-a-search)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing & Validation](#testing--validation)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [Author](#author)

---

## Overview

Karachi Traffic AI predicts current traffic conditions with a machine learning model, uses that prediction to weight every road segment in a graph of Karachi landmarks, and then finds the fastest route through that graph using a hand-implemented A\* Search — not a library call.

The result is a dark-themed interactive dashboard where you can pick an origin and destination, see the optimal route rendered on a live map, and inspect exactly how the algorithm found it: nodes expanded, execution time, and path cost, all in one view.

## Features

- 🧠 **Traffic prediction** — a `RandomForestRegressor` trained on historical hourly vehicle-count data predicts current congestion.
- 🔍 **A\* Search, from scratch** — binary min-heap priority queue, admissible straight-line-distance heuristic, `f(n) = g(n) + h(n)`. No `networkx.astar_path()`.
- 🗺️ **Live route rendering** — real road geometry pulled from OSRM and rendered on an interactive dark-mode Folium map.
- 📊 **Algorithm Insights panel** — nodes expanded, execution time (ms), path cost, and node exploration order, so the algorithm's behavior is fully observable, not just its output.
- 📈 **24-hour traffic forecast** — a Plotly chart showing predicted congestion across the day, with an AI-generated travel recommendation.
- ✅ **Validated, not assumed** — A\*'s optimality is cross-checked against Dijkstra's algorithm; see [Testing & Validation](#testing--validation).

## How It Works

<p align="center">
  <img src="docs/architecture-pipeline.png" alt="Development pipeline: Requirements Analysis, Design, Implementation, Testing, Documentation & Deployment" width="850">
</p>

<p align="center">
  <img src="docs/flowchart.png" alt="System flowchart: user input to rendered optimal route" width="420">
</p>

1. **User selects** an origin and destination landmark.
2. **Traffic is predicted** for the current hour using the trained Random Forest model.
3. **The graph is built**: every road's weight = real distance × a traffic-scaling factor.
4. **Custom A\* Search** explores the graph and returns the lowest-cost path.
5. **The route is rendered** on the map, alongside live algorithm performance metrics.

## The Algorithm: A\* Search

A\* evaluates each node using `f(n) = g(n) + h(n)`:

- `g(n)` — the exact, traffic-weighted cost from the start to node `n`.
- `h(n)` — an estimate of the remaining cost to the goal (straight-line distance).

Because `h(n)` never overestimates the true remaining road distance, it is **admissible** — which guarantees A\* returns the mathematically optimal path, while typically expanding far fewer nodes than an uninformed search like Dijkstra's algorithm.

Core loop, from [`astar.py`](DAA/astar.py):

```python
while open_heap:
    _, _, current = heapq.heappop(open_heap)
    if current not in open_set:
        continue
    open_set.discard(current)
    nodes_expanded += 1

    if current == goal:
        return _reconstruct_path(came_from, current), g_score[goal], nodes_expanded

    for neighbor, weight in graph.get(current, []):
        tentative_g = g_score[current] + weight
        if tentative_g < g_score[neighbor]:
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
            heapq.heappush(open_heap, (f_score[neighbor], next(counter), neighbor))
            open_set.add(neighbor)
```

**Complexity:** O(E log V) time with a binary heap, O(V) space.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| UI / Dashboard | Streamlit |
| Traffic Prediction | scikit-learn (Random Forest Regressor) |
| Pathfinding | Custom A\* Search (binary heap, no library pathfinding) |
| Map Rendering | Folium (CartoDB dark tiles) |
| Route Geometry | OSRM (Open Source Routing Machine) |
| Charts | Plotly |
| Validation | NetworkX (Dijkstra cross-check, testing only) |

## Project Structure

```
DAA/
├── app.py              # Streamlit dashboard — UI, route search, algorithm insights
├── astar.py             # Standalone A* Search implementation (the core algorithm)
├── utils.py              # Traffic forecast curve, risk thresholds, AI recommendations
├── graph.py             # Landmark coordinates, road connections, distance function
├── train_model.py        # Trains the Random Forest traffic-prediction model
├── map.py                # One-off script to visualize Karachi's road network (OSMnx)
├── data.csv              # Historical hourly traffic data
├── traffic_model.pkl      # Trained model artifact
└── requirements.txt        # Python dependencies
```

## Getting Started

**Prerequisites:** Python 3.11+ and pip.

```bash
# Clone the repo
git clone https://github.com/<your-username>/karachi-traffic-ai.git
cd karachi-traffic-ai/DAA

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), pick an origin and destination in the **Route Finder** tab, and click **Find Best Route**. Switch to **Algorithm Insights** to see exactly how A\* found it.

## Testing & Validation

The A\* implementation was validated against 7 test cases covering correctness, edge cases, performance, and integration — each run directly against `astar.py`:

| Test | Scenario | Result |
|---|---|---|
| TC-01 | Valid path: Saddar → Malir | ✅ Path found, cost = 23.68 km |
| TC-02 | A\* cost vs. Dijkstra (optimality check) | ✅ Identical: 23.676361 km, diff = 0 |
| TC-03 | Origin equals Destination | ✅ Trivial path, cost = 0 |
| TC-04 | Unreachable / disconnected node | ✅ Correctly returns `found=False` |
| TC-05 | Cost scales with traffic factor | ✅ 23.57 km (low) → 24.74 km (high) |
| TC-06 | Execution time across 56 landmark pairs | ✅ avg 0.008 ms, max 0.047 ms |
| TC-07 | Edge weight formula verification | ✅ Matches `dist × (1 + traffic/10000)` exactly |

**7/7 passed.** TC-02 is the key correctness guarantee — it confirms A\*'s result is identical to Dijkstra's provably-optimal result, not just fast-looking.

## Roadmap

- [ ] Real-time traffic data integration (currently prediction-based, not live)
- [ ] Multi-modal routing (walking, public transport)
- [ ] Turn-by-turn navigation mode
- [ ] Deploy to Streamlit Community Cloud for a public live demo

## Acknowledgments

Built for the **Design & Analysis of Algorithms (DAA)** course at IQRA University, under the guidance of **Sir Dr. Mohammad Affan Alim**.

## Author

**M. Rohail Hussain**
BSCS Student · IQRA University
[LinkedIn](#) · [GitHub](#)

---

<p align="center"><i>If this project was useful or interesting, consider giving it a ⭐</i></p>
