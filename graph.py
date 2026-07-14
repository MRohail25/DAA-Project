import networkx as nx
import math

LANDMARKS = {
    "Saddar": (24.8607, 67.0104),
    "DHA": (24.7960, 67.0700),
    "Gulshan": (24.9214, 67.0978),
    "Clifton": (24.8178, 67.0289),
    "Airport": (24.9008, 67.1681),
    "Malir": (24.8929, 67.1997),
    "North Nazimabad": (24.9497, 67.0368),
    "Lyari": (24.8751, 67.0119)
}

ROADS = [
    ("Saddar", "Clifton"),
    ("Saddar", "Lyari"),
    ("Saddar", "Gulshan"),
    ("Gulshan", "Airport"),
    ("Airport", "Malir"),
    ("Gulshan", "North Nazimabad"),
    ("North Nazimabad", "Lyari"),
    ("Clifton", "DHA")
]


def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt(
        (lat1 - lat2)**2 +
        (lon1 - lon2)**2
    ) * 111


def build_graph(traffic_factor):

    G = nx.Graph()

    for a, b in ROADS:

        lat1, lon1 = LANDMARKS[a]
        lat2, lon2 = LANDMARKS[b]

        distance = calculate_distance(
            lat1,
            lon1,
            lat2,
            lon2
        )

        weight = distance * (
            1 + traffic_factor / 10000
        )

        G.add_edge(
            a,
            b,
            weight=weight
        )

    return G


def heuristic(node1, node2):

    lat1, lon1 = LANDMARKS[node1]
    lat2, lon2 = LANDMARKS[node2]

    return calculate_distance(
        lat1,
        lon1,
        lat2,
        lon2
    )