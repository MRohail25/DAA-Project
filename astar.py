"""
astar.py
========
Standalone implementation of the A* Search Algorithm.

This module is intentionally decoupled from NetworkX. It builds a plain
adjacency-list graph and implements A* from scratch using a binary heap
(min-priority queue), so the algorithm's mechanics — open set, closed set,
g(n), h(n), f(n) = g(n) + h(n) — are fully visible and explainable.

Time Complexity : O(E log V)  — using a binary heap priority queue
Space Complexity: O(V)        — for g_score, f_score, came_from, open set


"""

import heapq
import itertools
import time

from graph import LANDMARKS, ROADS, calculate_distance


def build_adjacency_list(traffic_factor: float) -> dict:
    """
    Builds a weighted, undirected adjacency list from LANDMARKS/ROADS.

    Edge weight = straight-line distance (km) scaled by current predicted
    traffic (heavier traffic -> higher effective travel cost).

    Returns:
        dict: {node: [(neighbor, weight), ...], ...}
    """
    adjacency_list = {node: [] for node in LANDMARKS}

    for a, b in ROADS:
        lat1, lon1 = LANDMARKS[a]
        lat2, lon2 = LANDMARKS[b]
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        weight = distance * (1 + traffic_factor / 10000)

        adjacency_list[a].append((b, weight))
        adjacency_list[b].append((a, weight))  # undirected graph

    return adjacency_list


def heuristic(node1: str, node2: str) -> float:
    """
    Admissible heuristic h(n): straight-line (Euclidean-on-lat/lon) distance
    in kilometers between two landmarks. Since real road distance is always
    >= straight-line distance, this heuristic never overestimates the true
    cost -> A* is guaranteed to find the optimal path.
    """
    lat1, lon1 = LANDMARKS[node1]
    lat2, lon2 = LANDMARKS[node2]
    return calculate_distance(lat1, lon1, lat2, lon2)


def astar_search(graph: dict, start: str, goal: str) -> dict:
    """
    Runs A* search from `start` to `goal` on the given adjacency list.

    Algorithm:
        1. Maintain an open set (min-heap) ordered by f(n) = g(n) + h(n).
        2. Pop the node with the lowest f(n).
        3. If it's the goal, reconstruct the path and stop.
        4. Otherwise, relax all its neighbors: if a shorter path to a
           neighbor is found via the current node, update g(n)/f(n) and
           push the neighbor onto the open set.
        5. Repeat until the goal is reached or the open set is empty.

    Returns:
        dict with keys:
            found            (bool)
            path             (list[str] | None)
            cost             (float | None)  total path weight
            nodes_expanded   (int)   nodes popped from the open set
            exec_time_ms     (float) wall-clock time taken
    """
    start_time = time.perf_counter()

    # Tie-breaking counter avoids comparing node names directly when
    # two entries share the same f-score.
    counter = itertools.count()

    open_heap = [(0.0, next(counter), start)]
    open_set = {start}

    came_from = {}
    g_score = {node: float("inf") for node in graph}
    g_score[start] = 0.0

    f_score = {node: float("inf") for node in graph}
    f_score[start] = heuristic(start, goal)

    nodes_expanded = 0
    explored_order = []

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current not in open_set:
            continue  # stale entry (already relaxed via a better route)
        open_set.discard(current)

        nodes_expanded += 1
        explored_order.append(current)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return {
                "found": True,
                "path": path,
                "cost": g_score[goal],
                "nodes_expanded": nodes_expanded,
                "explored_order": explored_order,
                "exec_time_ms": (time.perf_counter() - start_time) * 1000,
            }

        for neighbor, weight in graph.get(current, []):
            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    heapq.heappush(
                        open_heap, (f_score[neighbor], next(counter), neighbor)
                    )
                    open_set.add(neighbor)

    # Goal unreachable
    return {
        "found": False,
        "path": None,
        "cost": None,
        "nodes_expanded": nodes_expanded,
        "explored_order": explored_order,
        "exec_time_ms": (time.perf_counter() - start_time) * 1000,
    }


def _reconstruct_path(came_from: dict, current: str) -> list:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ── Quick manual test (run: python astar.py) ────────────────────────────
if __name__ == "__main__":
    test_graph = build_adjacency_list(traffic_factor=50)
    result = astar_search(test_graph, "Saddar", "Malir")

    print("Path found   :", result["found"])
    print("Path         :", " -> ".join(result["path"]))
    print("Total cost   :", round(result["cost"], 3), "km (weighted)")
    print("Nodes expanded:", result["nodes_expanded"])
    print("Exec time (ms):", round(result["exec_time_ms"], 4))
