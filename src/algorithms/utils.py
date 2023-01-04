from typing import Tuple, List

import networkx as nx
import numpy as np
from copy import deepcopy


def compute_cost_matrix(graph: nx.Graph) -> Tuple[np.ndarray, List[List[List[int]]]]:
    """
    Computes the cost matrix on a graph.
    """
    n_nodes = len(graph)
    cost_matrix = np.zeros((n_nodes, n_nodes))
    shortest_paths = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]

    # using networkx built-in shortest path (dijkstra algorithm)
    for origin, path_lengths in dict(nx.all_pairs_dijkstra(graph)).items():
        for destination, length in path_lengths[0].items():
            cost_matrix[origin, destination] = length
        for destination, path in path_lengths[1].items():
            shortest_paths[origin][destination] = path

    return cost_matrix, shortest_paths


def collect_graph(
    graph: nx.Graph,
    transportation_plan: np.ndarray,
    shortest_paths: List[List[List[int]]],
) -> nx.Graph:
    """
    Constructs a graph obtained by adding the values of a transportation plan between couples of nodes
    on all the edges that form the path between the two nodes.
    """
    collected_graph = deepcopy(graph)
    nx.set_edge_attributes(collected_graph, 0., "ot")
    for origin, row in enumerate(transportation_plan):
        for destination, value in enumerate(row):
            if value and value > 1e-6:
                u = origin
                for v in shortest_paths[origin][destination][1:]:
                    collected_graph.edges[u, v]["ot"] += value
                    u = v

    return collected_graph
