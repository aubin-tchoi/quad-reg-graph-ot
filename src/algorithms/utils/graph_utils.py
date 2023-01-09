from copy import deepcopy
from typing import Tuple, List

import networkx as nx
import numpy as np


def extract_graph_info(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts information stored in a Networkx graph.
    """
    rho_0 = np.array(list(nx.get_node_attributes(graph, "rho_0").values()))
    rho_1 = np.array(list(nx.get_node_attributes(graph, "rho_1").values()))
    f = rho_1 - rho_0
    assert np.sum(f) < 1e-6  # should be zero

    cost_vector = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    incidence_matrix = nx.incidence_matrix(graph, oriented=True).T.toarray()

    return f, cost_vector, incidence_matrix


def add_ot_to_edges(graph: nx.Graph, flow: np.ndarray) -> None:
    """
    Adds the value of the optimal transportation map as edge attribute of a Networkx graph.
    """
    edge_indexes = [e for e in graph.edges()]
    for idx, value in enumerate(flow):
        if value and value > 0:
            graph.edges[edge_indexes[idx]]["ot"] = value


def recreate_ot_graph(graph: nx.Graph, flow: np.ndarray) -> nx.Graph:
    """
    Recreates a separate graph that only contains edges where the optimal transport is positive.
    """
    edge_indexes = [e for e in graph.edges()]
    sol_graph = nx.create_empty_copy(graph)
    for idx, value in enumerate(flow):
        if value and value > 1e-6:
            sol_graph.add_edge(*edge_indexes[idx], ot=value)

    return sol_graph


def compute_cost_matrix(graph: nx.Graph) -> Tuple[np.ndarray, List[List[List[int]]]]:
    """
    Computes the cost matrix on a graph.
    Useful in the Kantorovich formulation where a transport map is defined between each pair of vertices.
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
    Useful in the Kantorovich formulation where a transport map is defined between each pair of vertices.
    """
    collected_graph = deepcopy(graph)
    nx.set_edge_attributes(collected_graph, 0.0, "ot")
    for origin, row in enumerate(transportation_plan):
        for destination, value in enumerate(row):
            if value and value > 1e-6:
                u = origin
                for v in shortest_paths[origin][destination][1:]:
                    collected_graph.edges[u, v]["ot"] += value
                    u = v

    return collected_graph
