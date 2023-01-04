from typing import Tuple

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
