"""
The graphs generated for the experiments should be connected and bidirectional
to ensure the feasibility of the optimization problem.
"""
import networkx as nx
import numpy as np


def create_path_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a path graph (linear graph).
    """
    return nx.path_graph(graph_size).to_directed()


def create_cycle_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a cycle graph (circular graph).
    """
    return nx.cycle_graph(graph_size).to_directed()


def create_wheel_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a wheel graph (see Networkx's documentation).
    """
    return nx.wheel_graph(graph_size).to_directed()


def create_complete_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a complete graph.
    """
    return nx.complete_graph(graph_size).to_directed()


def create_watts_strogatz_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a connected graph using the Watts–Strogatz random graph generation model.
    """
    return nx.connected_watts_strogatz_graph(graph_size, 3, 0.4).to_directed()


def create_gnp_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a connected graph using the Erdős–Rényi random graph generation model.
    The loop looks ugly but something similar is actually performed in nx.connected_watts_strogatz_graph.
    """
    n_attempts = 100
    for i in range(n_attempts):
        # for p > (1 + eps) ln(n) / n, the Erdős–Rényi graph should be connected almost surely
        gnp_graph = nx.gnp_random_graph(graph_size, 2 * np.log(graph_size) / graph_size)
        if nx.is_connected(gnp_graph):
            print(
                f"Sparsity of the graph: {gnp_graph.size() / len(gnp_graph) ** 2 * 100:.2f}%."
            )
            return gnp_graph.to_directed()


def create_bipartite_graph(graph_size: int) -> nx.DiGraph:
    """
    Creates a connected bipartite graph using the Erdős–Rényi random graph generation model.
    The loop looks ugly but something similar is actually performed in nx.connected_watts_strogatz_graph.
    """
    n_attempts = 100
    for i in range(n_attempts):
        # for p > (1 + eps) ln(n) / n, the Erdős–Rényi graph should be connected almost surely
        bipartite_graph = nx.bipartite.random_graph(
            graph_size // 2, graph_size // 2, 2 * np.log(graph_size) / graph_size
        )
        if nx.is_connected(bipartite_graph):
            print(
                f"Sparsity of the graph: {bipartite_graph.size() / len(bipartite_graph) ** 2 * 100:.2f}%."
            )
            return bipartite_graph.to_directed()


def create_graph(graph_size: int, graph_type: str) -> nx.DiGraph():
    """
    Creates a graph using one of the functions above depending on the input graph type.
    """
    return (
        create_bipartite_graph(graph_size)
        if graph_type == "bipartite"
        else create_cycle_graph(graph_size)
        if graph_type == "cycle"
        else create_path_graph(graph_size)
        if graph_type == "path"
        else create_complete_graph(graph_size)
        if graph_type == "complete"
        else create_gnp_graph(graph_size)
    )
