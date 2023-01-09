import random
from typing import Tuple, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def add_random_weights(
    graph: nx.Graph,
    plot: bool = True,
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
) -> None:
    """
    Adds random weights on the edges of the graph.
    The weights are integers chosen randomly between 0 and 10 included.
    """
    # noinspection PyArgumentList
    for (_, __, w) in graph.edges(data=True):
        w["weight"] = random.random()

    if plot:
        print("Plotting the weights on each edge.")
        plt.figure()

        positions = positions or nx.spectral_layout(graph)
        edge_labels = {(u, v): weight for u, v, weight in graph.edges.data("weight")}

        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels)
        nx.draw(graph, positions, with_labels=True, node_size=500)

        plt.show()


def add_random_distributions(
    graph: nx.Graph,
    plot: bool = True,
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
    distribution: str = "dirichlet",
    nonzero_ratio: float = 1.0,
) -> None:
    """
    Adds two random distributions on the nodes of the graph.
    The distributions are added as node attributes (rho_0 and rho_1).
    They are sampled using a Dirichlet distribution.
    """
    n_nonzero = int(nonzero_ratio * len(graph))
    # sampling the distributions using Dirichlet distributions (no need to divide by the sum)
    if distribution == "dirichlet":
        rho_0 = np.random.dirichlet(np.ones(n_nonzero), size=1)[0]
        rho_1 = np.random.dirichlet(np.ones(n_nonzero), size=1)[0]
    else:
        rho_0 = np.random.random(n_nonzero)
        rho_0 = rho_0 / np.sum(rho_0)
        rho_1 = np.random.random(n_nonzero)
        rho_1 = rho_1 / np.sum(rho_1)

    source_indexes = np.random.choice(len(graph), size=n_nonzero, replace=False)
    sink_indexes = np.random.choice(len(graph), size=n_nonzero, replace=False)
    source, sink = 0, 0
    # noinspection PyArgumentList
    for i, (_, w) in enumerate(graph.nodes(data=True)):
        if i in source_indexes:
            w["rho_1"] = rho_1[source]
            source += 1
        else:
            w["rho_1"] = 0.0
        if i in sink_indexes:
            w["rho_0"] = rho_0[sink]
            sink += 1
        else:
            w["rho_0"] = 0.0

    if plot:
        print("Plotting the two distributions on each node.")
        positions = positions or nx.spectral_layout(graph)
        plt.figure(figsize=(14, 7))

        ax1 = plt.subplot(121)
        distribution = {
            node: round(graph.nodes[node]["rho_0"], 2) for node in graph.nodes()
        }
        nx.draw_networkx_labels(graph, positions, labels=distribution, font_size=10)
        nx.draw(
            graph,
            positions,
            node_color=list(distribution.values()),
            node_size=800,
            ax=ax1,
        )

        ax2 = plt.subplot(122)
        distribution = {
            node: round(graph.nodes[node]["rho_1"], 2) for node in graph.nodes()
        }
        nx.draw_networkx_labels(graph, positions, labels=distribution, font_size=10)
        nx.draw(
            graph,
            positions,
            node_color=list(distribution.values()),
            node_size=800,
            ax=ax2,
        )

        plt.show()


def plot_transportation_plan(
    graph: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    title: str = "",
    savefig: bool = False,
) -> None:
    """
    Plots a graph with the associated transportation plan.

    Args:
        graph: networkx graph whose nodes have attributes 'rho_0' and 'rho_1' for the distributions
            and attribute 'ot' on the edges for the value of the transportation plan.
        positions: positions of each node.
        title: title of the plot
        savefig: set to True to save the figure.
    """
    plt.figure()
    edge_labels = {(u, v): round(ot, 3) for u, v, ot in graph.edges.data("ot") if ot}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels)
    nx.draw(graph, positions, with_labels=True, node_size=200, arrowsize=15)
    plt.title(title)
    if savefig:
        plt.savefig(
            f"imgs/{title.replace(', ', '-').replace(': ', '_').replace('.', ',')}"
        )

    plt.show()
