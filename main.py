"""
Main file for the project.
Every function of this file can be considered as an entry-point for an experiment.
"""
import json
import pickle
import sys
import warnings
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from src import *

warnings.filterwarnings("ignore")
sns.set_theme()


def draw_non_uniqueness_example() -> None:
    """
    Draws a small graph that can be used as an example to display
    the non-uniqueness of the solutions of the non-regularized problem.
    """
    graph = nx.cycle_graph(3)

    positions = nx.circular_layout(graph)

    graph.edges[0, 1]["weight"] = 0.5
    graph.edges[1, 2]["weight"] = 0.5
    graph.edges[0, 2]["weight"] = 1.0

    graph.nodes[0]["f"] = -1
    graph.nodes[1]["f"] = 0
    graph.nodes[2]["f"] = 1

    edge_labels = {(u, v): weight for u, v, weight in graph.edges.data("weight")}
    node_labels = {node: graph.nodes[node]["f"] for node in graph.nodes()}

    plt.figure()
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=edge_labels, font_size=20
    )
    nx.draw_networkx_labels(graph, positions, labels=node_labels, font_size=20)
    nx.draw(
        graph,
        positions,
        node_color=list(node_labels.values()),
        node_size=2000,
        cmap=plt.cm.Oranges,
    )
    plt.show()


def show_example_impact_regularization() -> None:
    """
    Provides an example to show the impact of the quadratic regularization on a very small graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (0, {"rho_1": 1, "rho_0": 0}),
            (1, {"rho_1": 0, "rho_0": 0}),
            (2, {"rho_1": 0, "rho_0": 0}),
            (3, {"rho_1": 0, "rho_0": 0}),
            (4, {"rho_1": 0, "rho_0": 0}),
            (5, {"rho_1": 0, "rho_0": 1}),
            (6, {"rho_1": 0, "rho_0": 0}),
        ]
    )
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 6),
            (6, 3),
            (1, 4),
            (2, 3),
            (2, 6),
            (6, 4),
            (3, 5),
            (4, 5),
            (0, 6),
            (6, 5),
        ]
    )
    graph = graph.to_directed()
    nx.set_edge_attributes(graph, 1, "weight")
    alphas = [25, 10, 4, 2, 1, 0.5, 0.1]
    for alpha in alphas:
        basic_pipeline(
            deepcopy(graph),
            "cvxpy_reg",
            plot=True,
            alpha=alpha,
            verbose=False,
            gen_data=False,
        )


def run_all_algos(graph_size: int = 60, graph_type: str = "path") -> None:
    """
    Runs all the algorithms on a same graph with randomly generated data.
    """
    np.random.seed(0)
    graph = create_graph(graph_size, graph_type)
    add_random_weights(graph)
    add_random_distributions(graph)

    for algo_choice in choose_algo:
        basic_pipeline(
            deepcopy(graph),
            algo_choice,
            plot=False,
            alpha=5,
            verbose=False,
            gen_data=False,
        )


def compare_algo_sinkhorn(graph_size: int = 6, graph_type: str = "path") -> None:
    """
    Compares the algorithm with the cvxpy implementation on a bipartite graph.
    """
    np.random.seed(0)
    alphas = [10, 5, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    graph = create_graph(graph_size, graph_type)

    if graph_type == "bipartite":
        pos = nx.bipartite_layout(graph, list(graph.nodes)[: len(graph) // 2])
        add_random_weights(graph, True, pos)
        add_random_distributions(graph, True, pos)
    else:
        add_random_weights(graph)
        add_random_distributions(graph)

    results = {alpha: {} for alpha in alphas}
    for alpha in alphas:
        print(f"\n ---- alpha: {alpha} ----")
        results[alpha] = comparison_pipeline(
            deepcopy(graph),
            ["cvxpy_reg", "no_reg", "stable_sinkhorn", "algo"],
            alpha=alpha,
            verbose=False,
            gen_data=False,
        )

    print(json.dumps(results, indent=2))


def monitor_algo_evolution(graph_size: int = 60, graph_type: str = "bipartite") -> None:
    """
    Plots various graph describing the behavior of the algorithm over the iterations.
    """
    np.random.seed(0)
    alphas = [10, 5, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    graph = create_graph(graph_size, graph_type)

    add_random_weights(graph, False)
    add_random_distributions(graph, False)

    results = {alpha: () for alpha in alphas}
    for alpha in alphas:
        print(f"\n ---- alpha: {alpha} ----")
        _, __, sol, ___ = basic_pipeline(
            deepcopy(graph), "cvxpy_reg", alpha=alpha, plot=False
        )
        results[alpha] = basic_pipeline(
            deepcopy(graph),
            "algo",
            alpha=alpha,
            verbose=True,
            gen_data=False,
            plot=False,
            optimal_sol=sol,
        )

    # removing unserializable np.ndarray
    for alpha, values in results.items():
        results[alpha] = {
            "cost": values[0],
            "quadratic_term": values[1],
            "error": values[3],
        }
    print(json.dumps(results, indent=2))


def complete_experiment(
    graph_type: str = "bipartite", save_results: bool = True
) -> None:
    """
    Runs an experiment with various graph sizes, various values of alpha
    and saves the results in both a pickle file and a json file.
    The execution can be interrupted by pressing CTRL+C, the files will still be written with the current results.
    """
    graph_sizes = [50, 100, 500, 1000]
    alphas = [10, 5, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    n_runs_per_graph = 10
    proportion_of_sink = 0.1

    results = {graph_size: {} for graph_size in graph_sizes}
    try:
        for graph_size in graph_sizes:
            graph = create_graph(graph_size, graph_type)
            add_random_weights(graph, plot=False)

            for alpha in alphas:
                print(f"\nGraph size: {graph_size}, alpha: {alpha}")
                results[graph_size][alpha] = full_pipeline(
                    graph,
                    n_runs_per_graph,
                    proportion_of_sink,
                    alpha=alpha,
                    verbose=False,
                )
    # you can stop the execution using a CTRL+C
    except KeyboardInterrupt:
        print(
            f"\nExecution interrupted, {'saving the results' if save_results else 'exiting'}."
        )

    if save_results:
        identifier = sys.argv[1] if len(sys.argv) > 1 else graph_type
        identifier += "_" if identifier else ""

        pickle_file_path = (
            f"data/results_{identifier}{f'{datetime.now():%m_%d-%H_%M}'}.pickle"
        )
        print(f"\nSaving the data in a pickle file '{pickle_file_path}'.")
        with open(pickle_file_path, "wb+") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        json_file_path = (
            f"data/results_{identifier}{f'{datetime.now():%m_%d-%H_%M}'}.json"
        )
        print(f"Saving the data in a json file '{json_file_path}'.")
        remove_np_arrays(results)

        with open(json_file_path, "w+", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    complete_experiment()
