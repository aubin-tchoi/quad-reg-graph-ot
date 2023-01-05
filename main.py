import json
import pickle
import sys
import warnings
from copy import deepcopy
from datetime import datetime

import networkx as nx
import numpy as np

from src import *

warnings.filterwarnings("ignore")


def run_all_algos() -> None:
    np.random.seed(0)
    n_nodes = 10
    path_graph = nx.path_graph(n_nodes).to_directed()

    for algo_choice in choose_algo:
        basic_pipeline(
            deepcopy(path_graph), algo_choice, plot=False, alpha=5, verbose=False
        )


def compare_algo_cvx(graph_size: int = 10):
    """
    Compares the algorithm with the cvxpy implementation on a bipartite graph.
    """
    np.random.seed(0)
    alphas = [10, 5, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    bipartite_graph = create_bipartite_graph(graph_size)
    results = {alpha: {} for alpha in alphas}
    for alpha in alphas:
        results[alpha] = comparison_pipeline(
            deepcopy(bipartite_graph), ["cvxpy_reg", "algo"], alpha=alpha, verbose=False
        )

    print(results)


def complete_experiment() -> None:
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
            bipartite_graph = create_bipartite_graph(graph_size)
            add_random_weights(bipartite_graph, plot=False)

            for alpha in alphas:
                print(f"\nGraph size: {graph_size}, alpha: {alpha}")
                results[graph_size][alpha] = full_pipeline(
                    bipartite_graph,
                    n_runs_per_graph,
                    proportion_of_sink,
                    alpha=alpha,
                    verbose=False,
                )
    # you can stop the execution using a CTRL+C
    except KeyboardInterrupt:
        pass

    identifier = sys.argv[1] if len(sys.argv) > 1 else "bipartite"
    identifier += "_" if identifier else ""

    pickle_file_path = (
        f"data/results_{identifier}{f'{datetime.now():%m-%d-%H_%M}'}.pickle"
    )
    print(f"Saving the data in a pickle file '{pickle_file_path}'.")
    with open(pickle_file_path, "wb+") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    json_file_path = f"data/results_{identifier}{f'{datetime.now():%m-%d-%H_%M}'}.json"
    print(f"Saving the data in a json file '{json_file_path}'.")
    # removing unserializable np.ndarray
    for graph_size in results.values():
        for alpha in graph_size.values():
            for algo in alpha.values():
                algo.pop("solutions")

    with open(json_file_path, "w+", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    complete_experiment()
