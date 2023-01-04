import json
import pickle
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np

from src import *

warnings.filterwarnings("ignore")


def run_all_algos() -> None:
    np.random.seed(0)
    n_nodes = 10
    path_graph = nx.path_graph(n_nodes).to_directed()

    for algo in choose_algo:
        basic_pipeline(deepcopy(path_graph), algo, plot=False, alpha=5, verbose=False)


if __name__ == "__main__":
    graph_sizes = [50, 100, 500, 1000, 5000]
    alphas = [10, 5, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    n_runs_per_graph = 10
    proportion_of_sink = 0.1

    results = {graph_size: {} for graph_size in graph_sizes}
    try:
        for graph_size in graph_sizes:
            gnp_graph = create_gnp_graph(graph_size)
            add_random_weights(gnp_graph, plot=False)

            for alpha in alphas:
                print(f"\nGraph size: {graph_size}, alpha: {alpha}")
                results[graph_size][alpha] = full_pipeline(
                    gnp_graph,
                    n_runs_per_graph,
                    proportion_of_sink,
                    alpha=alpha,
                    verbose=False,
                )
    except KeyboardInterrupt:
        pass

    with open("data/results.pickle", "wb+") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/results.json", "w+", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
