from typing import Tuple, List, Dict, Union

import networkx as nx
import numpy as np
from copy import deepcopy
from src.algorithms import (
    timeit,
    return_runtime,
    checkpoint,
    choose_algo,
    t_wasserstein_1,
)
from src.data_generation import (
    add_random_weights,
    add_random_distributions,
    plot_transportation_plan,
)


@timeit
def basic_pipeline(
    graph: nx.Graph,
    algo_choice: str,
    plot: bool = True,
    gen_data: bool = True,
    *args,
    **kwargs,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Basic pipeline that runs a single algorithm on a graph.
    Relays argument to the function that runs the method through the args and kwargs.

    Args:
        graph: The graph to run the method on.
        algo_choice: A string that indicates the method to use. Refer to choose_algo for link between name and function.
        plot: Whether the initial data and output transport plan should be displayed or not.
        gen_data: Whether random data should be generated on the graph or not.
    """
    pos = nx.kamada_kawai_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)

    print(f"\nRunning algo {algo_choice}")
    dist, quad_term, sol, err, sparsity, sol_graph = choose_algo[algo_choice](
        graph, *args, **kwargs
    )
    print(
        f"Cost: {dist:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}, sparsity: {sparsity:.2f}"
    )

    if plot:
        plot_transportation_plan(sol_graph, pos, algo_choice)

    return dist, quad_term, sol, err


def comparison_pipeline(
    graph: nx.Graph,
    algo_choices: List[str],
    plot: bool = True,
    gen_data: bool = True,
    *args,
    **kwargs,
) -> Dict[str, Dict[str, Union[float, float, np.ndarray, float]]]:
    """
    Pipeline that runs a list of algorithms on a graph for comparison.
    Relays argument to the function that runs the method through the args and kwargs.

    Args:
        graph: The graph to run the method on.
        algo_choices: Strings that indicate the method to use. Refer to choose_algo for link between name and function.
        plot: Whether the initial data and output transport plan should be displayed or not.
        gen_data: Whether random data should be generated on the graph or not.
    """
    timer = checkpoint()
    results = {}
    pos = nx.spectral_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)
        timer("Time spent generating data on the graph")

    for algo_choice in algo_choices:
        print(f"\nRunning algo {algo_choice}")
        runtime, (cost, quad_term, sol, err, sparsity, sol_graph) = return_runtime(
            choose_algo[algo_choice]
        )(graph, *args, **kwargs)
        print(
            f"Cost: {cost:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}, "
            f"sparsity: {sparsity:.2f}, runtime: {runtime:.2f}"
        )

        if plot:
            plot_transportation_plan(
                sol_graph, pos, f"{algo_choice}, alpha: {kwargs['alpha']}", savefig=True
            )

        timer(f"Time spent on algo {algo_choice}")

        results[algo_choice] = {
            "cost": cost,
            "quadratic_term": quad_term,
            "error": err,
            "runtime": runtime,
        }

    return results


def timed_pipeline(
    graph: nx.Graph,
    algo_choice: str,
    plot: bool = True,
    gen_data: bool = True,
    *args,
    **kwargs,
) -> Tuple[float, float, np.ndarray, float, float]:
    """
    Basic pipeline that runs a single algorithm on a graph and displays its execution time.
    Relays argument to the function that runs the method through the args and kwargs.

    Args:
        graph: The graph to run the method on.
        algo_choice: A string that indicates the method to use. Refer to choose_algo for link between name and function.
        plot: Whether the initial data and output transport plan should be displayed or not.
        gen_data: Whether random data should be generated on the graph or not.
    """
    pos = nx.spectral_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)

    print(f"Running algo {algo_choice}")
    runtime, (dist, quad_term, sol, err, sparsity, sol_graph) = return_runtime(
        choose_algo[algo_choice]
    )(graph, *args, **kwargs)
    print(
        f"Cost: {dist:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}, "
        f"sparsity: {sparsity:.2f}, runtime: {runtime:.2f} s"
    )

    if plot:
        plot_transportation_plan(sol_graph, pos, algo_choice)

    return dist, quad_term, sol, err, runtime


@timeit
def kanto_pipeline(graph: nx.Graph, plot: bool = True, gen_data: bool = True) -> None:
    """
    Pipeline specific to the Kantorovich formulation that displays the fact that the solution has to be put back
    on the actual edges using the shortest paths.

    Args:
        graph: The graph to run the method on.
        plot: Whether the initial data and output transport plan should be displayed or not.
        gen_data: Whether random data should be generated on the graph or not.
    """
    pos = nx.spectral_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)

    # noinspection PyTupleAssignmentBalance
    dist, quad_term, sol, collected_sol, uncollected_sol = t_wasserstein_1(
        graph, return_uncollected_graph=True
    )
    print(f"\nWasserstein-1 distance: {dist:.4f}")
    if plot:
        plot_transportation_plan(uncollected_sol, pos, "Kantorovich")
        # noinspection PyTypeChecker
        plot_transportation_plan(collected_sol, pos, "Kantorovich")


def update_records(
    records: Dict[str, Union[float, int, List[np.ndarray]]],
    dist: float,
    quadratic_term: float,
    error: float,
    sparsity: float,
    solution: np.ndarray,
    runtime: float,
) -> None:
    """
    Adds a record to a dict of records taking into account unsuccessful runs.
    """
    if dist < 1e-12 or dist == np.inf or np.isnan(dist):
        records["fails"] += 1
    else:
        records["cost"] += dist
        records["quadratic_term"] += quadratic_term
        records["error"] += error
        records["sparsity"] += sparsity
    records["runtime"] += runtime
    records["solutions"].append(solution)


def average_records(
    records: Dict[str, Union[float, int, List[np.ndarray]]],
    n_runs_per_graph: int,
) -> None:
    """
    Averages the values of the costs and quadratic terms measured by dividing by the number of successful runs.
    Also updates the average runtime.
    """
    n_successful_runs = n_runs_per_graph > records["fails"]

    if n_successful_runs > 0:
        records["cost"] /= n_successful_runs
        records["quadratic_term"] /= n_successful_runs
        records["error"] /= n_successful_runs
        records["sparsity"] /= n_successful_runs

    records["runtime"] /= n_runs_per_graph


def remove_np_arrays(
    records: Dict[
        int, Dict[float, Dict[str, Dict[str, Union[float, int, List[np.ndarray]]]]]
    ],
) -> None:
    """
    Removes unserializable np.ndarray.
    """
    for graph_size in records.values():
        for alpha in graph_size.values():
            for algo in alpha.values():
                algo.pop("solutions")


@timeit
def full_pipeline(
    graph: nx.Graph, n_runs_per_graph: int, nonzero_ratio: float, *args, **kwargs
) -> Dict[str, Union[float, int, List[np.ndarray]]]:
    """
    Complete pipeline that runs all the algorithms on a graph and logs various averaged metrics.
    Relays argument to the function that runs the method through the args and kwargs.

    Args:
        graph: The graph to run the method on.
        n_runs_per_graph: Number of runs to perform on each graph.
        nonzero_ratio: The proportion of nodes that will be sinks or sources.
    """
    exclude_sinkhorn = False
    results = {
        algo: {
            "cost": 0.0,
            "quadratic_term": 0.0,
            "error": 0.0,
            "sparsity": 0.0,
            "runtime": 0,
            "fails": 0,
            "solutions": [],
        }
        for algo in choose_algo
        if "sinkhorn" not in algo or not exclude_sinkhorn
    }

    for n_run in range(n_runs_per_graph):
        graph_copy = deepcopy(graph)
        add_random_distributions(graph_copy, plot=False, nonzero_ratio=nonzero_ratio)

        for algo in choose_algo:
            if "sinkhorn" not in algo or not exclude_sinkhorn:
                print(
                    f"-- Run number {n_run:>{len(str(n_runs_per_graph))}} of algo {algo:<15}:",
                    end=" ",
                )
                runtime, (
                    dist,
                    quad_term,
                    sol,
                    err,
                    sparsity,
                    sol_graph,
                ) = return_runtime(choose_algo[algo])(graph_copy, *args, **kwargs)
                print(
                    f"cost: {dist:.2f}, quadratic term: {quad_term:.2f}, err: {err:.2f}, "
                    f"sparsity: {sparsity:.2f}, runtime: {runtime:.2f} s"
                )
                update_records(
                    results[algo], dist, quad_term, err, sparsity, sol, runtime
                )
        print("")

    for record in results.values():
        average_records(record, n_runs_per_graph)

    return results
