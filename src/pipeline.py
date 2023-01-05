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
    pos = nx.spectral_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)

    print(f"\nRunning algo {algo_choice}")
    dist, quad_term, sol, err, sol_graph = choose_algo[algo_choice](
        graph, *args, **kwargs
    )
    print(f"Cost: {dist:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}")

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
    timer = checkpoint()
    results = {}
    pos = nx.spectral_layout(graph)

    if gen_data:
        add_random_weights(graph, plot, pos)
        add_random_distributions(graph, plot, pos)
        timer("Time spent generating data on the graph")

    for algo_choice in algo_choices:
        print(f"\nRunning algo {algo_choice}")
        runtime, (cost, quad_term, sol, err, sol_graph) = return_runtime(
            choose_algo[algo_choice]
        )(graph, *args, **kwargs)
        print(
            f"Cost: {cost:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}, runtime: {runtime:.2f}"
        )

        if plot:
            plot_transportation_plan(sol_graph, pos, algo_choice)

        timer(f"Time spent on algo {algo_choice}")

        results[algo_choice] = {
            "cost": cost,
            "quadratic_term": quad_term,
            "error": err,
            "runtime": runtime,
        }

    return results


def timed_pipeline(
    graph: nx.Graph, algo_choice: str, plot: bool = True, *args, **kwargs
) -> Tuple[float, float, np.ndarray, float, float]:
    pos = nx.spectral_layout(graph)

    add_random_weights(graph, plot, pos)
    add_random_distributions(graph, plot, pos)

    print(f"Running algo {algo_choice}")
    runtime, (dist, quad_term, sol, err, sol_graph) = return_runtime(
        choose_algo[algo_choice]
    )(graph, *args, **kwargs)
    print(
        f"Cost: {dist:.2f}, quadratic term: {quad_term:.2f}, error: {err:.2f}, runtime: {runtime:.2f} s"
    )

    if plot:
        plot_transportation_plan(sol_graph, pos, algo_choice)

    return dist, quad_term, sol, err, runtime


@timeit
def kanto_pipeline(graph: nx.Graph, plot: bool = True, gen_data: bool = True) -> None:
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

    records["runtime"] /= n_runs_per_graph


@timeit
def full_pipeline(
    graph: nx.Graph, n_runs_per_graph: int, nonzero_ratio: float, *args, **kwargs
) -> Dict[str, Union[float, int, List[np.ndarray]]]:
    exclude_sinkhorn = True
    results = {
        algo: {
            "cost": 0.0,
            "quadratic_term": 0.0,
            "error": 0.0,
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
                runtime, (dist, quad_term, sol, err, sol_graph) = return_runtime(
                    choose_algo[algo]
                )(graph_copy, *args, **kwargs)
                print(
                    f"cost: {dist:.2f}, quadratic term: {quad_term:.2f}, err: {err:.2f}, runtime: {runtime:.2f} s"
                )
                update_records(results[algo], dist, quad_term, err, sol, runtime)
        print("")

    for record in results.values():
        average_records(record, n_runs_per_graph)

    return results
