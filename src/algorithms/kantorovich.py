from typing import Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

from src.algorithms.utils import compute_cost_matrix, collect_graph


# noinspection PyUnusedLocal
def t_wasserstein_1(
    graph: nx.Graph,
    verbose: bool = True,
    return_uncollected_graph: bool = False,
    alpha: float = 1.0,
) -> Tuple[float, float, np.ndarray, float, nx.Graph]:
    """
    Computes the Wasserstein-1 distance on a weighted graph that contains two distributions.
    Relies on the Kantorovich formulation (the variable is the transportation map between each pair of nodes).
    """
    # retrieving the two distributions
    rho_0 = np.array(list(nx.get_node_attributes(graph, "rho_0").values()))
    rho_1 = np.array(list(nx.get_node_attributes(graph, "rho_1").values()))

    n_nodes = len(graph)
    cost_matrix, shortest_paths = compute_cost_matrix(graph)

    transportation_plan = cp.Variable((n_nodes, n_nodes), nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, transportation_plan)))
    problem = cp.Problem(
        objective,
        [
            cp.matmul(transportation_plan, np.ones(n_nodes)) == rho_0,
            cp.matmul(transportation_plan.T, np.ones(n_nodes)) == rho_1,
        ],
    )
    problem.solve(cp.ECOS)

    if verbose:
        nonzero = np.count_nonzero(transportation_plan.value)
        total = n_nodes**2
        print(f"Optimal transportation plan (number of nonzero: {nonzero} / {total}):")
        print(np.round(transportation_plan.value, 2))

    # creating a copy of the graph that will only have the edges where there is a movement of mass
    uncollected_graph = nx.create_empty_copy(graph)
    # adding an attribute that will yield the value of the transportation plan
    uncollected_graph.add_edges_from(
        [
            (origin, destination, {"ot": value})
            for origin, row in enumerate(transportation_plan.value)
            for destination, value in enumerate(row)
            if value > 1e-3
        ]
    )

    # adding the values of the ot on the initial graph by putting back the plan on the actual edges
    collected_graph = collect_graph(graph, transportation_plan.value, shortest_paths)

    flow = np.array(list(nx.get_edge_attributes(collected_graph, "ot").values()))
    quadratic_term = float(np.sum(np.square(flow)))

    if verbose:
        nonzero = np.count_nonzero(flow)
        print(f"Optimal flow (number of nonzero: {nonzero} / {graph.size()}):")
        print(np.round(flow, 2))

    if return_uncollected_graph:
        # noinspection PyTypeChecker
        return (
            problem.value,
            quadratic_term,
            flow,
            collected_graph,
            uncollected_graph,
        )
    else:
        return (
            problem.value,
            quadratic_term,
            flow,
            0.,
            collected_graph,
        )
