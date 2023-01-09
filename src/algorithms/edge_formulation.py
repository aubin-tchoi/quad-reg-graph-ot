from typing import Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

from src.algorithms.utils import (
    extract_graph_info,
    add_ot_to_edges,
    recreate_ot_graph,
)


# noinspection PyUnusedLocal
def j_wasserstein_1(
    graph: nx.Graph, verbose: bool = True, alpha: float = 1.0
) -> Tuple[float, float, np.ndarray, float, float, nx.Graph]:
    """
    Computes the Wasserstein-1 distance on a weighted graph that contains two distributions.
    Relies on an alternative formulation (the variable indicates the flow on each edge).
    """
    n_edges = graph.size()
    f, cost_vector, incidence_matrix = extract_graph_info(graph)

    flow = cp.Variable(n_edges, nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(cost_vector, flow)))
    problem = cp.Problem(
        objective,
        [cp.matmul(incidence_matrix.T, flow) == f],
    )
    problem.solve(cp.ECOS)

    nonzero = np.count_nonzero(flow.value)
    if verbose:
        print(f"Optimal flow (number of nonzero: {nonzero} / {n_edges}):")
        print(np.round(flow.value, 2))

    add_ot_to_edges(graph, flow.value)
    sol_graph = recreate_ot_graph(graph, flow.value)

    quadratic_term = float(np.sum(np.square(flow.value)))
    err = np.linalg.norm(incidence_matrix.T @ flow.value - f)

    return (
        problem.value,
        quadratic_term,
        flow.value,
        err,
        1 - nonzero / n_edges,
        sol_graph,
    )
