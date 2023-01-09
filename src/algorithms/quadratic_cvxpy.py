from typing import Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

from src.algorithms.utils import (
    extract_graph_info,
    add_ot_to_edges,
    recreate_ot_graph,
)


def regularized_j_wasserstein_1(
    graph: nx.Graph, alpha: float, verbose: bool = True
) -> Tuple[float, float, np.ndarray, float, float, nx.Graph]:
    """
    Computes the regularized Wasserstein-1 distance on a weighted graph that contains two distributions.
    Relies on an alternative formulation (the variable indicates the flow on each edge).
    """
    n_edges = graph.size()
    f, cost_vector, incidence_matrix = extract_graph_info(graph)

    flow = cp.Variable(n_edges, nonneg=True)
    objective = cp.Minimize(
        cp.sum(cp.multiply(cost_vector, flow)) + alpha / 2 * cp.sum(cp.square(flow))
    )
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

    cost = cost_vector @ flow.value
    quadratic_term = float(np.sum(np.square(flow.value)))
    err = np.linalg.norm(incidence_matrix.T @ flow.value - f)

    return (
        cost,
        quadratic_term,
        np.array(flow.value),
        err,
        1 - nonzero / n_edges,
        sol_graph,
    )
