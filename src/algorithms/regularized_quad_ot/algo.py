from typing import Tuple

import networkx as nx
import numpy as np

from src.algorithms.regularized_quad_ot.algo_utils import (
    initialize_algorithm,
    edge_added_to_active_set,
    edge_removed_from_active_set,
)
from src.algorithms.graph_utils import extract_graph_info


def regularized_quadratic_ot(
    graph: nx.Graph,
    alpha: float,
    eps: float = 1e-8,
    max_iter: int = 5000,
    verbose: bool = True,
    always_print_stop_criteria: bool = True,
) -> Tuple[float, float, np.ndarray, float, nx.Graph]:
    # retrieving the constants stored on the networkx graph
    n_edges, n_nodes, edges = graph.size(), len(graph), list(graph.edges())
    f, cost_vector, incidence_matrix = extract_graph_info(graph)

    # initializing the algorithm
    p, v, M, L, N, R, active_edges = initialize_algorithm(
        incidence_matrix, cost_vector, n_nodes, edges
    )
    step_sizes, gradient_norms = [], []

    for n_iter in range(max_iter):
        # line search direction
        s = alpha * f - incidence_matrix.T @ M @ v
        gradient_norms.append(np.linalg.norm(s))

        if verbose:
            print(
                f"iteration {n_iter:>{len(str(max_iter))}}, gradient norm: {np.linalg.norm(s):.4f}"
            )

        if np.linalg.norm(s) < eps:
            if verbose or always_print_stop_criteria:
                print("Stopping criteria met.")
            break

        if n_iter % 2 == 0:
            inv_r = np.linalg.inv(R)
            s = inv_r @ inv_r.T @ (s - (N @ (N.T @ s)))

        s = s - np.ones(n_nodes) * np.sum(s) / n_nodes

        # line search step
        t_quadratic = (((alpha * f).T @ s) - (v.T @ M @ (incidence_matrix @ s))) / (
            s.T @ L @ s
        )
        h = -v / (incidence_matrix @ s)
        t_active_set = np.amin(np.where(h > 0, h, np.inf))
        t = min(t_quadratic, t_active_set)
        step_sizes.append(t)

        # stopping if the step size becomes too small
        if t < 1e-16:
            if verbose or always_print_stop_criteria:
                print("Step size too small, exiting.")
            break

        # updating the dual variable
        p += t * s

        previous_active_edges = active_edges.copy()

        v = incidence_matrix @ p - cost_vector
        active_edges = np.array(v > 0.0, dtype=float)
        M = np.diag(active_edges.reshape((n_edges,)))

        diff_active = active_edges - previous_active_edges
        edges_removed = list(np.where(diff_active == -1.0)[0])
        edges_added = list(np.where(diff_active == 1.0)[0])

        updated_edges = set(
            edges[idx]
            for idx in range(len(previous_active_edges))
            if bool(previous_active_edges[idx])
        )

        # new edge in active set
        for edge_added in edges_added:
            L, N, R = edge_added_to_active_set(
                L, N, R, incidence_matrix, edges, edge_added, n_nodes, updated_edges
            )

        # active edges to remove
        for edge_removed in edges_removed:
            L, N, R = edge_removed_from_active_set(
                L, N, R, incidence_matrix, edges, edge_removed, n_nodes, updated_edges
            )

    else:
        if verbose or always_print_stop_criteria:
            print("Maximum iteration number reached")

    # going back to the primal problem
    J = np.maximum(
        (incidence_matrix @ p - cost_vector) / alpha,
        np.zeros(n_edges),
    )

    if verbose:
        print(f"Optimal flow (number of nonzero: {np.count_nonzero(J)} / {n_edges}):")
        print(np.round(J, 2))

    edge_indexes = [e for e in graph.edges()]
    # creating a separate graph that only contains edges where the ot is positive
    sol_graph = nx.create_empty_copy(graph)
    for idx, value in enumerate(J):
        if value and value > 1e-6:
            sol_graph.add_edge(*edge_indexes[idx], ot=value)

    cost = cost_vector @ J
    quadratic_term = float(np.sum(np.square(J)))
    err = np.linalg.norm(incidence_matrix.T @ J - f)

    return cost, quadratic_term, J, err, sol_graph
