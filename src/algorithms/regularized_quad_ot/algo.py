from typing import Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.algorithms.graph_utils import extract_graph_info
from src.algorithms.regularized_quad_ot.algo_utils import (
    initialize_algorithm,
    edge_added_to_active_set,
    edge_removed_from_active_set,
)


def regularized_quadratic_ot(
    graph: nx.Graph,
    alpha: float,
    eps: float = 1e-8,
    max_iter: int = 5000,
    verbose: bool = True,
    always_print_stop_criteria: bool = True,
    plot_evolution: bool = True,
    optimal_sol: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, float, nx.Graph]:
    # retrieving the constants stored on the networkx graph
    n_edges, n_nodes, edges = graph.size(), len(graph), list(graph.edges())
    f, cost_vector, incidence_matrix = extract_graph_info(graph)

    # initializing the algorithm
    p, v, M, L, N, R, active_edges = initialize_algorithm(
        incidence_matrix, cost_vector, n_nodes, edges
    )
    step_sizes, gradient_norms, n_non_zeros = [], [], []
    constraint_violation, dist_with_opt, costs = [], [], []

    for n_iter in range(max_iter):
        # line search direction
        s = alpha * f - incidence_matrix.T @ M @ v
        gradient_norm = np.linalg.norm(s)
        gradient_norms.append(gradient_norm)

        if verbose:
            print(
                f"iteration {n_iter:>{len(str(max_iter))}}, gradient norm: {gradient_norm:.4f}"
            )

        if gradient_norm < eps:
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
        step_sizes.append(float(t))

        # stopping if the step size becomes too small
        if t < 1e-16:
            if verbose or always_print_stop_criteria:
                print("Step size too small, exiting.")
            break

        # step update
        p += t * s

        if plot_evolution:
            J = np.maximum(
                (incidence_matrix @ p - cost_vector) / alpha,
                np.zeros(n_edges),
            )
            n_non_zeros.append(np.count_nonzero(J) / n_edges)
            costs.append(float(cost_vector @ J))
            constraint_violation.append(np.linalg.norm(incidence_matrix.T @ J - f))
            if optimal_sol is not None:
                dist_with_opt.append(np.linalg.norm(J - optimal_sol))

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

    if plot_evolution:
        fig, axs = plt.subplots(2, 3, figsize=(20, 15))

        axs[0][0].plot(step_sizes, label=alpha)
        axs[0][0].set(title="Step size evolution", xlabel="$k$", ylabel="$t_k$")

        axs[0][1].plot(gradient_norms, label=alpha)
        axs[0][1].set(title="Gradient norm evolution", xlabel="$k$", ylabel="$||s_k||$")

        axs[0][2].plot(n_non_zeros, label=alpha)
        axs[0][2].set(title="Sparsity", xlabel="$k$", ylabel=r"$\frac{||J_k||_0}{|V|}$")

        axs[1][0].plot(costs, label=alpha)
        axs[1][0].set(title="Primal cost", xlabel="$k$", ylabel="$||c^T J_k||$")

        axs[1][1].plot(constraint_violation, label=alpha)
        axs[1][1].set(
            title="Constraint violation", xlabel="$k$", ylabel="$||D^T J_k - f||$"
        )

        axs[1][2].plot(dist_with_opt, label=alpha)
        axs[1][2].set(
            title="Distance to an optimal solution",
            xlabel="$k$",
            ylabel="$||J_k - J^*||$",
        )

        for ax_list in axs:
            for ax in ax_list:
                ax.legend()
        plt.show()

    return cost, quadratic_term, J, err, sol_graph
