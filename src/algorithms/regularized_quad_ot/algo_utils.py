from typing import Tuple, List, Set

import numpy as np

from src.algorithms.regularized_quad_ot.cholesky import cholupdate, choldowndate
from src.algorithms.regularized_quad_ot.graph_algo import (
    weakly_connected_components,
    normalized_flood_fill,
)


def edge_added_to_active_set(
    L,
    N,
    R,
    D,
    edges: List[Tuple[int, int]],
    edge_added: int,
    n_nodes: int,
    updated_edges: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies if an edge added to the active set merges two components or not and updates every variable accordingly.
    """
    updated_edges.add(edges[edge_added])

    L += np.outer(D[edge_added], D[edge_added].T)
    cholupdate(R, D[edge_added])

    start, end = edges[edge_added]
    c1_idx = int(np.where(np.abs(N[start, :]) > 1e-12)[0][0])
    c2_idx = int(np.where(np.abs(N[end, :]) > 1e-12)[0][0])

    # case where the two components are merged by edge_added
    if c1_idx != c2_idx:
        c1 = N[:, c1_idx].copy()
        c2 = N[:, c2_idx].copy()

        # defining the new merged component
        c1_component = np.where(np.abs(N[:, c1_idx]) > 1e-12)[0]
        c2_component = np.where(np.abs(N[:, c2_idx]) > 1e-12)[0]
        c_bar = np.hstack((c1_component, c2_component))
        new_component = np.array([i in c_bar for i in range(n_nodes)]) / np.sqrt(
            len(c_bar)
        )

        # remove two columns of the components
        N = np.delete(N, [c1_idx, c2_idx], 1)

        # adding the new merged column
        N = np.hstack((N, new_component.reshape(n_nodes, 1)))

        # performing the Cholesky updates / downdates
        cholupdate(R, new_component)
        choldowndate(R, c1)
        choldowndate(R, c2)

    return L, N, R


def edge_removed_from_active_set(
    L,
    N,
    R,
    D,
    edges: List[Tuple[int, int]],
    edge_removed: int,
    n_nodes: int,
    updated_edges: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies if an edg removed from the active set splits a component in two or not and updates every variable
    accordingly.
    """
    updated_edges.remove(edges[edge_removed])

    L -= np.outer(D[edge_removed], D[edge_removed].T)

    # check if the two components are splitting
    start, end = edges[edge_removed]
    joint_component_idx = int(np.where(np.abs(N[start, :]) > 1e-12)[0][0])
    c_bar = N[:, joint_component_idx].copy()

    is_splitting = True
    for component in weakly_connected_components(n_nodes, updated_edges):
        if start in component and end in component:
            is_splitting = False
            break
        elif start in component:
            c1 = np.array([i in component for i in range(n_nodes)]) / np.sqrt(
                len(component)
            )

        elif end in component:
            c2 = np.array([i in component for i in range(n_nodes)]) / np.sqrt(
                len(component)
            )

    if is_splitting:
        # deleting the split component
        N = np.delete(N, joint_component_idx, 1)

        # adding the two new columns
        # noinspection PyUnboundLocalVariable
        N = np.hstack((N, c1.reshape(n_nodes, 1), c2.reshape(n_nodes, 1)))

        # performing the Cholesky updates / downdates
        cholupdate(R, c1)
        cholupdate(R, c2)
        choldowndate(R, c_bar)

    # downdate on R
    choldowndate(R, D[edge_removed])

    return L, N, R


def initialize_algorithm(
    D, c, n_nodes: int, edges: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes every variable in the algorithm.
    """
    # randomly initializing the dual variable
    p = 10 * np.random.random(size=(n_nodes,))

    # computing v, M, L, N and R
    v = D @ p - c
    active_edges = np.array(v > 0.0, dtype=float)
    M = np.diag(active_edges)
    L = D.T @ M @ D
    N = normalized_flood_fill(
        n_nodes, [edges[idx] for idx in range(len(edges)) if bool(active_edges[idx])]
    )
    _, R = np.linalg.qr(np.vstack((M @ D, N.T)))

    return p, v, M, L, N, R, active_edges
