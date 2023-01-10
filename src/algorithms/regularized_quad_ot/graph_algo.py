from typing import Tuple, List, Iterable

import numpy as np


def weakly_connected_components(
    n_nodes: int, edges: Iterable[Tuple[int, int]], verbose: bool = False
) -> List[List[int]]:
    """
    Computes the weakly connected component of a graph based on a list of its edges.
    This function was reimplemented because the equivalent nx.weakly_connected_components will by default consider
    all the edges of the graph, and we want to restrict our version to the active set.
    """
    if verbose:
        print("\nComputing the weakly connected components.\nEdges:")
        print(edges)

    adjacency_matrix = [[] for _ in range(n_nodes)]
    for edge in edges:
        adjacency_matrix[edge[0]].append(edge[1])
        adjacency_matrix[edge[1]].append(edge[0])

    if verbose:
        print("Adjacency matrix:")
        print(adjacency_matrix)

    visited = [False for _ in range(n_nodes)]

    def dfs(parent_node: int, children_nodes: List[int]) -> List[int]:
        """
        Deep-first search.
        """
        visited[parent_node] = True
        children_nodes.append(parent_node)
        for children_node in adjacency_matrix[parent_node]:
            if not visited[children_node]:
                children_nodes = dfs(children_node, children_nodes)

        return children_nodes

    connected_components = []
    for node in range(n_nodes):
        if verbose:
            print(f"Visiting node {node}")
        if not visited[node]:
            node_children = []
            node_component = dfs(node, node_children)
            connected_components.append(node_component)
            if verbose:
                print(f"Node not visited yet.")
                print("Node component found:")
                print(node_component)

    return connected_components


def normalized_flood_fill(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Computes the normalized flood fill. Useful to initialize N by providing an orthogonal basis for the null space of
    the Lagrangian Lk.
    """
    N = np.array([], dtype=float).reshape(n_nodes, 0)
    for connected_component in weakly_connected_components(n_nodes, edges):
        col = (
            np.array([i in connected_component for i in range(n_nodes)])
            / np.sqrt(len(connected_component))
        ).reshape(n_nodes, 1)
        N = np.hstack((N, col))

    return N
