from typing import Tuple, List

import networkx as nx
import numpy as np
import torch

from src.algorithms.utils import compute_cost_matrix, collect_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running Sinkhorn on {device}.")


def after_sinkhorn(
    transportation_plan: np.ndarray,
    graph: nx.Graph,
    shortest_paths: List[List[List[int]]],
    cost_matrix: np.ndarray,
    f: np.ndarray,
    verbose: bool,
) -> Tuple[float, float, np.ndarray, float, float, nx.Graph]:
    """
    Formats the output of either version of Sinkhorn algorithm.
    """
    if verbose:
        nonzero = np.count_nonzero(transportation_plan)
        print(
            f"Optimal transportation plan (number of nonzero: {nonzero} / {transportation_plan.size}):"
        )
        print(np.round(transportation_plan, 2))

    # creating a copy of the graph that will only have the edges where there is a movement of mass
    uncollected_graph = nx.create_empty_copy(graph)
    # adding an attribute that will yield the value of the transportation plan
    uncollected_graph.add_edges_from(
        [
            (origin, destination, {"ot": value})
            for origin, row in enumerate(transportation_plan)
            for destination, value in enumerate(row)
            if value > 1e-3
        ]
    )

    # adding the values of the ot on the initial graph by putting back the plan on the actual edges
    collected_graph = collect_graph(graph, transportation_plan, shortest_paths)

    flow = np.array(list(nx.get_edge_attributes(collected_graph, "ot").values()))
    cost = float(np.sum(cost_matrix * transportation_plan))
    quadratic_term = float(np.sum(np.square(flow)))
    err = np.linalg.norm(nx.incidence_matrix(graph, oriented=True).toarray() @ flow - f)

    nonzero = np.count_nonzero(flow)
    if verbose:
        print(f"Optimal flow (number of nonzero: {nonzero} / {graph.size()}):")
        print(np.round(flow, 2))

    return cost, quadratic_term, flow, err, 1 - nonzero / graph.size(), collected_graph


def sinkhorn(
    graph: nx.Graph, alpha: float, verbose: bool = True
) -> Tuple[float, float, np.ndarray, float, float, nx.Graph]:
    """
    Computes the entropically-regularized Wasserstein-1 distance on a weighted graph using Sinkhorn algorithm.
    """
    epsilon = alpha

    n_nodes = len(graph)
    rho_0 = np.array(list(nx.get_node_attributes(graph, "rho_0").values()))
    rho_1 = np.array(list(nx.get_node_attributes(graph, "rho_1").values()))
    cost_matrix, shortest_paths = compute_cost_matrix(graph)

    K = np.exp(-(cost_matrix**2) / epsilon)
    u = torch.ones(n_nodes)
    v = torch.ones(n_nodes)

    K1 = torch.from_numpy(K).type(torch.FloatTensor)
    a1 = torch.from_numpy(rho_0).type(torch.FloatTensor)
    b1 = torch.from_numpy(rho_1).type(torch.FloatTensor)

    K1 = K1.to(device)
    u = u.to(device)
    v = v.to(device)
    a1 = a1.to(device)
    b1 = b1.to(device)

    n_iter = 4000
    for i in range(n_iter):
        u = a1 / (K1 * v[None, :]).sum(1)
        v = b1 / (K1 * u[:, None]).sum(0)

    transportation_plan = np.diag(u.cpu()) @ K @ np.diag(v.cpu())

    return after_sinkhorn(
        transportation_plan, graph, shortest_paths, cost_matrix, rho_1 - rho_0, verbose
    )


def stable_sinkhorn(
    graph: nx.Graph, alpha: float, verbose: bool = True
) -> Tuple[float, float, np.ndarray, float, float, nx.Graph]:
    """
    Computes the entropically-regularized Wasserstein-1 distance on a weighted graph using a version of Sinkhorn
    algorithm in log domain stabilized using a log-sum-exp trick.
    """
    epsilon = alpha

    n_nodes = len(graph)
    rho_0 = np.array(list(nx.get_node_attributes(graph, "rho_0").values()))
    rho_1 = np.array(list(nx.get_node_attributes(graph, "rho_1").values()))
    cost_matrix, shortest_paths = compute_cost_matrix(graph)

    C = torch.autograd.Variable(
        torch.from_numpy(cost_matrix).type(torch.FloatTensor).to(device)
    )

    def modified_cost(u_val, v_val):
        return (-C + u_val.unsqueeze(1) + v_val.unsqueeze(0)) / epsilon

    def stable_lse(A):
        # adding 10^-6 to prevent NaN
        return torch.log(
            torch.exp(A - torch.max(A)).sum(1, keepdim=True) + 1e-6
        ) + torch.max(A)

    u = torch.ones(n_nodes)
    v = torch.ones(n_nodes)

    a1 = torch.from_numpy(rho_0).type(torch.FloatTensor)
    b1 = torch.from_numpy(rho_1).type(torch.FloatTensor)

    u = u.to(device)
    v = v.to(device)
    a1 = a1.to(device)
    b1 = b1.to(device)

    n_iter = 10000
    for i in range(n_iter):
        u = epsilon * (torch.log(a1) - stable_lse(modified_cost(u, v)).squeeze()) + u
        v = (
            epsilon * (torch.log(b1) - stable_lse(modified_cost(u, v).t()).squeeze())
            + v
        )

    transportation_plan = torch.exp(modified_cost(u, v)).cpu()

    return after_sinkhorn(
        transportation_plan.numpy(),
        graph,
        shortest_paths,
        cost_matrix,
        rho_1 - rho_0,
        verbose,
    )
