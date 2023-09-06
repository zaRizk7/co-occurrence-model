from typing import List, Set

import networkx as nx

from EinsumNetwork import Graph

__all__ = ["bin_g_region_graph"]


def merge_scope(nodes: List[List[Graph.DistributionVector]]) -> Set[int]:
    # Expect a 2xN list

    return {scope for node in nodes for subnode in node for scope in subnode.scope}


def create_sum_node(scope: Set[int], replica_idx: int):
    node = Graph.DistributionVector(scope)
    node.einet_address.replica_idx = replica_idx
    return node


def generate_sum_nodes(scope: Set[int], size: int) -> List[Graph.DistributionVector]:
    return [create_sum_node(scope, idx) for idx in range(size)]


def generate_product_nodes(scope: Set[int], size: int) -> List[Graph.Product]:
    return [Graph.Product(scope) for _ in range(size)]


def gather_nodes(
    children: List[str],
    size: int,
    graph: nx.DiGraph,
    region_graph: nx.DiGraph,
    observed_nodes: List[str],
    with_repetition: bool,
) -> List[Graph.DistributionVector]:
    return [
        dfs(child, size, graph, region_graph, observed_nodes, with_repetition)
        for child in children
    ]


# Supposed to return a DistributionVector
def dfs(
    root: str,
    latent_size: int,
    graph: nx.DiGraph,
    region_graph: nx.DiGraph,
    observed_nodes: list,
    with_repetition: bool = True,
) -> List[Graph.DistributionVector]:
    children = list(graph.successors(root))

    if not with_repetition:
        latent_size = 1

    if len(children) <= 0:
        scope = {observed_nodes.index(root)}
        dist_nodes = generate_sum_nodes(scope, latent_size)
        region_graph.add_nodes_from(dist_nodes)
        return dist_nodes

    label, size = [int(num) for num in root.split(" ")]

    if not with_repetition:
        size = 1

    child_nodes = gather_nodes(
        children, size, graph, region_graph, observed_nodes, with_repetition
    )

    scope = merge_scope(child_nodes)
    sum_node = Graph.DistributionVector(scope)
    product_nodes = generate_product_nodes(scope, size)
    left_nodes, right_nodes = child_nodes

    for product_node, left_node, right_node in zip(
        product_nodes, left_nodes, right_nodes
    ):
        region_graph.add_edge(product_node, left_node)
        region_graph.add_edge(product_node, right_node)

    for product_node in product_nodes:
        region_graph.add_edge(sum_node, product_node)

    return [sum_node] * size


def aggregate(region_graph: nx.DiGraph) -> None:
    roots = Graph.get_roots(region_graph)
    for i in range(1, len(roots), 2):
        scope = [*roots[i - 1].scope, *roots[i].scope]

        sum_node = Graph.DistributionVector(scope)

        sum_node.einet_address.replica_idx = 0

        product_node = Graph.Product(scope)

        region_graph.add_edge(sum_node, product_node)
        region_graph.add_edge(product_node, roots[i - 1])
        region_graph.add_edge(product_node, roots[i])

    if len(roots) > 1:
        aggregate(region_graph)


def bin_g_region_graph(
    graph: nx.DiGraph, with_repetition: bool = True, with_aggregate: bool = True
) -> nx.DiGraph:
    roots = [node for node, in_edge in graph.in_degree if in_edge <= 0]
    observed_nodes = list(graph.nodes)[:80]

    region_graph = nx.DiGraph()
    for root in roots:
        node = dfs(root, 1, graph, region_graph, observed_nodes, with_repetition)[0]

    if with_aggregate:
        aggregate(region_graph)

    return region_graph
