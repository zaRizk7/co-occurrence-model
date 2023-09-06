import networkx as nx
import numpy as np

from EinsumNetwork import Graph

__all__ = ["rat_spn_region_graph"]


def rat_spn_region_graph(
    num_vars: int,
    num_repetition: int,
    max_depth: int,
) -> nx.DiGraph:
    region_graph = nx.DiGraph()

    ids = tuple(range(num_vars))
    root_node = Graph.DistributionVector(ids)

    for replica_idx in range(num_repetition):
        split(
            num_vars,
            max_depth,
            replica_idx,
            region_graph,
            root_node,
        )

    return region_graph


def split(
    num_vars: int,
    max_depth: int,
    replica_idx: int,
    region_graph: nx.DiGraph,
    root_node: Graph.DistributionVector,
):
    ids = root_node.scope

    if len(ids) <= 1 or max_depth <= 0:
        return

    ids = np.random.permutation(ids)

    splits = np.array_split(ids, 2)

    product_node = Graph.Product(ids)
    region_graph.add_edge(root_node, product_node)
    for s in splits:
        split_node = Graph.DistributionVector(s)
        split_node.einet_address.replica_idx = replica_idx

        region_graph.add_edge(product_node, split_node)

        split(
            num_vars,
            max_depth - 1,
            replica_idx,
            region_graph,
            split_node,
        )
