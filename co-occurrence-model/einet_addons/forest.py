from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import torch

from math import log

from EinsumNetwork import EinsumNetwork, Graph

__all__ = ["EiNetForest", "EiNetTree"]


def gather_nodes(
    graph: nx.DiGraph, root: Graph.DistributionVector
) -> List[Union[Graph.DistributionVector, Graph.Product]]:
    nodes = []
    nodes.append(root)
    for child in graph.successors(root):
        nodes.extend(gather_nodes(graph, child))
    return nodes


def remap_scope(
    graph: nx.DiGraph,
    mapping: Dict[int, int],
    root: Optional[Graph.DistributionVector] = None,
    visited: set = set(),
) -> None:
    if root is None:
        root = Graph.get_roots(graph)[0]

    if root in visited:
        return

    visited.add(root)
    root.scope = tuple(mapping[i] for i in root.scope)
    for child in graph.successors(root):
        remap_scope(graph, mapping, child)


def get_subgraph(
    graph: nx.DiGraph, root: Optional[Graph.DistributionVector]
) -> Tuple[nx.DiGraph, Dict[int, int]]:
    nodes = gather_nodes(graph, root)

    mapping = {v: k for k, v in enumerate(root.scope)}

    subgraph = deepcopy(graph.subgraph(nodes))
    remap_scope(subgraph, mapping)

    return subgraph, mapping


class EiNetTree(torch.nn.Module):
    __constants__ = ["idx_original"]

    def __init__(
        self,
        graph: nx.DiGraph,
        root: Graph.DistributionVector,
        args: EinsumNetwork.Args,
    ) -> None:
        super().__init__()

        tree, mapping = get_subgraph(graph, root)

        tree_args = deepcopy(args)
        tree_args.num_var = len(mapping)

        idx_original = torch.tensor(list(mapping.keys()))
        self.register_buffer("idx_original", idx_original)

        self.einet = EinsumNetwork.EinsumNetwork(tree, tree_args)

    def forward(self, input: torch.Tensor):
        return self.einet(input.index_select(index=self.idx_original, dim=1))

    def set_marginalization_idx(
        self, idx: Optional[Union[List[int], torch.Tensor]] = None
    ) -> None:
        if idx is None:
            self.einet.set_marginalization_idx(idx)
            return

        if isinstance(idx, list):
            idx = torch.tensor(idx)

        idx_remap = torch.isin(self.idx_original, idx, assume_unique=True)
        idx_remap = torch.where(idx_remap)
        idx_remap = idx_remap[0]

        if len(idx_remap) <= 0:
            idx_remap = None

        self.einet.set_marginalization_idx(idx_remap)

    def initialize(self) -> None:
        self.einet.initialize()

    def em_process_batch(self) -> None:
        self.einet.em_process_batch()

    def em_update(self) -> None:
        self.einet.em_update()


class EiNetForest(torch.nn.Sequential):
    def __init__(self, graph: nx.DiGraph, args: EinsumNetwork.Args) -> None:
        super().__init__()

        for i, root in enumerate(Graph.get_roots(graph)):
            self.add_module(str(i), EiNetTree(graph, root, args))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = [tree(input) for tree in self]
        output = [(out - log(out.size(dim=-1))) for out in output]
        output = [out.logsumexp(dim=-1) for out in output]
        output = torch.stack(output, dim=-1)
        return output.sum(dim=-1, keepdims=True)

    def set_marginalization_idx(
        self, idx: Optional[Union[List[int], torch.Tensor]] = None
    ) -> None:
        for module in self:
            module.set_marginalization_idx(idx)

    def initialize(self) -> None:
        for module in self:
            module.initialize()

    def em_process_batch(self) -> None:
        for module in self:
            module.em_process_batch()

    def em_update(self) -> None:
        for module in self:
            module.em_update()
