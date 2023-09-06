from typing import Tuple

import mat73
import networkx as nx
import numpy as np
import pandas as pd
from einops import repeat
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from scipy.special import logsumexp

__all__ = ["log_likelihood", "load"]


def gather_log_prob(
    root: str, data: pd.DataFrame, model: BayesianNetwork
) -> np.ndarray:
    cpd = model.get_cpds(root)
    prob = np.take(np.transpose(cpd.values), data[root], axis=-1)
    return np.log(prob)


def tree_log_likelihood(
    root: str, data: pd.DataFrame, model: BayesianNetwork
) -> np.ndarray:
    children = list(model.successors(root))

    if len(children) <= 0:
        return gather_log_prob(root, data, model)

    cpd = model.get_cpds(root)
    log_probs = np.log(np.transpose(cpd.values))

    if len(log_probs.shape) <= 1:
        log_probs = repeat(log_probs, "d -> d n", n=len(data))
    else:
        log_probs = repeat(log_probs, "d c -> d c n", n=len(data))

    res = 0
    for child in children:
        log_probs += tree_log_likelihood(child, data, model)

    if np.any(log_probs > 0):
        raise ValueError(f"All log probs must be a negative value! {log_probs}")

    # expect return shape (d n) or (n)
    if len(log_probs.shape) <= 2:
        return logsumexp(log_probs, 0)

    return logsumexp(log_probs, 1)


def log_likelihood(data: pd.DataFrame, model: BayesianNetwork) -> np.ndarray:
    roots = [root for root, weight in model.in_degree if weight <= 0]

    return np.sum([tree_log_likelihood(root, data, model) for root in roots])


def load(
    directory: str,
    df: pd.DataFrame,
    sep: str = "\\n",
    add_bracket: bool = True,
    eps: float = 1e-1,
) -> Tuple[nx.DiGraph, BayesianNetwork]:
    mat = mat73.loadmat(directory)["t_hat"]

    BIN_G = nx.DiGraph()

    BIN_G.add_nodes_from([i for i in range(1, len(mat["t"]) + 1)])

    map1 = {k: v.replace(" ", sep) for k, v in enumerate(df.columns, 1)}
    map2 = {
        k: f"{k} [{v}]" if add_bracket else f"{k} {v}"
        for k, v in enumerate(mat["nsyms"][len(df.columns) :], len(df.columns) + 1)
    }
    maps = {**map1, **map2}

    for i, src in enumerate(mat["t"], 1):
        for tgts in src:
            if isinstance(tgts, np.ndarray):
                tgts = tgts.astype(np.int64)
                for tgt in tgts:
                    BIN_G.add_edge(i, tgt)

    BIN_G = nx.relabel_nodes(BIN_G, maps)

    nodes = list(BIN_G.nodes)
    BN = BayesianNetwork(BIN_G, latents=nodes[df.shape[1] :])

    for i, children in enumerate(mat["t"]):
        parent = nodes[i]
        if children[0] is not None:
            j, k = children[0].astype(int) - 1

            table_j, table_k = mat["p"][i][0]

            table_j *= len(df)
            table_k *= len(df)

            table_j += eps
            table_k += eps

            table_j /= table_j.sum(0)
            table_k /= table_k.sum(0)

            BN.add_cpds(
                TabularCPD(
                    nodes[j], table_j.shape[0], table_j, [parent], [table_j.shape[-1]]
                ),
                TabularCPD(
                    nodes[k], table_k.shape[0], table_k, [parent], [table_k.shape[-1]]
                ),
            )

    for i, table in zip(mat["t0"].astype(int), mat["p0"]):
        i -= 1

        table *= len(df)
        table += eps
        table /= table.sum(0)

        BN.add_cpds(TabularCPD(nodes[i], table.shape[0], table[:, None]))

    return BIN_G, BN
