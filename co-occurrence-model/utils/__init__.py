from typing import Dict

import einops
import torch
import torch.nn.functional as F

from utils import bin_g, dataset, pgmpy

__all__ = [
    "bin_g",
    "dataset",
    "pgmpy",
    "mutual_information",
    "record_gradient_norm",
]


def mutual_information(data: torch.Tensor, eps: float = 1e-1) -> torch.Tensor:
    """Calculate mutual information with infinitesimal smoothing."""
    mi = torch.zeros([data.size(1)] * 2).to(data.device)
    max_value = data.max() + 1
    for i in range(data.size(1)):
        for j in range(data.size(1)):
            if j >= i:
                mi[i, j] = -torch.inf
                continue

            x = F.one_hot(data[:, [i, j]], max_value)
            x = x.float().to(data.device)
            xi, xj = einops.rearrange(x, "n d k -> d n k")

            cxixj = einops.einsum(xi, xj, "n k1, n k2 -> k1 k2")
            cxi = einops.einsum(xi, "n k -> k")
            cxj = einops.einsum(xj, "n k -> k")

            cxixj = cxixj + eps
            cxi = cxi + eps
            cxj = cxj + eps

            pxixj = cxixj / cxixj.sum().expand_as(cxixj)
            pxi = cxi / cxi.sum().expand_as(cxi)
            pxj = cxj / cxj.sum().expand_as(cxj)
            pxipxj = einops.einsum(pxi, pxj, "k1, k2 -> k1 k2")

            pmi = pxixj.log() - pxipxj.log()
            mi[i, j] = (pxixj * pmi).sum()

    return mi


def record_gradient_norm(model: torch.nn.Module, p: int = 2) -> Dict[str, float]:
    grad_norm = dict()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        with torch.no_grad():
            grad_norm[name] = param.grad.norm(p).item()

    return grad_norm
