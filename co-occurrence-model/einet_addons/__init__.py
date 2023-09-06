from typing import Dict, List, Union

import torch

from EinsumNetwork.EinsumNetwork import EinsumNetwork
from utils import record_gradient_norm
from einet_addons.forest import *
from einet_addons.structure import *

__all__ = ["log_likelihood", "log_posterior", "EiNetForest"]


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: Union[EiNetForest, EinsumNetwork],
    device: Union[str, torch.device],
) -> List[Dict[str, float]]:
    grad_norm = []
    model = model.to(device)

    model.train()
    for inputs in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        outputs.sum().backward()
        grad_norm.append(record_gradient_norm(model))
        model.em_process_batch()
    model.em_update()
    return grad_norm


def log_likelihood(
    dataloader: torch.utils.data.DataLoader,
    model: Union[EiNetForest, EinsumNetwork],
    device: Union[str, torch.device],
) -> float:
    score = 0
    model = model.to(device)

    model.train()
    with torch.inference_mode():
        for inputs in dataloader:
            inputs = inputs.to(device)
            log_joint = model(inputs)
            score += log_joint.sum().item()
    return score


def log_posterior(
    dataloader: torch.utils.data.DataLoader,
    model: Union[EiNetForest, EinsumNetwork],
    device: Union[str, torch.device],
) -> float:
    score = 0
    model = model.to(device)

    model.eval()
    with torch.inference_mode():
        for inputs in dataloader:
            inputs = inputs.to(device)
            idx = torch.arange(inputs.size(-1)).to(device)

            model.set_marginalization_idx()
            log_joint = model(inputs)

            for i in idx:
                idx_marginal = idx[idx == i]
                model.set_marginalization_idx(idx_marginal)
                log_marginal = model(inputs)

                score += (log_joint - log_marginal).sum().item()
    return score
