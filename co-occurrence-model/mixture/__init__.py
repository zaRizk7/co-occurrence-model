from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader

import einops
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily


from utils.dataset import ObjectCooccurrenceCOCODataset

__all__ = ["CategoricalMixture", "create_mixture", "initialize_parameters"]


class CategoricalMixture(torch.nn.Module):
    def __init__(self, params: torch.nn.ParameterDict, step_size: float = 1e-1) -> None:
        if 0 < step_size > 1:
            raise ValueError("step_size must be between zero and one!")

        super().__init__()
        self.params = params
        self.step_size = step_size

    @property
    def max_value(self) -> int:
        return self.params["probs"].size(-1)

    @property
    def num_vars(self) -> int:
        return self.params["probs"].size(1)

    @property
    def dist(self) -> torch.distributions.MixtureSameFamily:
        return create_mixture(self.params)

    def e_step(self, inputs: torch.Tensor) -> torch.Tensor:
        log_prior = self.dist.mixture_distribution.logits
        log_likelihood = self.dist.component_distribution.log_prob(inputs)

        return (log_prior + log_likelihood).softmax(-1)

    def m_step(self, inputs: torch.Tensor, posterior: torch.Tensor) -> None:
        posterior_expand = einops.repeat(posterior, "n k -> n k 1 1")
        count = einops.repeat(posterior.sum(0), "k -> k 1 1")

        prior_ = posterior.mean(0)
        probs_ = (posterior_expand * inputs).sum(0) / count

        prior_.mul_(self.step_size)
        probs_.mul_(self.step_size)

        self.params["prior"].mul_(1 - self.step_size).add_(prior_)
        self.params["probs"].mul_(1 - self.step_size).add_(probs_)

    def em_step(self, inputs: torch.Tensor) -> None:
        inputs_expand = einops.repeat(inputs, "n d -> n 1 d")
        inputs_onehot = F.one_hot(inputs_expand, self.max_value)

        posterior = self.e_step(inputs_expand)
        self.m_step(inputs_onehot, posterior)

    def forward(
        self, inputs: torch.Tensor, index: Optional[List] = None
    ) -> torch.Tensor:
        probs = self.params["probs"]
        if index is not None:
            index = torch.tensor(index).to(inputs.device)
            probs = probs.index_select(index=index, dim=1)
            inputs = inputs.index_select(index=index, dim=1)

        params = dict(prior=self.params["prior"], probs=probs)
        dist = create_mixture(params)

        return dist.log_prob(inputs)


def create_mixture(
    params: Union[Dict[str, torch.Tensor], torch.nn.ParameterDict]
) -> MixtureSameFamily:
    return MixtureSameFamily(
        Categorical(params["prior"]), Independent(
            Categorical(params["probs"]), 1)
    )


def initialize_parameters(
    dataset: ObjectCooccurrenceCOCODataset, num_mixtures: int
) -> torch.nn.ParameterDict:
    max_value = dataset.features.to_numpy().max()
    sampler = torch.distributions.Categorical(
        logits=torch.zeros(dataset.features.shape[0])
    )

    prior = torch.zeros(num_mixtures).softmax(-1)
    probs = dataset.features.iloc[sampler.sample([num_mixtures])].to_numpy()
    probs = torch.from_numpy(probs)
    probs = F.one_hot(probs, max_value + 1).float()

    params = torch.nn.ParameterDict()
    params["prior"] = prior
    params["probs"] = probs

    for param in params.parameters():
        param.requires_grad = False

    return params


def log_likelihood(
    dataloader: DataLoader, model: CategoricalMixture, device: Union[str, torch.device]
) -> float:
    score = 0
    model = model.to(device)

    model.eval()
    for inputs in dataloader:
        inputs = inputs.to(device)
        score += model(inputs).sum().item()
    return score


def log_posterior(
    dataloader: DataLoader, model: CategoricalMixture, device: Union[str, torch.device]
) -> float:
    score = 0
    model = model.to(device)

    model.eval()
    for inputs in dataloader:
        inputs = inputs.to(device)
        idx_query = torch.arange(inputs.size(-1))

        log_likelihood = model(inputs, idx_query)
        for i in idx_query:
            idx_evidence = idx_query[idx_query != i]
            log_marginal = model(inputs, idx_evidence)
            score += (log_likelihood - log_marginal).sum().item()

    return score
