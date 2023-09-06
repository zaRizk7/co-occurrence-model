import torch
import einops

from typing import Optional, Union, List, Dict
from utils import record_gradient_norm
from math import log
import torch.nn.functional as F

__all__ = ['MADE', 'MaskedAutoregressiveLinear',
           'train_one_epoch', 'log_likelihood', 'log_posterior']


class MADE(torch.nn.Sequential):
    __constants__ = ["in_features"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

        for module in reversed(self):
            if isinstance(module, MaskedAutoregressiveLinear):
                module.set_mode(output=True)
                self.num_features = module.out_features
                break

    def forward(
        self, input: torch.Tensor, order: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if order is None:
            order = torch.randperm(self.num_features)

        order = order.to(input.device)
        units_original = order + 1

        output = input
        units = units_original
        for module in self:
            if isinstance(module, MaskedAutoregressiveLinear):
                if module.is_output:
                    output, units = module(
                        output, units_original, self.num_features)
                    continue
                output, units = module(output, units, self.num_features)
            else:
                output = module(output)

        return output


class MaskedAutoregressiveLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features", "in_dims", "out_dims"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_dims: int = 1,
        out_dims: int = 1,
        bias: bool = True,
        weight_condition: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.in_dims = in_dims

        self.out_features = out_features
        self.out_dims = out_dims

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, out_dims, in_features, in_dims)
        )

        self.register_parameter("bias", None)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, out_dims))

        self.register_parameter("weight_condition", None)
        if weight_condition:
            self.weight_condition = torch.nn.Parameter(self.weight.clone())

        self.reset_parameters()
        self.is_output = False
        self.mask_pattern = "i j k l, i k -> i j k l"
        self.transform_pattern = "i j k l, ... k l -> ... i j"
        self.conditioning_pattern = "i j k l -> i j"

    def set_mode(self, output: bool = False):
        self.is_output = output
        return self

    def extra_repr(self) -> str:
        return "in_features={}, in_dims={}, out_features={}, out_dims={},\nbias={}, weight_condition={}".format(
            self.in_features,
            self.in_dims,
            self.out_features,
            self.out_dims,
            self.bias is not None,
            self.weight_condition is not None,
        )

    def forward(
        self, input: torch.Tensor, units: torch.Tensor, num_features: int
    ) -> torch.Tensor:
        units_successor = self.sample_units(units, num_features)
        units_successor = units_successor.to(input.device)
        weight_mask = self.create_mask(units, units_successor)

        weight = einops.einsum(self.weight, weight_mask, self.mask_pattern)
        output = einops.einsum(weight, input, self.transform_pattern)

        if self.bias is not None:
            output += self.bias

        if self.weight_condition is not None:
            weight_condition = einops.einsum(
                self.weight_condition, weight_mask, self.mask_pattern
            )
            output += einops.einsum(weight_condition,
                                    input, self.conditioning_pattern)

        return output, units_successor

    def sample_units(self, units: torch.Tensor, num_features: int) -> torch.Tensor:
        size = self.out_features
        if self.is_output:
            size = self.in_features
        return torch.randint(units.min(), num_features, size=[size])

    def create_mask(self, units: torch.Tensor, units_successor: torch.Tensor):
        if self.is_output:
            units = units.unsqueeze(dim=-1)
            return units > units_successor
        units_successor = units_successor.unsqueeze(dim=-1)
        return units_successor >= units

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

        if self.weight_condition is not None:
            torch.nn.init.zeros_(self.weight_condition)


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: MADE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_sample: int,
    one_hot_input: bool,
    device: Union[str, torch.device],
    max_value: int,
) -> List[Dict[str, float]]:
    grad_norm = []
    model = model.to(device)

    model.train()
    for inputs in dataloader:
        targets = inputs.to(device)

        if one_hot_input:
            inputs = F.one_hot(inputs, max_value+1)
        else:
            inputs = inputs.unsqueeze(-1)
        inputs = inputs.float().to(device)

        loss = 0
        for _ in range(num_sample):
            dist = torch.distributions.Categorical(logits=model(inputs))
            dist = torch.distributions.Independent(dist, 1)
            loss -= dist.log_prob(targets).sum()
        loss /= (inputs.size(0) * num_sample)

        loss.backward()
        grad_norm.append(record_gradient_norm(model))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return grad_norm


def log_likelihood(dataloader: torch.utils.data.DataLoader,
                   model: MADE, num_sample: int,
                   one_hot_input: bool,  device: Union[str, torch.device],
                   max_value: int) -> float:
    score = 0
    model = model.to(device)

    model.eval()
    with torch.inference_mode():
        for inputs in dataloader:
            targets = inputs.to(device)

            if one_hot_input:
                inputs = F.one_hot(inputs, max_value+1)
            else:
                inputs = inputs.unsqueeze(-1)
            inputs = inputs.float().to(device)

            sample = []
            for _ in range(num_sample):
                dist = torch.distributions.Categorical(logits=model(inputs))
                dist = torch.distributions.Independent(dist, 1)
                sample.append(dist.log_prob(targets) - log(num_sample))
            sample = torch.stack(sample, -1)
            score += sample.logsumexp(-1).sum().item()

    return score


def log_posterior(dataloader: torch.utils.data.DataLoader,
                  model: MADE, num_sample: int,
                  one_hot_input: bool,  device: Union[str, torch.device],
                  max_value: int) -> float:
    score = 0
    model = model.to(device)

    model.eval()
    with torch.inference_mode():
        for inputs in dataloader:
            targets = inputs.to(device)

            if one_hot_input:
                inputs = F.one_hot(inputs, max_value+1)
            else:
                inputs = inputs.unsqueeze(-1)
            inputs = inputs.float().to(device)

            order_original = torch.arange(targets.size(-1))
            order_original = order_original.to(device)

            for shifts in range(order_original.size(-1)):
                order = order_original.roll(shifts)
                sample = []
                for _ in range(num_sample):
                    order_ = order.clone()
                    order_[:-1] = order_[:-
                                         1][torch.randperm(order_[:-1].size(0))]
                    outputs = model(inputs, order_)
                    outputs = outputs.index_select(dim=1, index=order[-1])
                    dist = torch.distributions.Categorical(
                        logits=outputs)
                    dist = torch.distributions.Independent(dist, 1)
                    sample.append(dist.log_prob(
                        targets.index_select(dim=1, index=order[-1])))
                sample = torch.stack(sample, -1) - log(num_sample)
                score += sample.logsumexp(-1).sum().item()

    return score
