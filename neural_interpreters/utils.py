from collections import defaultdict

import os
from copy import deepcopy

from typing import Union, Dict, Callable, List, Tuple

import numpy as np
import yaml
from einops import rearrange
from addict import Addict

import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field


def gather(tensor: torch.Tensor, world_size: int):
    containers = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.barrier()
    dist.all_gather(containers, tensor)
    return containers


def gather_reduce(tensor: torch.Tensor, world_size: int, reduction: str = "mean"):
    containers = gather(tensor, world_size)
    pre_reduced = torch.stack(containers)
    if reduction == "mean":
        return pre_reduced.mean(0)
    elif reduction == "sum":
        return pre_reduced.sum(0)
    elif reduction == "stack":
        return pre_reduced
    else:
        raise NotImplementedError(f"Reduction mode {reduction} not implemented.")


def sync_values(
    value, reduction: str, device: Union[str, torch.device], world_size: int
):
    if not torch.is_tensor(value):
        if isinstance(value, np.ndarray):
            value_type = "ndarray"
        elif isinstance(value, (float, int)):
            value_type = "py_scalar"
        else:
            raise TypeError
        value = torch.tensor(value).to(device)
    else:
        value_type = "tensor"
    gathered_value = gather_reduce(value, world_size, reduction=reduction)
    if value_type != "tensor":
        gathered_value = gathered_value.cpu()
        if value_type == "ndarray":
            gathered_value = gathered_value.numpy()
        elif value_type == "py_scalar":
            gathered_value = gathered_value.item()
    return gathered_value


class NotAvailable(object):
    @classmethod
    def it_is(cls, obj):
        return isinstance(obj, cls)


@dataclass
class ModelOutput(object):
    predictions: Dict[str, torch.Tensor]
    output_patches: Union[torch.Tensor, None]
    input_patches: Union[torch.Tensor, None]


@dataclass
class LossOutput(object):
    value: torch.Tensor
    loss_terms: Dict[str, torch.Tensor]
    count: int = 0

    @torch.no_grad()
    def accumulate(self, other: "LossOutput") -> "LossOutput":
        self.value += other.value
        for head_name in self.loss_terms.keys():
            self.loss_terms[head_name] += other.loss_terms[head_name]
        self.count += 1
        return self

    @torch.no_grad()
    def aggregate(self, mode: str = "mean") -> "LossOutput":
        if mode == "sum":
            self.count = 0
            return self
        else:
            assert mode == "mean"
        self.value /= self.count
        for head_name in self.loss_terms.keys():
            self.loss_terms[head_name] /= self.count
        self.count = 0
        return self

    def sync(self, sync_fn: Callable) -> "LossOutput":
        synced_value = sync_fn(value=self.value, reduction="mean")
        synced_loss_terms = {}
        for head_name in self.loss_terms.keys():
            synced_loss_terms[head_name] = sync_fn(
                value=self.loss_terms[head_name], reduction="mean"
            )
        return LossOutput(
            value=synced_value, loss_terms=synced_loss_terms, count=self.count
        )

    def state_dict(self):
        state = {"value": self.value.item()}
        for key, value in self.loss_terms.items():
            state[key] = value.item()
        return state

    def detach(self, copy=True):
        loss_value = self.value.detach()
        loss_terms = {}
        for key, value in self.loss_terms.items():
            loss_terms[key] = value.detach()
        if copy:
            (loss_value, loss_terms, count) = deepcopy(
                (loss_value, loss_terms, self.count)
            )
        else:
            count = self.count
        return LossOutput(value=loss_value, loss_terms=loss_terms, count=count)

    def to(self, device, copy=False):
        if copy:
            _self = deepcopy(self)
        else:
            _self = self
        _self.value = _self.value.to(device)
        loss_terms = {}
        for key, value in _self.loss_terms.items():
            loss_terms[key] = value.to(device)
        _self.loss_terms = loss_terms
        return _self

    def cpu(self, copy=False):
        return self.to("cpu", copy=copy)


@dataclass
class AccuracyOutput(object):
    individual_accuracies: Dict[str, torch.Tensor]
    individual_counts: Dict[str, torch.Tensor]

    @torch.no_grad()
    def accumulate(self, other: "AccuracyOutput"):
        for dataset_name in self.individual_accuracies.keys():
            # Compute self attries
            self_accuracy = self.individual_accuracies[dataset_name]
            self_counts = self.individual_counts[dataset_name]
            self_hits = self_accuracy * self_counts
            # Compute other attries
            other_accuracy = other.individual_accuracies[dataset_name]
            other_counts = other.individual_counts[dataset_name]
            other_hits = other_accuracy * other_counts
            # Accumulate
            new_hits = self_hits + other_hits
            new_count = self_counts + other_counts
            self.individual_accuracies[dataset_name] = new_hits / new_count
            self.individual_counts[dataset_name] = new_count
        return self

    @torch.no_grad()
    def aggregate(self, mode: str = "mean") -> "AccuracyOutput":
        return self

    @property
    def combined(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack(list(self.individual_accuracies.values())).mean()

    @property
    def combined_counts(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack(list(self.individual_counts.values())).sum()

    def sync(self, sync_fn: Callable) -> "AccuracyOutput":
        synced_individual_accuracies = {}
        synced_individual_counts = {}
        for dataset_name in self.individual_accuracies.keys():
            counts = self.individual_counts[dataset_name]
            accuracy = self.individual_accuracies[dataset_name]
            hits = accuracy * counts
            synced_counts = sync_fn(value=counts, reduction="sum")
            synced_hits = sync_fn(value=hits, reduction="sum")
            synced_accuracy = synced_hits / synced_counts
            # Write out
            synced_individual_counts[dataset_name] = synced_counts
            synced_individual_accuracies[dataset_name] = synced_accuracy
        return AccuracyOutput(
            individual_accuracies=synced_individual_accuracies,
            individual_counts=synced_individual_counts,
        )

    def state_dict(self):
        return {k: v.item() for k, v in self.individual_accuracies.items()}


@dataclass()
class WILDSMetricAccumulator(object):
    predictions: Dict[str, List[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    labels: Dict[str, List[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    metadata: Dict[str, List[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @torch.no_grad()
    def accumulate(
        self,
        model_output: ModelOutput,
        data: Tuple[
            torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]
        ],
    ):
        images, labels, metadata = data
        for name in model_output.predictions.keys():
            assert name in labels
            assert name in metadata
            raw_predictions = torch.argmax(model_output.predictions[name], dim=-1).cpu()
            raw_labels = labels[name].cpu()
            raw_metadata = metadata[name][0].cpu()
            self.predictions[name].append(raw_predictions)
            self.labels[name].append(raw_labels)
            self.metadata[name].append(raw_metadata)
        return self

    @torch.no_grad()
    def aggregate(self):
        # Concatenate everything along 0th axis
        predictions = {
            key: [torch.cat(val, dim=0)] for key, val in self.predictions.items()
        }
        labels = {key: [torch.cat(val, dim=0)] for key, val in self.labels.items()}
        metadata = {key: [torch.cat(val, dim=0)] for key, val in self.metadata.items()}
        return WILDSMetricAccumulator(
            predictions=predictions, labels=labels, metadata=metadata
        )


def make_mlp(
    in_features: int,
    out_features: int,
    capacity: int = None,
    num_layers: int = 2,
    activation: Union[Callable, None] = nn.ReLU,
    dropout: float = 0.0,
    trailing_dropout: bool = False,
    trailing_activation: bool = False,
    leading_dropout: bool = False,
    leading_activation: bool = False,
    trailing_bias: bool = True,
) -> nn.Module:
    """Utility function for rolling a quick MLP."""
    layer_sequence = []
    if activation is not None and leading_activation:
        layer_sequence.append(activation())
    if dropout > 0.0 and leading_dropout:
        layer_sequence.append(nn.Dropout(p=dropout))
    if num_layers == 1:
        layer_sequence.append(nn.Linear(in_features, out_features, bias=trailing_bias))
    elif num_layers >= 2:
        assert activation is not None
        layer_sequence.extend([nn.Linear(in_features, capacity), activation()])
        if dropout > 0.0:
            layer_sequence.append(nn.Dropout(p=dropout))
        for layer_num in range(num_layers - 2):
            layer_sequence.extend([nn.Linear(capacity, capacity), activation()])
            if dropout > 0.0:
                layer_sequence.append(nn.Dropout(p=dropout))
        layer_sequence.append(nn.Linear(capacity, out_features, bias=trailing_bias))
    else:
        raise ValueError
    if activation is not None and trailing_activation:
        layer_sequence.append(activation())
    if dropout > 0.0 and trailing_dropout:
        layer_sequence.append(nn.Dropout(p=dropout))
    if len(layer_sequence) == 1:
        return layer_sequence[0]
    else:
        return nn.Sequential(*layer_sequence)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), "n i j -> (i j) n")  # [n*n, 2] pos[n] = (i, j)
    rel_pos = (
        pos[None, :] - pos[:, None]
    )  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos


def freeze_existing_module_parameters(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module


def unfreeze_existing_module_parameters(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = True
    return module


def get_latest_checkpoint_path(checkpoint_directory: str):
    # We'll need to figure out what the latest checkpoint is
    latest_checkpoint_iter = max(
        [int(fn.strip("ckpt_iter_.pt")) for fn in os.listdir(checkpoint_directory)]
    )
    # Now, we make the file name and return
    checkpoint_path = os.path.join(
        checkpoint_directory, f"ckpt_iter_{latest_checkpoint_iter}.pt"
    )
    assert os.path.exists(checkpoint_path), (
        f"Path {checkpoint_path} was " f"expected to exist, but it doesn't."
    )
    return checkpoint_path


class DummyModule(nn.Module):
    def __init__(self, *_, **__):
        super(DummyModule, self).__init__()

    def forward(self, *_, **__):
        return self


def read_yaml(path):
    with open(path, "r") as f:
        d = Addict(yaml.load(f, Loader=yaml.FullLoader))
    return d
