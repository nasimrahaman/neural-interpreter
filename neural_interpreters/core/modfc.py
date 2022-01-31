import math
from typing import Optional, Callable, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_interpreters.utils import make_mlp
import einops as eo


def get_activation(activation: Union[str, nn.Module, type, None]) -> nn.Module:
    if isinstance(activation, str):
        activation = getattr(torch.nn, activation)()
    elif isinstance(activation, nn.Module):
        pass
    elif isinstance(activation, type):
        activation = activation()
    else:
        assert activation is None, f"Can't parse: {activation}"
    return activation


class ModFC(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Callable] = None,
        eps: float = 1e-6,
    ):
        super(ModFC, self).__init__()
        self.native_weight = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(out_features, in_features), a=math.sqrt(5)
            )
        )
        if bias:
            self.native_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.native_bias = None
        self.scale = nn.Linear(context_features, in_features, bias=False)
        self.activation = activation
        self.eps = eps

    def _einsum_implementation(
        self, input: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # input.shape = ...C
        # context.shape = ...C
        # Compute the scale to modulate with
        scale = self.scale(context)
        # Now, compute the prenormalized weights
        prenormalized_weights = torch.einsum(
            "...j,ij->...ij", scale, self.native_weight
        )
        normalizer = (
            prenormalized_weights.pow(2).sum(-1, keepdim=True) + self.eps
        ).sqrt()
        normalized_weights = prenormalized_weights / normalizer
        # Do the dot product and add in the bias
        weights_dot_input = torch.einsum("...ij,...j->...i", normalized_weights, input)
        # Add in the bias (should be broadcastable)
        if self.native_bias is not None:
            output = weights_dot_input + self.native_bias
        else:
            output = weights_dot_input
        # Activate if required
        if self.activation is not None:
            output = self.activation(output)
        return output

    def _naive_implementation(
        self, input: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # input.shape = ...C
        # context.shape = ...C
        # scale.shape = ...C
        scale = F.layer_norm(self.scale(context), [input.shape[-1]])
        output = F.linear(input * scale, self.native_weight, bias=self.native_bias)
        # Activate if required
        if self.activation is not None:
            output = self.activation(output)
        return output

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        return self._naive_implementation(input, context)


class FC(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Callable] = None,
    ):
        super(FC, self).__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        output = self.lin(input)
        if self.activation is not None:
            output = self.activation(output)
        return output


class GModFC(nn.Module):
    """Gated ModFC Layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Optional[Union[Callable, str]] = None,
        apply_activation_on: str = "aux",
    ):
        super(GModFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.bias = bias
        assert apply_activation_on in ["main", "aux", "both"]
        self.apply_activation_on = apply_activation_on
        # Modules
        self.main_lin = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        self.aux_lin = nn.Linear(
            self.context_features, self.out_features, bias=self.bias
        )
        self.activation = get_activation(activation)

    def forward(self, input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        main_output = self.main_lin(input)
        aux_output = self.aux_lin(context)
        if self.activation is not None:
            if self.apply_activation_on == "main":
                output = self.activation(main_output) * aux_output
            elif self.apply_activation_on == "aux":
                output = main_output * self.activation(aux_output)
            elif self.apply_activation_on == "both":
                output = self.activation(main_output) * self.activation(aux_output)
            else:
                raise ValueError(
                    f"Unexpected thing to apply activation "
                    f"on: {self.apply_activation_on}"
                )
        else:
            # No activation, so we're doing the "bilinear" thing (though it's not that)
            output = main_output * aux_output
        return output


class HyperFC(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_features: int,
        bias: bool = True,
        *,
        activation: Union[str, nn.Module, None] = None,
        use_context_pre_layernorm: bool = True,
        hyper_capacity: Optional[int] = None,
        hyper_capacity_compression: Optional[int] = None,
        hyper_activation: Union[str, nn.Module, None] = "GELU",
        hyper_depth: int = 2,
        reduce_context: bool = False,
    ):
        super(HyperFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.bias = bias
        self.activation = get_activation(activation)
        self.use_context_pre_layernorm = use_context_pre_layernorm
        self.reduce_context = reduce_context
        # Initialize the hypernetwork
        self.num_weight_params = self.in_features * self.out_features
        self.num_bias_params = self.out_features if self.bias else 0
        self.num_params = self.num_weight_params + self.num_bias_params
        if hyper_capacity is not None:
            self.hyper_capacity = hyper_capacity
        elif hyper_capacity_compression is not None:
            self.hyper_capacity = self.in_features // hyper_capacity_compression
        else:
            self.hyper_capacity = self.in_features
        self.hyper_activation = type(get_activation(hyper_activation))
        self.hyper_depth = hyper_depth
        self.hyper_net = make_mlp(
            in_features=self.context_features,
            out_features=self.num_params,
            capacity=self.hyper_capacity,
            activation=self.hyper_activation,
            num_layers=self.hyper_depth,
            trailing_activation=False,
            trailing_bias=False,
        )
        if self.use_context_pre_layernorm:
            self.context_pre_layernorm = nn.LayerNorm(self.context_features)
        else:
            self.context_pre_layernorm = nn.Identity()
        self.slow_params = nn.Parameter(self._get_slow_params())
        # This mixes between a fast hypernetwork and the slow parameters.
        # Initing it to zero means that the initial parameters are slow
        self.mixing_alpha = nn.Parameter(torch.tensor(0.0))

    def _get_slow_params(self):
        # This initializes the weights like they would be in nn.Linear
        slow_fc = nn.Linear(self.in_features, self.out_features, self.bias)
        # slow_fc.weight.shape = OI
        slow_weights = eo.rearrange(slow_fc.weight.data, "o i -> (o i)")
        if self.bias:
            slow_bias = slow_fc.bias.data
            slow_params = torch.cat([slow_weights, slow_bias], dim=0)
        else:
            slow_params = slow_weights
        return slow_params

    def form_params(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        # This converts a single large vector in to two weights and bias vectors
        assert params.shape[-1] == self.num_params
        # Where O = self.out_features and I = self.in_features, we have:
        # weight.shape = ...OI
        # bias.shape = ...O or None
        weight = eo.rearrange(
            params[..., 0 : self.num_weight_params],
            "... (o i) -> ... o i",
            o=self.out_features,
            i=self.in_features,
        )
        if self.bias:
            bias = params[..., self.num_weight_params :]
        else:
            bias = None
        return weight, bias

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        # input.shape = ...C
        # context.shape = ...C OR UC
        if self.reduce_context:
            # context.shape = BUVC, which we reduce to UC
            assert context.dim() == 4, "Can't reduce context if not 4D."
            context = context[0, :, 0, :]
        # First, get the parameters
        # params.shape = ...P
        fast_params = self.hyper_net(self.context_pre_layernorm(context))
        slow_params = self.slow_params
        # This entails some broadcasting, but it should be fine.
        params = self.mixing_alpha * fast_params + slow_params
        # Reshape to weight and bias.
        # Where O = self.out_features and I = self.in_features, we have:
        # weight.shape = ...OI
        # bias.shape = ...O
        weight, bias = self.form_params(params)
        # We now have two cases. The efficient one is where context.shape = UC
        if context.dim() == 2 and input.dim() == 4:
            # This is the efficient code path, where we know that:
            # input.shape = BUVC
            # context.shape = UC
            output = torch.einsum("uoi,buvi->buvo", weight, input)
            if bias is not None:
                # bias.shape = UO
                output = output + bias[None, :, None, :]
        else:
            # In this code path where we don't know anything
            output = torch.einsum("...oi,...i->...o", weight, input)
            if bias is not None:
                # bias.shape = ...O
                output = output + bias
        # Activate
        if self.activation is not None:
            output = self.activation(output)
        return output


class ModSequential(nn.Sequential):
    # noinspection PyMethodOverriding
    def forward(self, input: torch.Tensor, context: torch.Tensor):
        for module in self:
            input = module(input, context)
        return input


def get_modfc_cls(name, **kwargs) -> Callable:
    cls = globals()[name]
    return lambda *args, **kwargs_: cls(*args, **{**kwargs, **kwargs_})
