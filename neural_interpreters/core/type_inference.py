from typing import Optional

import torch
import torch.nn as nn

from neural_interpreters.core.utils import VectorQuantizer, QuantizedTensorContainer
from neural_interpreters.utils import make_mlp


class TypeInference(nn.Module):
    def __init__(
        self,
        variable_features: int,
        type_features: int,
        num_types: int,
        capacity: Optional[int] = None,
        depth: int = 2,
        use_layernorm: bool = True,
    ):
        super(TypeInference, self).__init__()
        self.variable_features = variable_features
        self.type_features = type_features
        self.num_types = num_types
        self.capacity = capacity or variable_features
        self.depth = depth
        self.use_layernorm = use_layernorm
        self.body = make_mlp(
            in_features=self.variable_features,
            out_features=self.type_features,
            capacity=self.capacity,
            num_layers=self.depth,
            activation=nn.GELU,
            trailing_activation=False,
        )
        self.quantizer = VectorQuantizer(
            num_features=self.type_features, num_quantities=self.num_types
        )
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.variable_features)
        else:
            self.layernorm = nn.Identity()

    def forward(
        self, variables: torch.Tensor, quantize: bool = True
    ) -> QuantizedTensorContainer:
        # variables.shape = B...C
        # types.shape = B...C
        unquantized_types = self.body(self.layernorm(variables))
        if quantize:
            types = self.quantizer(unquantized_types)
        else:
            types = QuantizedTensorContainer(
                quantized=unquantized_types,
                commitment_loss=None,
                codebook=None,
                input=unquantized_types,
            )
        return types
