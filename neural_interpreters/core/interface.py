import math
from copy import deepcopy

from addict import Dict
from typing import Optional

import torch
import torch.nn as nn

from neural_interpreters.baselines.base_model import _BaseModel
from neural_interpreters.core.nfc import Program
from neural_interpreters.core.utils import (
    PositionalEncoding,
    ProgramOutput,
    NeuralInterpreterOutput,
)


class NeuralInterpreter(_BaseModel):
    def __init__(
        self,
        *,
        program_kwargs: dict,
        num_classes_in_dataset,
        use_shared_cls_tokens=False,
        use_shared_prediction_head=False,
        patch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        num_entities: Optional[int] = None,
        image_input_channels: int = 3,
        pre_positional_encoding: bool = False,
        layerwise_positional_encoding: bool = True,
        use_pre_input_layernorm: bool = False,
        detach_patch_embeddings: bool = False,
        input_type: str = "patches",
    ):
        assert input_type in ["patches", "entities"]
        dim = program_kwargs["script_kwargs"]["variable_features"]
        super(NeuralInterpreter, self).__init__(
            dim=dim,
            num_classes_in_dataset=num_classes_in_dataset,
            use_shared_cls_tokens=use_shared_cls_tokens,
            use_shared_prediction_head=use_shared_prediction_head,
            patch_size=patch_size,
            image_size=image_size,
            image_input_channels=image_input_channels,
            num_entities=num_entities,
            input_type=input_type,
            detach_patch_embeddings=detach_patch_embeddings
        )
        self._init_num_classes()
        self._init_cls_tokens()
        self._init_prediction_heads()
        if input_type == "patches":
            self._init_patching()
        elif input_type == "entities":
            self._init_entities()
        self.pre_positional_encoding = pre_positional_encoding
        self.layerwise_positional_encoding = layerwise_positional_encoding
        self.use_pre_input_layernorm = use_pre_input_layernorm
        self.transformer = Program(**self._patch_program_kwargs(program_kwargs))
        if self.use_pre_input_layernorm:
            self.pre_input_layernorm = nn.LayerNorm(self.dim)
        else:
            self.pre_input_layernorm = nn.Identity()

    def _patch_program_kwargs(self, program_kwargs: dict) -> dict:
        program_kwargs = Dict(deepcopy(program_kwargs))
        if self.input_type == "patches":
            patched_image_size = int(math.sqrt(self.num_patches))
        else:
            patched_image_size = None
        program_kwargs.script_kwargs.function_pod_kwargs.loc_pod_kwargs.image_size = (
            patched_image_size
        )
        return program_kwargs.to_dict()

    def forward(self, image, mask=None):
        if self.input_type == "patches":
            _patches = self._extract_and_embed_patches(image)
            x, (b, n, _) = _patches["embedded_patches"], _patches["embedded_shape"]
        elif self.input_type == "entities":
            _entities = self._extract_and_embed_entities(image)
            x, (b, n, _) = _entities["embedded_entities"], _entities["embedded_shape"]
        else:
            raise ValueError
        if self.detach_patch_embeddings:
            x = x.detach()
        input_patches = x

        # Token bookkeeping
        cls_tokens = self._get_cls_tokens(batch_size=b)
        x = torch.cat(cls_tokens + [x], dim=1)

        # Init and apply the positional encoding
        positional_encoding = PositionalEncoding(self.pos_embedding)
        if self.pre_positional_encoding:
            x = positional_encoding.apply(x)
        if not self.layerwise_positional_encoding:
            # This makes sure that the inner layer don't get pos emb.
            positional_encoding = None
        # Apply the model
        x = self.pre_input_layernorm(x)
        x = self.dropout(x)
        program_output = self.transformer(
            x, mask, positional_encoding=positional_encoding
        )  # type: ProgramOutput
        x = program_output.output

        # Run it through the prediction heads
        outputs = self._get_predictions(x)

        return NeuralInterpreterOutput(
            predictions=outputs,
            output_patches=x[:, len(cls_tokens) :],
            input_patches=input_patches,
            program_output=program_output,
        )

    def add_functions(self, *args, **kwargs):
        self.transformer.add_functions(*args, **kwargs)
        return self

    def finetune(
        self,
        *,
        num_new_functions: Optional[int] = None,
        freeze_existing_parameters=False,
        **super_kwargs,
    ) -> "NeuralInterpreter":
        model = super(NeuralInterpreter, self).finetune(
            freeze_existing_parameters=freeze_existing_parameters, **super_kwargs
        )
        if num_new_functions is not None:
            model = model.add_functions(
                num_new_functions=num_new_functions,
                freeze_existing_parameters=freeze_existing_parameters,
            )
        return model
