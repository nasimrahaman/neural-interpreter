import math

from typing import Optional, Iterable, Tuple, Union, List

import torch
import torch.nn as nn
import einops as eo

from torch.utils.hooks import RemovableHandle

from neural_interpreters.core.kernels import Kernel, DotProductKernel
from neural_interpreters.core.modfc import ModSequential, get_modfc_cls
from neural_interpreters.core.type_inference import TypeInference
from neural_interpreters.core.utils import (
    QuantizedTensorContainer,
    LOCPodOutput,
    FunctionPodOutput,
    ScriptOutput,
    ProgramOutput,
    PositionalEncoding,
    RelativePositionalEncoder,
    FunctionDropout,
)
from neural_interpreters.utils import make_mlp


class LOCPod(nn.Module):
    """
    A LOCPod is a collection of Lines of Code (LOC) that are run in parallel.
    """

    VALID_RESIDUAL_MODES = {"gating", "vanilla", "modulated_attn"}

    def __init__(
        self,
        variable_features: int,
        result_features: int,
        key_query_features: int,
        code_features: int,
        num_heads: int = 1,
        no_residual_after_attention: bool = False,
        residual_mode: str = "gating",
        trailing_activation=True,
        use_fc_after_attention: bool = False,
        scale_attention_weights: bool = False,
        layernorm_before_positional_encoding: bool = False,
        relative_positional_encoder: Union[RelativePositionalEncoder, bool] = None,
        relative_positional_encoder_kwargs: Optional[dict] = None,
        mod_fc_class: str = "ModFC",
        mod_fc_kwargs: Optional[dict] = None,
        image_size: Optional[int] = None,
        epsilon: float = 1e-6,
    ):
        super(LOCPod, self).__init__()
        assert residual_mode in self.VALID_RESIDUAL_MODES
        # Meta
        self.variable_features = variable_features
        self.result_features = result_features
        self.key_query_features = key_query_features
        self.code_features = code_features
        self.num_heads = num_heads
        self.no_residual_after_attention = no_residual_after_attention
        self.residual_mode = residual_mode
        self.trailing_activation = trailing_activation
        self.use_fc_after_attention = use_fc_after_attention
        self.scale_attention_weights = scale_attention_weights
        self.layernorm_before_positional_encoding = layernorm_before_positional_encoding
        self.epsilon = epsilon
        # Submodules
        # Positional Encoding
        if (
            isinstance(relative_positional_encoder, bool)
            and relative_positional_encoder
        ):
            # This is the case where relative_positional_encoder is True (or False)
            # We auto-create one.
            assert image_size is not None, (
                "image_size must be given if " "relative_positional_encoder is True"
            )
            self.relative_positional_encoder = RelativePositionalEncoder(
                num_features=key_query_features,
                image_size=image_size,
                num_heads=num_heads,
                code_features=code_features,
                **(relative_positional_encoder_kwargs or {}),
            )
        elif isinstance(relative_positional_encoder, nn.Module):
            self.relative_positional_encoder = relative_positional_encoder
        else:
            assert relative_positional_encoder is None
            self.relative_positional_encoder = nn.Identity()
        # Layer norms
        self.pre_attention_layernorm = nn.LayerNorm(variable_features)
        self.pre_mlp_layernorm = nn.LayerNorm(result_features)
        # All classes to do with ModFC
        modfc_cls = get_modfc_cls(mod_fc_class, **(mod_fc_kwargs or {}))
        # QKV projectors
        self.query_compiler = modfc_cls(
            variable_features,
            key_query_features * num_heads,
            code_features,
            bias=False,
        )
        self.key_compiler = modfc_cls(
            variable_features,
            key_query_features * num_heads,
            code_features,
            bias=False,
        )
        self.value_compiler = modfc_cls(
            variable_features, result_features, code_features
        )
        # Post attention linear layer
        if use_fc_after_attention:
            self.post_attention_fc = modfc_cls(
                result_features, result_features, code_features, activation=None,
            )
        else:
            self.post_attention_fc = None
        # MLP
        self.pointwise_mlp = ModSequential(
            modfc_cls(
                result_features, result_features, code_features, activation=nn.GELU()
            ),
            modfc_cls(
                result_features,
                result_features,
                code_features,
                activation=(nn.GELU() if trailing_activation else nn.Identity()),
            ),
        )
        # Gains. These may help stabilize the training at initialization.
        self.register_buffer("post_attention_gain", torch.tensor(2.5))
        self.register_buffer("post_mlp_gain", torch.tensor(2.5))

    def forward(
        self,
        variables: torch.Tensor,
        codes: torch.Tensor,
        function_variable_affinities: torch.Tensor,
        positional_encoding: Optional[PositionalEncoding] = None,
    ):
        # Function signature: BUVC, {UC, BUVC}, BUV -> BUVC
        # variables.shape = BUVC
        # signatures.shape = UC
        # codes.shape = UC or BUVC
        # variable_function_affinities.shape = BUV
        # Validate shapes
        assert variables.dim() == 4, (
            f"variables.ndim should be 4 (BUVC) if "
            f"is_header=False, got {variables.dim()} instead."
        )
        B, U, V, C = variables.shape
        if codes.dim() == 2:
            # Expecting a UC tensor
            assert list(codes.shape) == [U, self.code_features]
            codes = eo.repeat(codes, "u c -> b u v c", b=B, v=V)
        else:
            assert list(codes.shape) == [B, U, V, C]
        assert list(function_variable_affinities.shape) == [B, U, V]
        # Init a dummy positional encoding if required
        positional_encoding = (
            PositionalEncoding(None)
            if positional_encoding is None
            else positional_encoding
        )
        # Expand codes to the right shape
        # Apply the layer norm
        if self.layernorm_before_positional_encoding:
            # First layernorm, then positional encoding
            normalized_variables = positional_encoding.apply(
                self.pre_attention_layernorm(variables)
            )
        else:
            normalized_variables = self.pre_attention_layernorm(
                positional_encoding.apply(variables)
            )
        # Compute queries, keys and values
        # queries.shape = BUV(HC) ---> BUVHC
        queries = self.query_compiler(normalized_variables, codes)
        queries = eo.rearrange(queries, "b u v (h c) -> b u v h c", h=self.num_heads)
        keys = self.key_compiler(normalized_variables, codes)
        keys = eo.rearrange(keys, "b u v (h c) -> b u v h c", h=self.num_heads)
        values = self.value_compiler(normalized_variables, codes)
        values = eo.rearrange(values, "b u v (h c) -> b u v h c", h=self.num_heads)
        # Get the scale factor for DPA weights
        attention_scale = (
            math.sqrt(keys.shape[-1]) if self.scale_attention_weights else 1.0
        )
        # Compute the DPA weights
        pe_kwargs = (
            {}
            if isinstance(self.relative_positional_encoder, nn.Identity)
            else {"codes": codes}
        )
        dot_product_weights = torch.softmax(
            self.relative_positional_encoder(
                torch.einsum("burhc,bushc->bursh", queries, keys), **pe_kwargs,
            )
            / attention_scale,
            dim=-2,
        )
        # Modulate with kernel
        kernel_modulated_weights = torch.einsum(
            "bur,bus,bursh->bursh",
            function_variable_affinities,
            function_variable_affinities,
            dot_product_weights,
        )
        # Normalize the kernel modulated weights
        kernel_modulated_weights = kernel_modulated_weights / (
            kernel_modulated_weights.sum(-2, keepdim=True) + self.epsilon
        )
        # Modulate values to compute output
        attention_pre_outputs = eo.rearrange(
            torch.einsum("bursh,bushc->burhc", kernel_modulated_weights, values),
            "b u r h c -> b u r (h c)",
        )
        if self.post_attention_fc is not None:
            attention_pre_outputs = self.post_attention_fc(attention_pre_outputs, codes)
        # Now, the question is how to manage the update for variables which are not
        # supposed to be handled by a function.
        # We have two codepaths -- one where we don't apply a residual connection and
        # one where we do. Where we don't apply residual, we kill the elements that
        # aren't supposed to be touched by the function. Where we apply a residual,
        # we use the kernel as a normalized gating (that sums to one).
        affinity_gate = function_variable_affinities[..., None]
        if (
            self.variable_features != self.result_features
        ) or self.no_residual_after_attention:
            attention_outputs = torch.einsum(
                "buvc,buv->buvc", attention_pre_outputs, function_variable_affinities
            )
        else:
            if self.residual_mode == "gating":
                # If kernel weight is zero, we let the inputs (variables) pass right
                # through like an identity.
                attention_outputs = ((1 - affinity_gate) * variables) + (
                    affinity_gate * self.post_attention_gain * attention_pre_outputs
                )
            elif self.residual_mode == "vanilla":
                # This code path might be weird because attention_pre_outputs is
                # not weighted by the affinities.
                attention_outputs = variables + attention_pre_outputs
            elif self.residual_mode == "modulated_attn":
                # This should have been 'vanilla', but that was already taken
                # (and we don't want to bork older configs)
                attention_outputs = variables + affinity_gate * attention_pre_outputs
            else:
                raise NotImplementedError
        # Normalize for the next run
        normalized_attention_outputs = self.pre_mlp_layernorm(attention_outputs)
        # Run through the pointwise MLP
        # results.shape = BUVC
        mlp_outputs = self.pointwise_mlp(normalized_attention_outputs, codes)
        # Modulate the output by the kernel. The story here is the same as above.
        mlp_outputs = torch.einsum(
            "buvc,buv->buvc", mlp_outputs, function_variable_affinities
        )
        if self.residual_mode == "gating":
            # If the kernel is zero, we let the attention outputs pass right through
            # (like an identity).
            results = ((1 - affinity_gate) * attention_outputs) + (
                affinity_gate * self.post_mlp_gain * mlp_outputs
            )
        elif self.residual_mode in ["vanilla", "modulated_attn"]:
            results = attention_outputs + mlp_outputs
        else:
            raise NotImplementedError
        return LOCPodOutput(input=variables, output=results)


class FunctionPod(nn.Module):
    """
    A FunctionPod is a collection of functions that are run in parallel.
    Each function comprises multiple lines of code -- in analogy, each
    `FunctionPod` comprises a bunch of `LOCPods`.
    """

    VALID_RESIDUAL_MODES = {"gating", "vanilla"}

    def __init__(
        self,
        *,
        num_functions: int,
        code_features: int,
        num_heads_in_loc: int,
        num_features_per_loc_head: int,
        num_loc_pods: int = 1,
        loc_pod_kwargs: Optional[dict] = None,
        quantize_function_signature: bool = True,
        quantize_variable_types: bool = True,
        variable_features: Optional[int] = None,
        type_features: Optional[int] = None,
        num_types: Optional[int] = None,
        type_inference: Optional[TypeInference] = None,
        kernel: Optional[Kernel] = None,
        kernel_kwargs: Optional[dict] = None,
        no_residual: bool = False,
        residual_mode: str = "gating",
        normalize_function_variable_affinities: bool = False,
        num_temporary_variables: int = 0,
        consolidate_function_embeddings: bool = False,
        function_embedding_features: Optional[int] = None,
        deconsolidation_mlp_capacity: Optional[int] = None,
        deconsolidation_mlp_depth: Optional[int] = 2,
        detach_function_signatures: bool = False,
        detach_function_codes: bool = False,
        detach_function_output_signatures: bool = False,
        detach_function_embeddings: bool = False,
        epsilon: float = 1e-6,
    ):
        super(FunctionPod, self).__init__()
        assert residual_mode in self.VALID_RESIDUAL_MODES
        # Meta
        self.num_functions = num_functions
        self.code_features = code_features
        self.num_loc_pods = num_loc_pods
        self.num_heads_in_loc = num_heads_in_loc
        self.num_features_per_loc_head = num_features_per_loc_head
        self.quantize_function_signature = quantize_function_signature
        self.quantize_variable_types = quantize_variable_types
        self.no_residual = no_residual
        self.residual_mode = residual_mode
        self.normalize_function_variable_affinities = (
            normalize_function_variable_affinities
        )
        self.num_temporary_variables = num_temporary_variables
        self.consolidate_function_embeddings = consolidate_function_embeddings
        self.detach_function_signatures = detach_function_signatures
        self.detach_function_codes = detach_function_codes
        self.detach_function_output_signatures = detach_function_output_signatures
        self.detach_function_embeddings = detach_function_embeddings
        self.epsilon = epsilon
        if type_inference is not None:
            self.type_inference = type_inference
            self.variable_features = type_inference.variable_features
            self.type_features = type_inference.type_features
        else:
            assert None not in [variable_features, type_features, num_types]
            self.variable_features = variable_features
            self.type_features = type_features
            self.type_inference = TypeInference(
                variable_features=variable_features,
                type_features=type_features,
                num_types=num_types,
            )
        self.quantizer = self.type_inference.quantizer
        self.kernel = (
            kernel if kernel is not None else DotProductKernel(**(kernel_kwargs or {}))
        )
        # Procedure embeddings
        self.function_embedding_features = (
            function_embedding_features
            if function_embedding_features is not None
            else self.variable_features
        )
        self.deconsolidation_mlp_capacity = (
            deconsolidation_mlp_capacity or self.function_embedding_features
        )
        self.deconsolidation_mlp_depth = deconsolidation_mlp_depth
        self._initialize_function_embeddings()
        # Temporary variables
        # self._initialize_temporary_variables()
        # LOCPods
        self.loc_pods = nn.ModuleList(
            [
                LOCPod(
                    variable_features=self.variable_features,
                    result_features=self.variable_features,
                    key_query_features=self.num_heads_in_loc
                    * self.num_features_per_loc_head,
                    code_features=self.code_features,
                    num_heads=self.num_heads_in_loc,
                    **(loc_pod_kwargs or {}),
                )
                for _ in range(self.num_loc_pods)
            ]
        )
        # Store for hooks
        self._hooks = {}

    def add_functions(
        self,
        num_new_functions: int,
        freeze_existing_parameters: bool = False,
        init_strategy: str = "random",
    ):
        def hook(grad):
            grad.data[:-num_new_functions] = 0.0
            return grad

        def extend_tensor(
            x: torch.Tensor, num_channels: int
        ) -> Tuple[torch.Tensor, Optional[RemovableHandle]]:
            if init_strategy == "random":
                extension = torch.randn(
                    num_new_functions, num_channels, device=x.device, dtype=x.dtype
                )
            elif init_strategy == "mirror":
                assert num_new_functions <= x.shape[0]
                extension_idxs = torch.randperm(x.shape[0], device=x.device)[
                    :num_new_functions
                ]
                extension = x.data[extension_idxs]
            else:
                raise NotImplementedError
            # Reparameterize
            extended = nn.Parameter(
                torch.cat([x.data, extension.data], dim=0), requires_grad=True
            )
            if freeze_existing_parameters:
                # Don't want gradients flowing to the frozen parts of the tensor
                hook_handle = extended.register_hook(hook)
            else:
                hook_handle = None
            return extended, hook_handle

        if self.consolidate_function_embeddings:
            (
                self.function_embeddings,  # noqa
                self._hooks["function_embeddings"],
            ) = extend_tensor(
                self.function_embeddings, self.function_embedding_features
            )
        else:
            (
                self.function_signatures,  # noqa
                self._hooks["function_signatures"],
            ) = extend_tensor(self.function_signatures, self.type_features)
            (
                self.function_output_signatures,  # noqa
                self._hooks["function_output_signatures"],
            ) = extend_tensor(self.function_output_signatures, self.type_features)
            self.function_codes, self._hooks["function_codes"] = extend_tensor(  # noqa
                self.function_codes, self.code_features
            )
        self.num_functions += num_new_functions
        return self

    def remove_functions(self, function_indices: List[int]):
        new_function_indices = [
            new_function_index
            for new_function_index in range(self.num_functions)
            if new_function_index not in function_indices
        ]
        new_num_functions = len(new_function_indices)
        new_function_indices = torch.tensor(
            new_function_indices, device=self.function_signatures.device
        )
        assert not self.consolidate_function_embeddings, (
            "Removing functions for consolidated function "
            "embeddings is not implemented as of now."
        )

        def rewrap(p):
            return torch.nn.Parameter(p.data[new_function_indices])

        # Remove signatures
        self.function_signatures = rewrap(self.function_signatures)  # noqa
        # Remove codes
        self.function_codes = rewrap(self.function_codes)  # noqa
        # Remove output signatures
        # noinspection PyAttributeOutsideInit
        self.function_output_signatures = rewrap(self.function_output_signatures)
        # Reduce the number of functions
        self.num_functions = new_num_functions
        return self

    def release_hooks(self):
        for key, hook in self._hooks.items():
            if hook is not None:
                hook.remove()
        self._hooks.clear()
        return self

    def _initialize_function_embeddings(self):
        if self.consolidate_function_embeddings:
            # Consolidated, so we only need a single parameter vector per function
            self.function_embeddings = nn.Parameter(
                torch.randn(self.num_functions, self.function_embedding_features)
            )
            shared_deconsolidation_mlp_kwargs = dict(
                capacity=self.deconsolidation_mlp_capacity,
                num_layers=self.deconsolidation_mlp_depth,
                trailing_activation=False,
                activation=nn.GELU,
            )
            self.function_embeddings_to_signatures = make_mlp(
                in_features=self.function_embedding_features,
                out_features=self.type_features,
                **shared_deconsolidation_mlp_kwargs,
            )
            self.function_embeddings_to_output_signatures = make_mlp(
                in_features=self.function_embedding_features,
                out_features=self.type_features,
                **shared_deconsolidation_mlp_kwargs,
            )
            self.function_embeddings_to_codes = make_mlp(
                in_features=self.function_embedding_features,
                out_features=self.code_features,
                **shared_deconsolidation_mlp_kwargs,
            )
            # Set the variables that are not needed to None
            self.function_signatures = None
            self.function_output_signatures = None
            self.function_codes = None
        else:
            # Not consolidated, so we have a separate parameter for signature,
            # code and output signature.
            self.function_signatures = nn.Parameter(
                torch.randn(self.num_functions, self.type_features)
            )
            self.function_output_signatures = nn.Parameter(
                torch.randn(self.num_functions, self.type_features)
            )
            self.function_codes = nn.Parameter(
                torch.randn(self.num_functions, self.code_features)
            )
            # Set the variables that are not needed to None
            self.function_embeddings = None
            self.function_embeddings_to_signatures = None
            self.function_embeddings_to_output_signatures = None
            self.function_embeddings_to_codes = None

    def _initialize_temporary_variables(self):
        if self.num_temporary_variables == 0:
            # Nothing to do here
            self.temporary_variables = None
            self.function_embeddings_to_temporary_variables = None
            return
        if self.consolidate_function_embeddings:
            # We're dealing with consolidated function embeddings, which means we'll
            # need to roll a MLP
            self.function_embeddings_to_temporary_variables = make_mlp(
                in_features=self.function_embedding_features,
                out_features=self.variable_features * self.num_temporary_variables,
                capacity=self.deconsolidation_mlp_capacity,
                num_layers=self.deconsolidation_mlp_depth,
                trailing_activation=False,
                activation=nn.GELU,
            )
            self.temporary_variables = None
        else:
            # We'll need to instantiate freely floating parameters
            self.temporary_variables = nn.Parameter(
                torch.randn(
                    self.num_functions,
                    self.num_temporary_variables,
                    self.variable_features,
                )
            )
            self.function_embeddings_to_temporary_variables = None

    def fetch_function_variables(self,) -> Iterable[torch.Tensor]:
        if self.consolidate_function_embeddings:
            if self.detach_function_embeddings:
                function_embeddings = self.function_embeddings.detach()
            else:
                function_embeddings = self.function_embeddings
            function_signatures = self.function_embeddings_to_signatures(
                function_embeddings
            )
            function_output_signatures = self.function_embeddings_to_output_signatures(
                function_embeddings
            )
            function_codes = self.function_embeddings_to_codes(function_embeddings)
        else:
            function_signatures = self.function_signatures  # noqa
            function_output_signatures = self.function_output_signatures  # noqa
            function_codes = self.function_codes
        if self.detach_function_signatures:
            function_signatures = function_signatures.detach()
        if self.detach_function_output_signatures:
            function_output_signatures = function_output_signatures.detach()
        if self.detach_function_codes:
            function_codes = function_codes.detach()
        return function_signatures, function_output_signatures, function_codes

    def fetch_temporary_variables(self, batch_size: int) -> torch.Tensor:
        assert self.num_temporary_variables > 0
        if self.consolidate_function_embeddings:
            # temporary_variables.shape = u(vc)
            temporary_variables = self.function_embeddings_to_temporary_variables(
                self.function_embeddings
            )
            temporary_variables = eo.repeat(
                temporary_variables,
                "u (v c) -> b u v c",
                b=batch_size,
                v=self.num_temporary_variables,
            )
        else:
            temporary_variables = eo.repeat(
                self.temporary_variables, "u v c -> b u v c", b=batch_size
            )
        # temporary_variables.shape = b u v c
        return temporary_variables

    def append_temporary_variables(
        self, variables: torch.Tensor, function_variable_affinities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # variables.shape = BUVC
        # function_variable_affinities.shape = BUV
        if self.num_temporary_variables == 0:
            # Nothing to do here, this function is a no-op.
            return variables, function_variable_affinities
        B, U, V = function_variable_affinities.shape
        assert variables.dim() == 4, (
            f"Expecting a BUVC tensor, got one of " f"shape {variables.shape}."
        )
        # First, fetch the temporary variables
        # temporary_variables.shape = BUVC
        temporary_variables = self.fetch_temporary_variables(
            batch_size=variables.shape[0]
        )
        assert temporary_variables.shape[-2] == self.num_temporary_variables
        # Next, append it to the variables
        variables_with_temporary_variables = torch.cat(
            [temporary_variables, variables], dim=-2
        )
        # Finally, we need to create a new FVA
        fva_with_temporary_variables = torch.cat(
            [
                torch.ones(
                    (B, U, self.num_temporary_variables),
                    dtype=function_variable_affinities.dtype,
                    device=function_variable_affinities.device,
                ),
                function_variable_affinities,
            ],
            dim=-1,
        )
        return variables_with_temporary_variables, fva_with_temporary_variables

    def forward(
        self,
        variables: torch.Tensor,
        positional_encoding: Optional[PositionalEncoding] = None,
        function_dropout: Optional[FunctionDropout] = None,
    ):
        # variables.shape = BVC
        # The first step is to infer the types of the variables.
        # shape = BVC
        variable_types: QuantizedTensorContainer = self.type_inference(
            variables, quantize=self.quantize_variable_types
        )
        # Read out the function signatures, codes and output signatures
        (
            function_signatures,
            function_output_signatures,
            function_codes,
        ) = self.fetch_function_variables()
        # Optionally, quantize the signature of the function if required
        if self.quantize_function_signature:
            # quantized.shape = UC
            function_signatures: QuantizedTensorContainer = self.quantizer(
                function_signatures
            )
            function_output_signatures: QuantizedTensorContainer = self.quantizer(
                function_output_signatures
            )
        else:
            # quantized.shape = UC
            function_signatures = QuantizedTensorContainer(
                quantized=function_signatures,  # noqa
                commitment_loss=None,
                codebook=None,
                input=function_signatures,  # noqa
            )
            function_output_signatures = QuantizedTensorContainer(
                quantized=function_output_signatures,  # noqa
                commitment_loss=None,
                codebook=None,
                input=function_output_signatures,  # noqa
            )
        # Next, we compute the affinity between functions and variables with the kernel
        # shape = BUV
        function_variable_affinities: torch.Tensor = self.kernel(
            function_signatures.quantized, variable_types.quantized
        )
        # Normalize along functions if required
        if self.normalize_function_variable_affinities:
            function_variable_affinities = function_variable_affinities / (
                function_variable_affinities.sum(1, keepdim=True) + self.epsilon
            )
        # Evaluate LOCs. The LOCPod expects variables to have the shape BUVC, so we'll
        # have to reshape from BVC
        input_variables = variables
        variables = eo.repeat(variables, "b v c -> b u v c", u=self.num_functions)
        loc_pod_outputs = []
        for loc in self.loc_pods:
            loc_pod_output = loc(
                variables=variables,
                codes=function_codes,
                function_variable_affinities=function_variable_affinities,
                positional_encoding=positional_encoding,
            )
            variables = loc_pod_output.output
            loc_pod_outputs.append(loc_pod_output)
        # The last step is to reduce BUVC to BVC. To do so, consider that the BUVC
        # measures the contribution of function U on variable V. If a variable is
        # therefore not compatible with a function, we expect it to be eliminated
        # when it's weighted with the kernel.
        output = torch.einsum(
            "buv,buvc->bvc",
            function_variable_affinities,
            (function_dropout if function_dropout is not None else (lambda x: x))(
                variables
            ),
        )
        if not self.no_residual:
            if self.residual_mode == "gating":
                # For this gating, we let the variables through in proportion to which
                # it was required and modified by all functions. If a variable was not
                # required by any function, the gate is zero and we let the input variable
                # pass unaltered.
                # affinity_gate.shape = BV1
                affinity_gate = eo.reduce(
                    function_variable_affinities, "b u v -> b v ()", reduction="mean"
                )
                output = ((1 - affinity_gate) * input_variables) + (
                    affinity_gate * output
                )
            elif self.residual_mode == "vanilla":
                output = input_variables + output
            else:
                raise NotImplementedError
        # Evaluate the type of output coming out from the function
        output_types: QuantizedTensorContainer = self.type_inference(
            variables, quantize=self.quantize_variable_types
        )
        # Get the last output and fix up the
        return FunctionPodOutput(
            input=input_variables,
            output=output,
            loc_pod_outputs=loc_pod_outputs,
            variable_types=variable_types,
            function_signatures=function_signatures,
            function_output_signatures=function_output_signatures,
            function_variable_affinities=function_variable_affinities,
            output_types=output_types,
        )


class Script(nn.Module):
    """
    A Script a wrapper around FunctionPod that applies the latter recursively.
    This is like the main() method of a script that uses the functions as defined
    in the script.
    """

    def __init__(
        self,
        *,
        variable_features: int,
        num_iterations: int,
        no_residual: bool = True,
        # Type inference
        type_features: int,
        num_types: int,
        type_inference_kwargs: Optional[dict] = None,
        # Function pod
        function_pod_kwargs: dict,
        # Function dropout
        function_dropout_kwargs: Optional[dict] = None,
    ):
        super(Script, self).__init__()
        # Meta
        self.variable_features = variable_features
        self.num_iterations = num_iterations
        self.no_residual = no_residual
        # Make the type inference engine, which is shared between all variables
        self.type_inference = TypeInference(
            variable_features=variable_features,
            type_features=type_features,
            num_types=num_types,
            **(type_inference_kwargs or {}),
        )
        # Make the function pod
        self.function_pod = FunctionPod(
            type_inference=self.type_inference, **function_pod_kwargs
        )
        # Make the function dropout. This is a no-op by default
        self.function_dropout = FunctionDropout(**(function_dropout_kwargs or {}))

    def add_functions(self, *args, **kwargs):
        self.function_pod.add_functions(*args, **kwargs)
        return self

    def forward(
        self,
        variables: torch.Tensor,
        positional_encoding: Optional[PositionalEncoding] = None,
    ):
        # variables.shape = BVC
        input_variables = variables
        function_pod_outputs = []
        self.function_dropout.reset()
        for iter_num in range(self.num_iterations):
            function_pod_output: FunctionPodOutput = self.function_pod(
                variables,
                positional_encoding=positional_encoding,
                function_dropout=self.function_dropout,
            )
            variables = function_pod_output.output
            function_pod_outputs.append(function_pod_output)
        self.function_dropout.reset()
        if not self.no_residual:
            output = variables + input_variables
        else:
            output = variables
        # Done
        return ScriptOutput(
            input=input_variables,
            output=output,
            function_pod_outputs=function_pod_outputs,
        )


class Program(nn.Module):
    """
    A program is a stack of `Script`s.
    """

    def __init__(self, *, num_scripts: int, script_kwargs: dict):
        super(Program, self).__init__()
        self.num_scripts = num_scripts
        self.scripts = nn.ModuleList(
            [Script(**script_kwargs) for _ in range(self.num_scripts)]
        )

    def add_functions(self, *args, **kwargs):
        for script in self.scripts:
            script.add_functions(*args, **kwargs)
        return self

    def forward(
        self,
        variables: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positional_encoding: Optional[PositionalEncoding] = None,
    ):
        # variables.shape = BVC
        assert mask is None, "No support for masks as of now."
        input_variables = variables
        script_outputs = []
        for script in self.scripts:
            script_output = script(
                variables, positional_encoding=positional_encoding
            )  # type: ScriptOutput
            script_outputs.append(script_output)
            variables = script_output.output
        return ProgramOutput(
            input=input_variables, output=variables, script_outputs=script_outputs
        )
