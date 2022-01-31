import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F
import einops as eo

from neural_interpreters.utils import ModelOutput


@dataclass
class LOCPodOutput(object):
    input: torch.Tensor
    output: torch.Tensor
    # More might be added


@dataclass
class FunctionPodOutput(object):
    input: torch.Tensor
    output: torch.Tensor
    loc_pod_outputs: List["LOCPodOutput"]
    variable_types: "QuantizedTensorContainer"
    function_signatures: "QuantizedTensorContainer"
    function_output_signatures: "QuantizedTensorContainer"
    function_variable_affinities: torch.Tensor
    output_types: "QuantizedTensorContainer"


@dataclass
class ScriptOutput(object):
    input: torch.Tensor
    output: torch.Tensor
    function_pod_outputs: List["FunctionPodOutput"]


@dataclass
class ProgramOutput(object):
    input: torch.Tensor
    output: torch.Tensor
    script_outputs: List["ScriptOutput"]


@dataclass
class NeuralInterpreterOutput(ModelOutput):
    program_output: "ProgramOutput"


class FunctionDropout(nn.Module):
    def __init__(
        self,
        keep_proba: float = 1.0,
        persist_mask: bool = False,
        validation_behavior: str = "scale_down",
    ):
        super(FunctionDropout, self).__init__()
        self.keep_proba = keep_proba
        self.persist_mask = persist_mask
        assert validation_behavior in ["scale_down", "no_scaling"]
        self.validation_behavior = validation_behavior
        # Privates
        self._mask = None

    def get_mask(
        self,
        batch_size: int,
        num_functions: int,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ):
        mask = None
        # If persisting, check if a mask already exists
        if self.persist_mask and self._mask is not None:
            mask = self._mask
        if mask is None:
            # If we make it here, we'll have to make a mask
            mask = torch.distributions.Bernoulli(
                probs=torch.tensor(self.keep_proba, dtype=dtype, device=device)
            ).sample((batch_size, num_functions))
        assert mask.shape[0] == batch_size
        assert mask.shape[1] == num_functions
        if self.persist_mask:
            self._mask = mask
        return mask

    def forward(self, variables: torch.Tensor) -> torch.Tensor:
        # variables.shape = BU...
        if self.keep_proba == 1:
            # Nothing to drop here, this module is a no-op
            return variables
        if not self.training:
            # Multiply by the keep proba to preserve distribution
            if self.validation_behavior == "scale_down":
                variables = variables * self.keep_proba
            return variables
        # If we get here, we're training
        B, U, *trailing_dims = variables.shape
        # Get the mask
        mask = self.get_mask(
            batch_size=B,
            num_functions=U,
            device=variables.device,
            dtype=variables.dtype,
        ).reshape(B, U, *[1 for _ in trailing_dims])
        # Broadcast-multiply with the mask
        variables = variables * mask
        # Done
        return variables

    def reset(self):
        self._mask = None
        return self


@dataclass
class PositionalEncoding(object):
    # shape = 1VC
    positional_encoding: Optional[torch.Tensor]

    def __post_init__(self):
        if self.positional_encoding is None:
            # This is the case where there's no positional encoding to be applied.
            return
        if self.positional_encoding.dim() == 2:
            # Add in the leading extra dimension
            self.positional_encoding = self.positional_encoding[None]
        message = (
            f"Expecting a 1VC tensor, got one "
            f"of shape {list(self.positional_encoding.shape)}"
        )
        assert self.positional_encoding.dim() == 3, message
        assert self.positional_encoding.shape[0] == 1, message

    @property
    def num_positional_encoding_tokens(self):
        return self.positional_encoding.shape[1]

    def apply(
        self, tensor: torch.Tensor, reshape_sequence: Optional[str] = "auto"
    ) -> torch.Tensor:
        # tensor.shape = BUVC or BVC
        if self.positional_encoding is None:
            return tensor
        if tensor.dim() == 3 and reshape_sequence == "auto":
            # tensor.shape = BVC, there's no need for reshaping
            reshape_sequence = None
        elif tensor.dim() == 4 and reshape_sequence == "auto":
            # tensor.shape = BUVC
            reshape_sequence = "b v c -> b () v c"
        else:
            assert reshape_sequence != "auto", (
                f"Can't infer reshape sequence"
                f" automatically for {tensor.dim()}-D tensors."
            )
        if reshape_sequence is not None:
            positional_encoding = eo.rearrange(
                self.positional_encoding, reshape_sequence
            )
        else:
            positional_encoding = self.positional_encoding
        # Now, the tensor might contain more tokens than there are positional
        # encodings -- this could be due to leading cls tokens for which we don't
        # need positional encodings.
        tensor[..., -self.num_positional_encoding_tokens :, :] = (
            tensor[..., -self.num_positional_encoding_tokens :, :] + positional_encoding
        )
        return tensor


@dataclass
class QuantizedTensorContainer(object):
    quantized: torch.Tensor
    commitment_loss: Optional[torch.Tensor]
    codebook: Optional[torch.Tensor]
    input: torch.Tensor

    @property
    def output(self):
        return self.quantized


class VectorQuantizer(nn.Module):
    def __init__(self, num_features: int, num_quantities: int):
        super(VectorQuantizer, self).__init__()
        self.num_features = num_features
        self.num_quantities = num_quantities
        self.codebook = nn.Parameter(torch.randn(num_quantities, num_features))

    def forward(self, input: torch.Tensor):
        # input.shape = ...C
        # flattened.shape = NC
        # codebook.shape = MC
        ellip_shape = input.shape[0:-1]
        flattened = input.reshape(-1, input.shape[-1])
        # Normalize
        flattened = F.normalize(flattened, dim=-1)
        codebook = F.normalize(self.codebook, dim=-1)
        # Compute similarities to all tensors in the codebook.
        # shape = NM
        similarities_to_codebook = torch.einsum("nc,mc->nm", flattened, codebook)
        # Find the indices of the codes that are most similar
        # shape = N
        _, best_match_idx_in_codebook = similarities_to_codebook.max(1)
        # Replace inputs with the best match in codebook
        # shape = NC
        quantized = F.embedding(best_match_idx_in_codebook, codebook)
        # Compute commitment loss
        # shape = N
        commitment_loss = 1 - torch.cosine_similarity(
            quantized.detach(), flattened, dim=-1
        )
        # Straight through the quantized
        quantized_straight_through = flattened + (quantized - flattened).detach()
        # Reshape everything to have the right shape
        commitment_loss = commitment_loss.reshape(*ellip_shape)
        quantized_straight_through_shape = ellip_shape + (
            quantized_straight_through.shape[-1],
        )
        quantized_straight_through = quantized_straight_through.reshape(
            *quantized_straight_through_shape
        )
        return QuantizedTensorContainer(
            quantized=quantized_straight_through,
            commitment_loss=commitment_loss,
            codebook=codebook,
            input=input,
        )


class MomentumVectorQuantizer(nn.Module):
    # Based on:
    # https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    def __init__(
        self,
        num_features: int,
        num_quantities: int,
        decay: float = 0.8,
        epsilon: float = 1e-5,
    ):
        super(MomentumVectorQuantizer, self).__init__()
        self.num_features = num_features
        self.num_quantities = num_quantities
        self.decay = decay
        self.epsilon = epsilon
        embeddings = torch.randn(num_features, num_quantities)
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.zeros(num_quantities))
        self.register_buffer("averaged_embeddings", embeddings.clone())

    def forward(self, input: torch.Tensor) -> "QuantizedTensorContainer":
        # input.shape = ...C
        ellip_shape = input.shape[0:-1]
        # flattened.shape = NC
        flattened = input.reshape(-1, input.shape[-1])
        # Normalize
        flattened = F.normalize(flattened, dim=-1)
        # embedding.shape = CM
        normalized_embeddings = F.normalize(self.embeddings, dim=0)
        # Compute similarities to all tensors in embeddings.
        # similarities.shape = NM
        similarities = torch.einsum("nc,cm->nm", flattened, normalized_embeddings)
        # best_match_idx.shape = N
        _, best_match_idx = similarities.max(1)
        # quantized.shape = ...C
        quantized = F.embedding(
            best_match_idx.reshape(*ellip_shape), normalized_embeddings.transpose(0, 1)
        )
        if self.training:
            # best_match_onehot.shape = NM
            best_match_onehot = F.one_hot(best_match_idx, self.num_quantities).type(
                input.type
            )
            # This updates the size of the cluster
            self._ema_inplace(self.cluster_size, best_match_onehot.sum(0), self.decay)
            # TODO Continue from
            #  https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py#L41
            #  but give it a closer thought.
            # In particular, we want the moving average to be on the sphere.
        raise NotImplementedError

    @staticmethod
    def _ema_inplace(moving_average, new, decay):
        return moving_average.data.mul_(decay).add_(new, alpha=(1 - decay))

    @staticmethod
    def _laplace_smoothing(x, num_quantities, epsilon=1e-5):
        return (x + epsilon) / (x.sum() + num_quantities * epsilon)


class RelativePositionalEncoder(nn.Module):
    # Based on https://arxiv.org/pdf/1911.03584.pdf
    def __init__(
        self,
        num_features: int,
        image_size: int,
        num_heads: int = 1,
        share_positional_encoding_between_functions: bool = True,
        code_features: Optional[int] = None,
        function_head_mixing_mode: str = "dot",
    ):
        super(RelativePositionalEncoder, self).__init__()
        self.num_features = num_features
        self.image_size = image_size
        self.num_heads = num_heads
        self.share_positional_encoding_between_functions = (
            share_positional_encoding_between_functions
        )
        self.code_features = code_features
        assert function_head_mixing_mode in ["dot", "seq_sum", "sep_sum"], (
            f"Invalid mixing " f"mode: {function_head_mixing_mode}"
        )
        self.function_head_mixing_mode = function_head_mixing_mode
        # Init row and column embeddings
        self.row_embeddings = nn.Embedding(2 * image_size - 1, num_features)
        self.col_embeddings = nn.Embedding(2 * image_size - 1, num_features)
        # Init the projections to heads
        self.head_keys_row = nn.Linear(num_features, num_heads, bias=False)
        self.head_keys_col = nn.Linear(num_features, num_heads, bias=False)
        # Init the mod-fc's if required
        if not self.share_positional_encoding_between_functions:
            # The idea is to jointly reduce the (XX)C tensor (row/col embeddings)
            # with HC head keys and UC function keys to get a U(XX)H tensor.
            assert code_features is not None
            out_features = {
                "dot": num_features,
                "seq_sum": num_features,
                "sep_sum": num_heads,
            }[function_head_mixing_mode]
            self.function_keys_row = nn.Linear(code_features, out_features, bias=False)
            self.function_keys_col = nn.Linear(code_features, out_features, bias=False)
        else:
            self.function_keys_row, self.function_keys_col = None, None
        # Register the relative indices as buffer
        deltas = torch.arange(image_size).view(1, -1) - torch.arange(image_size).view(
            -1, 1
        )
        # shift the delta to [0, 2 * max_position_embeddings - 1]
        relative_indices = deltas.add_(image_size - 1).reshape(-1)
        self.register_buffer("relative_indices", relative_indices)

    def _mix_function_and_heads(
        self,
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor,
        codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # row_embeddings.shape = (XX)C
        # col_embeddings.shape = (XX)C
        # codes.shape = UC'
        if self.function_head_mixing_mode == "dot":
            # function_keys_row.shape = function_keys_col.shape = UC
            function_keys_row = self.function_keys_row(codes)
            function_keys_col = self.function_keys_col(codes)
            # Time to do a joint reduction
            row_scores = torch.einsum(
                "xc,hc,uc->uxh",
                row_embeddings,
                self.head_keys_row.weight,
                function_keys_row,
            )
            col_scores = torch.einsum(
                "xc,hc,uc->uxh",
                col_embeddings,
                self.head_keys_col.weight,
                function_keys_col,
            )
        elif self.function_head_mixing_mode == "seq_sum":
            # function_keys_row.shape = function_keys_col.shape = UC
            function_keys_row = self.function_keys_row(codes)
            function_keys_col = self.function_keys_col(codes)
            # function_row_embeddings.shape = U(XX)C
            function_row_embeddings = (
                row_embeddings[None, :, :] + function_keys_row[:, None, :]
            )
            function_col_embeddings = (
                col_embeddings[None, :, :] + function_keys_col[:, None, :]
            )
            # row_scores.shape = col_scores.shape = U(XX)H
            row_scores = self.head_keys_row(function_row_embeddings)
            col_scores = self.head_keys_col(function_col_embeddings)
        elif self.function_head_mixing_mode == "sep_sum":
            # function_keys_row.shape = function_keys_col.shape = U1H
            function_keys_row = self.function_keys_row(codes)[:, None, :]
            function_keys_col = self.function_keys_col(codes)[:, None, :]
            # {row,col}_pre_scores.shape = 1(XX)H
            row_pre_scores = self.head_keys_row(row_embeddings)[None]
            col_pre_scores = self.head_keys_col(col_embeddings)[None]
            # {row,col}_scores = U(XX)H
            row_scores = row_pre_scores + function_keys_row
            col_scores = col_pre_scores + function_keys_col
        else:
            raise NotImplementedError
        return row_scores, col_scores

    def forward(
        self,
        pre_softmax_attention_weights: torch.Tensor,
        codes: Optional[torch.Tensor] = None,
    ):
        # pre_softmax_attention_weights.shape = BURSH
        # code.shape = UC, if available
        # R = S = number of entities, H = number of heads.
        # Also, R = S = (X + D) * (X + D), where X = image size.
        B, U, R, S, H = pre_softmax_attention_weights.shape
        num_image_tokens = self.image_size ** 2
        assert H == self.num_heads, "Number of ehads don't match."
        assert R == S, "Only self-attention for now."
        assert R >= num_image_tokens, "Image size is wrong."
        if codes is not None:
            if codes.dim() == 4:
                # This assumes that codes is a BUVC tensor, repeated along B and V.
                codes = codes[0, :, 0, :]
            else:
                assert codes.dim() == 2, "`codes` must either be a UC or a BUVC tensor."
        # row_embeddings.shape = col_embeddings.shape = (XX)C
        row_embeddings = self.row_embeddings(self.relative_indices)
        col_embeddings = self.col_embeddings(self.relative_indices)
        if self.share_positional_encoding_between_functions:
            # Recall that head_keys_row.shape = head_keys_col.shape = CH
            # row_scores.shape = col_scores.shape = 1(XX)H
            row_scores = self.head_keys_row(row_embeddings)[None]
            col_scores = self.head_keys_col(col_embeddings)[None]
        else:
            assert codes is not None, (
                "Function codes must be provided when not sharing"
                " positional encodings among functions."
            )
            row_scores, col_scores = self._mix_function_and_heads(
                row_embeddings, col_embeddings, codes
            )
        # At this point, row_scores.shape = col_scores.shape = U(XX)H (where U can be 1)
        row_scores = eo.rearrange(
            row_scores,
            "u (x1 x2) h -> u () x1 () x2 h",
            x1=self.image_size,
            x2=self.image_size,
        )
        col_scores = eo.rearrange(
            col_scores,
            "u (x1 x2) h -> u x1 () x2 () h",
            x1=self.image_size,
            x2=self.image_size,
        )
        # attention_scores.shape = (XX)(XX)H
        attention_scores = eo.rearrange(
            row_scores + col_scores, "u i1 j1 i2 j2 h -> u (i1 j1) (i2 j2) h"
        )
        assert (
            attention_scores.shape[1] == attention_scores.shape[2] == num_image_tokens
        )
        # Make the positional delta. Encoding for the class tokens don't get anything
        positional_encoding = torch.zeros_like(pre_softmax_attention_weights)
        # This should go through only if U = 1 (shared pe) or U is the number of
        # functions. The former case means broadcasting along U.
        positional_encoding[
            :, :, -num_image_tokens:, -num_image_tokens:, :
        ] = attention_scores
        # Finally, the final output is the sum of attention and positional weights
        return pre_softmax_attention_weights + positional_encoding
